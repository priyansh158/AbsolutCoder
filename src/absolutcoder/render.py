"""Append-only terminal display for one streamed turn.

The renderer takes the ``{"type", "ns", "data"}`` parts that
``agent.stream(..., stream_mode=["messages", "updates"], subgraphs=True, version="v2")``
yields. ``ns == ()`` is the main loop; a non-empty ``ns`` is a worker.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from weakref import WeakKeyDictionary

from langchain_core.messages import AIMessage, RemoveMessage, ToolMessage
from rich.console import Console
from rich.text import Text

from absolutcoder.history import IMAGES_TRIMMED_KEY
from absolutcoder.keys import KeyListener, default_key_listener
from absolutcoder.prompting import ACCENT

ACTIVITY_MODES = ("collapsed", "full")
_TRIMMED_IMAGES_NODE = "SeenImagesMiddleware."  # history.SeenImagesMiddleware's graph nodes
COOK_LABEL = "Lemme cook…"
SHOW_HINT = "↓ show calls"
HIDE_HINT = "↑ hide calls"
SUMMARY_MARK = "●"  # a text dot; "⏺" is drawn as a blue emoji button on Windows
CALLS_HINT = "/calls to see them"
REFRESHES_PER_SECOND = 10
_EXPAND_KEYS = frozenset({"down", "right"})
_COLLAPSE_KEYS = frozenset({"up", "left"})

_OUTPUT_LOCKS: WeakKeyDictionary[Console, threading.RLock] = WeakKeyDictionary()
_OUTPUT_LOCKS_GUARD = threading.Lock()


def output_lock(console: Console) -> threading.RLock:
    """The one lock every writer to ``console`` holds, so lines from other threads stay whole.

    The renderer holds it for each write and for the open-line state; a writer
    on another thread takes it, ends any open line with the current renderer's
    ``end_line()`` and then prints its own whole line. It is re-entrant so that
    call works while the lock is held.
    """

    with _OUTPUT_LOCKS_GUARD:
        lock = _OUTPUT_LOCKS.get(console)
        if lock is None:
            lock = _OUTPUT_LOCKS[console] = threading.RLock()
        return lock

# Nodes of langchain's agent graph (main loop and workers alike).
MODEL_NODE = "model"
TOOLS_NODE = "tools"

_PATH_TOOLS = {"read_file", "write_file", "edit_file", "delete", "ls"}
_PATTERN_TOOLS = {"glob", "grep"}
_SUMMARY_KEYS = {"execute": ("command",), "web_fetch": ("url",), "web_search": ("query",)}


def message_text(content) -> str:
    """Return the visible text of message content: text blocks only."""

    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(part for part in _text_parts(content) if part).strip()
    return ""


def _text_parts(content: list) -> list[str]:
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return parts


def _chunk_text(content) -> str:
    """Like ``message_text`` but unstripped, so streamed tokens join correctly."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_text_parts(content))
    return ""


def _one_line(text: str, limit: int) -> str:
    flat = " ".join(str(text).split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1] + "…"


def _figure(value) -> int:
    """A token count as a non-negative int; anything non-numeric counts as 0."""

    if isinstance(value, bool):
        return 0
    if not isinstance(value, (int, float)):
        return 0
    try:
        return max(int(value), 0)
    except (ValueError, OverflowError):  # nan, infinity
        return 0


def summarize_call(name: str, args: dict) -> str:
    """A one-line, at most 80-character summary of a tool call's arguments."""

    if not isinstance(args, dict):
        args = {}
    if name in _PATH_TOOLS:
        keys: tuple[str, ...] = ("file_path", "path")
    elif name in _PATTERN_TOOLS:
        keys = ("pattern",)
    else:
        keys = _SUMMARY_KEYS.get(name, ())
    for key in keys:
        if key in args:
            return _one_line(str(args[key]), 80)
    try:
        text = json.dumps(args, separators=(",", ":"), ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(args)
    return _one_line(text, 80)


def format_elapsed(seconds: float) -> str:
    """``12s``, ``1m 04s`` or ``1h 02m 03s``."""

    total = max(int(seconds), 0)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _count(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _clip(text: str, cells: int, style: str = "") -> Text:
    """``text`` on one line, cut with an ellipsis to at most ``cells`` columns."""

    line = Text(" ".join(str(text).split()), style=style, no_wrap=True, overflow="ellipsis")
    line.truncate(max(cells, 1), overflow="ellipsis")
    return line


def _finish_interrupted_stop(region) -> None:  # noqa: ANN001 - a rich Live
    """Undo what a ``Live.stop()`` cut short by Ctrl+C can leave behind.

    ``stop()`` marks the region stopped first, so calling it again does
    nothing. This stops the refresh thread, takes the region off the console's
    live stack and render hooks, erases it and shows the cursor again. It uses
    rich's private fields, each only when present.
    """

    console = region.console
    with region._lock:  # the refresh thread checks its stop flag under this lock
        thread = getattr(region, "_refresh_thread", None)
        if thread is not None:
            thread.stop()
            region._refresh_thread = None
        for name in ("_live_stack", "_render_hooks"):
            stack = getattr(console, name, None)
            if isinstance(stack, list) and region in stack:
                stack.remove(region)
        live_render = getattr(region, "_live_render", None)
        if region.transient and live_render is not None and live_render.last_render_height:
            console.control(live_render.restore_cursor())
        console.show_cursor(True)


class _StatusView:
    """The live region's renderable: rebuilt on every refresh so the spinner and clock move."""

    def __init__(self, renderer: StreamRenderer) -> None:
        self._renderer = renderer

    def __rich_console__(self, console, options):  # noqa: ANN001, ANN204 - rich protocol
        yield from self._renderer.status_lines(options.max_width)


class StreamRenderer:
    """Print the main model's text, tool calls and results, and worker activity.

    After ``start(activity="collapsed")`` on a real terminal, tool calls,
    results and worker lines go only to ``call_log`` while one live status
    line (with a spinner) stays at the bottom; the arrow keys show and hide
    the log. Main text is then printed as whole messages above that line.
    Everywhere else every line is printed as it happens; ``call_log`` is
    kept either way.
    """

    def __init__(
        self,
        console: Console,
        *,
        live: bool | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.console = console
        self._output = output_lock(console)
        self.live = console.is_terminal if live is None else live
        self.call_log: list[str] = []  # every tool, result, task and worker line of this turn
        self._clock = clock
        self._view_lock = threading.RLock()
        self._collapsed = False  # lines go to the log, not the screen
        self._cooking = False  # the collapsed view was used this turn: a summary is due
        self._summary_shown = False
        self._live_region = None  # the running rich Live, if any
        self._keys: KeyListener | None = None
        self._keys_running = False
        self._expanded = False
        self._pause_depth = 0
        self._started_at: float | None = None
        self._call_count = 0
        self._error_count = 0
        self._pending_error: Exception | None = None  # raised while drawing, shown later
        self._drawing_failed = False
        self._unshown_text = ""  # main text streamed while collapsed, not on screen yet
        self._spinner = None
        self.errors: list[Exception] = []
        self.tools_used: set[str] = set()  # names of the tools called this turn, main loop and workers
        self._labels: dict[tuple, str] = {}
        self._calls: dict[str, dict] = {}  # tool_call_id -> most recent call with that id
        self._shown_ids: set[str] = set()  # main message ids already on screen
        self._shown_calls: set[tuple] = set()  # (ns, AI message id, index of the call)
        self._shown_results: set[str] = set()  # ToolMessage ids
        self._usage_by_id: dict[str, tuple[int, int, int]] = {}  # message id -> (input, cached, output)
        self._anonymous_usage: list[tuple[int, int, int]] = []  # messages without an id
        self._streamed = ""  # text streamed for the current main message
        self._last_text: str | None = None
        self._open_line = False

    # --- the collapsed view ----------------------------------------------------

    def start(self, *, activity: str = "collapsed", key_listener: KeyListener | None = None) -> bool:
        """Begin the collapsed view when ``activity`` asks for it and this is a real terminal.

        Returns whether it began. If the live region or the key listener cannot
        start, the turn is shown in full, exactly as without this call.
        """

        if activity != "collapsed" or self._cooking:
            return False
        if not self.console.is_terminal or self.console.is_dumb_terminal:
            return False
        try:
            self._keys = default_key_listener() if key_listener is None else key_listener
            self._started_at = self._clock()
        except Exception:  # noqa: BLE001 - no view: the turn is shown in full
            return False
        if not self._open_view():
            return False
        self._cooking = True
        return True

    @property
    def collapsed(self) -> bool:
        """Whether lines go to the call log instead of the screen right now."""

        return self._collapsed

    @property
    def expanded(self) -> bool:
        return self._expanded

    def pause(self) -> None:
        """Take the live region and key listener down, e.g. for a prompt; ``resume`` brings them back."""

        with self._view_lock:
            self._pause_depth += 1
            first = self._pause_depth == 1
        if first and self._collapsed:
            self._close_view()

    def resume(self, *, reopen: bool = True) -> None:
        """Undo one ``pause``; the last one brings the view back unless ``reopen`` is false."""

        with self._view_lock:
            self._pause_depth = max(self._pause_depth - 1, 0)
            last = self._pause_depth == 0
        if last and reopen and self._collapsed and self._live_region is None:
            if not self._open_view():
                self._collapsed = False  # the rest of the turn is shown in full

    @contextmanager
    def paused(self) -> Iterator[None]:
        """Pause for the body; an exception (Ctrl+C at the prompt) leaves the view down."""

        self.pause()
        completed = False
        try:
            yield
            completed = True
        finally:
            self.resume(reopen=completed)

    def stop_view(self) -> None:
        """Stop the live region and key listener for good; safe to call more than once."""

        self._close_view()
        self._collapsed = False

    def abort(self) -> None:
        """The turn stopped on an error: end the view and any open line, the summary, then cut-off text."""

        self._end_view()
        self._show_unshown_text()

    def _show_unshown_text(self) -> None:
        """Print main text that streamed while collapsed but whose message never completed."""

        text, self._unshown_text = self._unshown_text.strip(), ""
        if text:
            self._print_text(None, text)

    def status_lines(self, width: int | None = None) -> list[Text]:
        """The live region's lines: the status line, then the newest log lines when expanded."""

        try:
            return self._status_lines(self.console.width if width is None else width)
        except Exception as error:  # noqa: BLE001 - drawing must never break a turn
            self._record_later(error)
            return [Text(COOK_LABEL, no_wrap=True)]

    def _status_lines(self, width: int) -> list[Text]:
        limit = max(width - 1, 1)
        now = self._clock()
        if self._spinner is None:
            from rich.spinner import Spinner  # noqa: PLC0415 - only the live view needs it

            self._spinner = Spinner("dots", style=ACCENT)
        frame = self._spinner.render(now)
        frame = frame if isinstance(frame, Text) else Text(str(frame))
        started = self._started_at if self._started_at is not None else now
        line = Text(no_wrap=True, overflow="ellipsis")
        line.append_text(frame)
        line.append(f" {COOK_LABEL} {format_elapsed(now - started)} · {_count(self._call_count, 'call')}")
        if self._error_count:
            line.append(f" · {_count(self._error_count, 'error')}", style="red")
        hint = "   " + (HIDE_HINT if self._expanded else SHOW_HINT)
        log = self.call_log
        last = log[-1] if log else ""
        room = limit - line.cell_len - len(" · ") - Text(hint).cell_len
        if last and room >= 4:
            line.append(" · ")
            line.append_text(_clip(last, room, "dim"))
        line.append(hint, style="dim")
        line.truncate(limit, overflow="ellipsis")
        lines = [line]
        if self._expanded:
            count = max(3, min(20, self.console.height - 6))
            lines.extend(_clip(entry, limit, "dim") for entry in log[-count:])
        return lines

    def on_key(self, key: str) -> None:
        """Arrow down/right shows the call log, up/left hides it; redrawn at once."""

        if key in _EXPAND_KEYS:
            self._expanded = True
        elif key in _COLLAPSE_KEYS:
            self._expanded = False
        else:
            return
        with self._view_lock:
            region = self._live_region
            if region is None or not region.is_started:
                return  # a refresh after stop() would draw the region again, for good
            try:
                region.refresh()
            except Exception as error:  # noqa: BLE001 - drawing must never break a turn
                self._record_later(error)

    def _open_view(self) -> bool:
        """Start the live region and the key listener; on any failure neither runs."""

        from rich.live import Live  # noqa: PLC0415

        with self._view_lock:
            region = None
            try:
                region = Live(
                    _StatusView(self),
                    console=self.console,
                    transient=True,
                    auto_refresh=True,
                    refresh_per_second=REFRESHES_PER_SECOND,
                    redirect_stdout=False,  # the console already writes through sys.stdout
                    redirect_stderr=False,
                )
                region.start()
                self._live_region = region
                if self._keys is not None:
                    self._keys.start(self.on_key)
                    self._keys_running = True
            except Exception:  # noqa: BLE001 - no view: the turn is shown in full
                self._live_region = None
                if region is not None:
                    try:
                        region.stop()
                    except Exception:  # noqa: BLE001
                        pass
                return False
            self._collapsed = True
            return True

    def _close_view(self) -> None:
        """Stop the key listener, then the live region (which erases itself).

        Ordinary errors are recorded, never raised. Ctrl+C while the listener
        stops still stops the region before it propagates, and the region is
        forgotten only once its own stop() has finished, so a later call can
        stop it if that was interrupted too.
        """

        with self._view_lock:
            keys = self._keys if self._keys_running else None
            self._keys_running = False
        try:
            if keys is not None:
                try:
                    keys.stop()
                except Exception as error:  # noqa: BLE001
                    self._record_later(error)
        finally:
            with self._view_lock:  # no key refresh can run between stop() and forgetting it
                region = self._live_region
                if region is not None:
                    try:
                        if region.is_started:
                            region.stop()
                        else:
                            _finish_interrupted_stop(region)  # an earlier stop() was cut short
                    except Exception as error:  # noqa: BLE001
                        self._record_later(error)
                    self._live_region = None

    def _end_view(self) -> None:
        """Stop the view, end any open line, and print the summary once if the view was used."""

        self.stop_view()
        self.end_line()
        self._flush_pending_error()
        if not self._cooking or self._summary_shown:
            return
        self._summary_shown = True
        now = self._clock()
        elapsed = now - (now if self._started_at is None else self._started_at)
        text = f" Cooked for {format_elapsed(elapsed)}"
        if self._call_count:
            text += f" · {_count(self._call_count, 'call')} · {CALLS_HINT}"
        line = Text(SUMMARY_MARK)  # the dot in the normal text colour, the rest dim
        line.append(text, style="dim")
        self._print(line)

    def _record_later(self, error: Exception) -> None:
        """Record an error raised while drawing; its line is printed from the main thread."""

        if self._drawing_failed:
            return  # the same error on every refresh: one record is enough
        self._drawing_failed = True
        self.errors.append(error)
        if len(self.errors) == 1:
            self._pending_error = error

    def _flush_pending_error(self) -> None:
        error, self._pending_error = self._pending_error, None
        if error is not None:
            try:
                self._print(f"display error: {type(error).__name__}: {error}")
            except Exception:  # noqa: BLE001 - the console itself failed
                pass

    # --- public ---------------------------------------------------------------

    def feed(self, part: dict) -> None:
        """Render one part; never raises. A renderer bug is recorded in ``errors``."""

        try:
            if self._pending_error is not None:
                self._flush_pending_error()
            if not isinstance(part, dict):
                return
            ns = part.get("ns") or ()
            ns = tuple(ns) if isinstance(ns, (tuple, list)) else ()
            kind = part.get("type")
            if kind == "messages":
                self._on_message(ns, part.get("data"))
            elif kind == "updates":
                self._on_update(ns, part.get("data"))
        except Exception as error:  # noqa: BLE001 - the display must never break a turn
            self._record(error)

    def finish(self, snapshot) -> None:
        self._end_view()
        self._unshown_text = ""  # every message completed; the snapshot below has the answer
        values = getattr(snapshot, "values", None)
        messages = values.get("messages", []) if isinstance(values, dict) else []
        for message in reversed(messages or []):
            if not isinstance(message, AIMessage) or message.tool_calls:
                continue
            text = message_text(message.content)
            shown = (message.id and message.id in self._shown_ids) or (
                text and text == self._last_text
            )
            if not shown:
                self._print_text(message.id, text)
            break
        self.end_line()
        self._print_usage()

    def interrupted(self) -> None:
        self._end_view()
        self._show_unshown_text()
        self._print("interrupted")
        self._print_usage()

    def end_line(self) -> None:
        """End a streamed line that has no newline yet."""

        with self._output:
            if self._open_line:
                self._open_line = False
                self.console.print(markup=False, highlight=False)

    # --- parts ------------------------------------------------------------------

    def _on_message(self, ns: tuple, data) -> None:
        if not isinstance(data, (tuple, list)) or len(data) != 2:
            return
        message, metadata = data
        # Usage is never counted from streamed chunks: see _count_usage.
        if not isinstance(metadata, dict):
            return
        if ns:
            name = metadata.get("lc_agent_name")
            if isinstance(name, str) and name:
                self._labels[ns] = name
            return  # worker text is never streamed
        if metadata.get("langgraph_node") != MODEL_NODE:
            return
        if metadata.get("lc_source") == "summarization" or not isinstance(message, AIMessage):
            return
        if self._collapsed:
            # The collapsed view prints whole messages only (a partial line would break
            # it); the text is kept so an interrupted message is not lost.
            self._unshown_text += _chunk_text(message.content)
            return
        if not self.live:
            return
        text = _chunk_text(message.content)
        if not text:
            return
        with self._output:
            self._open_line = True
            self.console.print(text, end="", markup=False, highlight=False, soft_wrap=True)
        if message.id:
            self._shown_ids.add(message.id)
        self._streamed += text
        self._last_text = self._streamed.strip()

    def _on_update(self, ns: tuple, data) -> None:
        if not isinstance(data, dict):
            return
        for node, output in data.items():
            if node == "__interrupt__" or not isinstance(output, dict):
                continue
            messages = output.get("messages")
            if not isinstance(messages, list):
                messages = [messages] if messages is not None else []
            if str(node).startswith(_TRIMMED_IMAGES_NODE):
                continue  # earlier results with their images trimmed: history, not activity
            rewrite = any(isinstance(m, RemoveMessage) for m in messages)
            if rewrite and node != TOOLS_NODE:
                continue  # a history replay (patching, summarisation), not new activity
            for message in messages:
                if isinstance(message, ToolMessage):
                    self._on_result(ns, message)
                elif isinstance(message, AIMessage) and not rewrite:
                    try:
                        if node == MODEL_NODE:
                            self._on_ai(ns, message)
                    finally:
                        self._count_usage(message)

    def _on_ai(self, ns: tuple, message: AIMessage) -> None:
        if not ns:
            self._on_main_text(message)
        for index, call in enumerate(message.tool_calls or []):
            key = (ns, message.id, index) if message.id else None
            if key is not None:
                if key in self._shown_calls:
                    continue
                self._shown_calls.add(key)
            self._on_call(ns, call)

    def _on_main_text(self, message: AIMessage) -> None:
        self._unshown_text = ""  # the completed message is shown (or already was) below
        text = message_text(message.content)
        shown = (
            (message.id and message.id in self._shown_ids)
            or (text and text == self._streamed.strip())
            or (not message.id and text and text == self._last_text)
        )
        if shown:
            if message.id:
                self._shown_ids.add(message.id)
        else:
            self._print_text(message.id, text)
        self._streamed = ""

    def _count_usage(self, message) -> None:
        """Record the token usage of one completed AI message, once per message id.

        Only messages in ``updates`` parts are counted. Streamed chunks in
        ``messages`` parts carry a ``lc_run--…`` id where the completed message
        keeps the provider's id (``resp_…``, ``chatcmpl-…``), so the two cannot be
        matched and counting both would double the figures. As a result, an
        internal model call whose result never appears as a message in an
        ``updates`` part (summarisation) is not counted. Never raises.
        """

        try:
            if not isinstance(message, AIMessage):
                return
            usage = message.usage_metadata
            if not isinstance(usage, dict):
                return
            details = usage.get("input_token_details")
            inp = _figure(usage.get("input_tokens"))
            cached = _figure(details.get("cache_read") if isinstance(details, dict) else 0)
            figures = (inp, min(cached, inp), _figure(usage.get("output_tokens")))
            self._store_usage(message.id, figures)
        except Exception as error:  # noqa: BLE001 - counting must never hide a message
            self._record(error)

    def _store_usage(self, message_id, figures: tuple[int, int, int]) -> None:
        if not message_id:
            self._anonymous_usage.append(figures)
            return
        seen = self._usage_by_id.get(message_id)
        if seen is not None:
            figures = tuple(max(old, new) for old, new in zip(seen, figures))
        self._usage_by_id[message_id] = figures

    @property
    def usage(self) -> dict[str, int]:
        """Token totals for this turn: input, cached (part of input), output, model calls."""

        rows = [*self._usage_by_id.values(), *self._anonymous_usage]
        return {
            "input": sum(row[0] for row in rows),
            "cached": sum(row[1] for row in rows),
            "output": sum(row[2] for row in rows),
            "calls": len(rows),
        }

    def _print_usage(self) -> None:
        usage = self.usage
        if usage["calls"] == 0 or usage["input"] + usage["output"] == 0:
            return
        calls = usage["calls"]
        self._print(
            f"tokens: {usage['input']:,} in ({usage['cached']:,} cached), "
            f"{usage['output']:,} out, {calls} model call{'' if calls == 1 else 's'}"
        )

    def _on_call(self, ns: tuple, call: dict) -> None:
        call_id = call.get("id")
        if call_id:
            self._calls[call_id] = call
        name = str(call.get("name", ""))
        if name:
            self.tools_used.add(name)
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        self._call_count += 1
        if ns:
            self._activity(f"  [{self._label(ns)}] tool: {name}({summarize_call(name, args)})")
        elif name == "task":
            role = args.get("subagent_type", "worker")
            description = _one_line(str(args.get("description", "")), 100)
            self._activity(f"  task [{role}]: {description}")
        else:
            self._activity(f"  tool: {name}({summarize_call(name, args)})")

    def _on_result(self, ns: tuple, message: ToolMessage) -> None:
        extra = message.additional_kwargs if isinstance(message.additional_kwargs, dict) else {}
        if extra.get(IMAGES_TRIMMED_KEY):
            return  # a result shown in an earlier call, rewritten without its images
        if message.id:
            if message.id in self._shown_results:
                return
            self._shown_results.add(message.id)
        call = self._calls.get(message.tool_call_id, {})
        name = message.name or str(call.get("name", "")) or "tool"
        if not ns and name == "task":
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            self._activity(f"  task [{args.get('subagent_type', 'worker')}] finished")
            return
        prefix = f"[{self._label(ns)}] " if ns else ""
        if message.status == "error":
            self._error_count += 1
            lines = message_text(message.content).splitlines() or [""]
            self._activity(f"  {prefix}error: {name}: {_one_line(lines[0], 120)}")
        else:
            self._activity(f"  {prefix}done: {name}")

    # --- output -----------------------------------------------------------------

    def _record(self, error: Exception) -> None:
        self.errors.append(error)
        if len(self.errors) > 1:
            return
        try:
            self._line(f"display error: {type(error).__name__}: {error}")
        except Exception:  # noqa: BLE001 - the console itself failed; nothing more to show
            self._open_line = False

    def _label(self, ns: tuple) -> str:
        return self._labels.get(ns, "worker")

    def _print_text(self, message_id: str | None, text: str) -> None:
        if message_id:
            self._shown_ids.add(message_id)
        if text:
            self._line(text)
            self._last_text = text

    def _activity(self, text: str) -> None:
        """A tool, result, task or worker line: logged always, printed unless collapsed."""

        self.call_log.append(text)
        if not self._collapsed:
            self._line(text)
            return
        with self._output:
            # As _line does: the logged line ends the text a repeated answer is compared with.
            self._streamed = ""
            self._last_text = None

    def _line(self, text: str) -> None:
        with self._output:
            self.end_line()
            self._streamed = ""
            # Any line ends the text a repeated answer is compared with;
            # _print_text sets it again after a text line.
            self._last_text = None
            self._print(text)

    def _print(self, text: str | Text, *, style: str | None = None) -> None:
        with self._output:
            # Off a terminal (a pipe or a file) a whole line is never hard-wrapped.
            soft_wrap = not self.console.is_terminal
            self.console.print(text, markup=False, highlight=False, soft_wrap=soft_wrap, style=style)

