"""Append-only terminal display for one streamed turn.

The renderer takes the ``{"type", "ns", "data"}`` parts that
``agent.stream(..., stream_mode=["messages", "updates"], subgraphs=True, version="v2")``
yields. ``ns == ()`` is the main loop; a non-empty ``ns`` is a worker.
"""

from __future__ import annotations

import json
import threading
from weakref import WeakKeyDictionary

from langchain_core.messages import AIMessage, RemoveMessage, ToolMessage
from rich.console import Console

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


class StreamRenderer:
    """Print the main model's text, tool calls and results, and worker activity."""

    def __init__(self, console: Console, *, live: bool | None = None) -> None:
        self.console = console
        self._output = output_lock(console)
        self.live = console.is_terminal if live is None else live
        self.errors: list[Exception] = []
        self._labels: dict[tuple, str] = {}
        self._calls: dict[str, dict] = {}  # tool_call_id -> most recent call with that id
        self._shown_ids: set[str] = set()  # main message ids already on screen
        self._shown_calls: set[tuple] = set()  # (ns, AI message id, index of the call)
        self._shown_results: set[str] = set()  # ToolMessage ids
        self._streamed = ""  # text streamed for the current main message
        self._last_text: str | None = None
        self._open_line = False

    # --- public ---------------------------------------------------------------

    def feed(self, part: dict) -> None:
        """Render one part; never raises. A renderer bug is recorded in ``errors``."""

        try:
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

    def interrupted(self) -> None:
        self.end_line()
        self._print("interrupted")

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
        if not isinstance(metadata, dict):
            return
        if ns:
            name = metadata.get("lc_agent_name")
            if isinstance(name, str) and name:
                self._labels[ns] = name
            return  # worker text is never streamed
        if not self.live or metadata.get("langgraph_node") != MODEL_NODE:
            return
        if metadata.get("lc_source") == "summarization" or not isinstance(message, AIMessage):
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
            rewrite = any(isinstance(m, RemoveMessage) for m in messages)
            if rewrite and node != TOOLS_NODE:
                continue  # a history replay (patching, summarisation), not new activity
            for message in messages:
                if isinstance(message, ToolMessage):
                    self._on_result(ns, message)
                elif isinstance(message, AIMessage) and node == MODEL_NODE and not rewrite:
                    self._on_ai(ns, message)

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

    def _on_call(self, ns: tuple, call: dict) -> None:
        call_id = call.get("id")
        if call_id:
            self._calls[call_id] = call
        name = str(call.get("name", ""))
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        if ns:
            self._line(f"  [{self._label(ns)}] tool: {name}({summarize_call(name, args)})")
        elif name == "task":
            role = args.get("subagent_type", "worker")
            description = _one_line(str(args.get("description", "")), 100)
            self._line(f"  task [{role}]: {description}")
        else:
            self._line(f"  tool: {name}({summarize_call(name, args)})")

    def _on_result(self, ns: tuple, message: ToolMessage) -> None:
        if message.id:
            if message.id in self._shown_results:
                return
            self._shown_results.add(message.id)
        call = self._calls.get(message.tool_call_id, {})
        name = message.name or str(call.get("name", "")) or "tool"
        if not ns and name == "task":
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            self._line(f"  task [{args.get('subagent_type', 'worker')}] finished")
            return
        prefix = f"[{self._label(ns)}] " if ns else ""
        if message.status == "error":
            lines = message_text(message.content).splitlines() or [""]
            self._line(f"  {prefix}error: {name}: {_one_line(lines[0], 120)}")
        else:
            self._line(f"  {prefix}done: {name}")

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

    def _line(self, text: str) -> None:
        with self._output:
            self.end_line()
            self._streamed = ""
            self._print(text)

    def _print(self, text: str) -> None:
        with self._output:
            # Off a terminal (a pipe or a file) a whole line is never hard-wrapped.
            soft_wrap = not self.console.is_terminal
            self.console.print(text, markup=False, highlight=False, soft_wrap=soft_wrap)

