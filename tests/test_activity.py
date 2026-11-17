"""The collapsed "Lemme cook" view: one live status line, arrow keys, the summary and /calls."""

import io
import re
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from rich.console import Console

from absolutcoder.config import Config
from absolutcoder.keys import NoKeys, PosixDecoder, decode_posix, decode_windows, default_key_listener
from absolutcoder.render import COOK_LABEL, StreamRenderer, format_elapsed
from tests.support import role_models, scripted_model, tool_call, tool_names_since_last_human

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07")
WORKER_NS = ("tools:abc",)


class FakeKeys:
    """Records start/stop and lets a test press keys."""

    def __init__(self, *, fail=False):
        self.events: list[str] = []
        self.on_key = None
        self.fail = fail

    def start(self, on_key):
        if self.fail:
            raise OSError("no keyboard")
        self.events.append("start")
        self.on_key = on_key

    def stop(self):
        self.events.append("stop")
        self.on_key = None

    def press(self, key):
        assert self.on_key is not None, "listener is not running"
        self.on_key(key)


class FakeClock:
    def __init__(self, now=100.0):
        self.now = now

    def __call__(self):
        return self.now


def _terminal(width=100, height=30):
    output = io.StringIO()
    console = Console(file=output, force_terminal=True, width=width, height=height, legacy_windows=False)
    return console, output


def _plain_console():
    output = io.StringIO()
    return Console(file=output, force_terminal=False, no_color=True, width=120), output


def _text(output) -> str:
    """Everything written, without escape codes; a carriage return starts a new line."""

    return _ANSI.sub("", output.getvalue().replace("\r", "\n"))


def _printed(output) -> list[str]:
    """The lines that stay on screen: the live region's frames (erased on a terminal) left out."""

    return [line.rstrip() for line in _text(output).splitlines() if line.strip() and COOK_LABEL not in line]


def _upd(node, messages, ns=()):
    return {"type": "updates", "ns": ns, "data": {node: {"messages": messages}}}


def _call(name, args, call_id, *, message_id=None, content=""):
    return AIMessage(
        content=content,
        id=message_id or "ai-" + call_id,
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def _snapshot(*messages):
    return SimpleNamespace(values={"messages": list(messages)})


def _feed_activity(renderer):
    """Two main tool calls (one fails) and one worker call, with a narration line."""

    first = _call("read_file", {"file_path": "/src/app.py"}, "c1", content="Let me look.")
    renderer.feed(_upd("model", [first]))
    renderer.feed(_upd("tools", [ToolMessage(content="text", tool_call_id="c1", name="read_file", id="t1")]))
    second = _call("execute", {"command": "pytest -q"}, "c2")
    renderer.feed(_upd("model", [second]))
    failed = ToolMessage(content="boom\nmore", tool_call_id="c2", name="execute", id="t2", status="error")
    renderer.feed(_upd("tools", [failed]))
    worker = _call("grep", {"pattern": "TODO"}, "w1")
    renderer.feed(_upd("model", [worker], ns=WORKER_NS))
    renderer.feed(_upd("tools", [ToolMessage(content="x", tool_call_id="w1", name="grep", id="t3")], ns=WORKER_NS))


_EXPECTED_LOG = [
    "  tool: read_file(/src/app.py)",
    "  done: read_file",
    "  tool: execute(pytest -q)",
    "  error: execute: boom",
    "  [worker] tool: grep(TODO)",
    "  [worker] done: grep",
]


def _collapsed(width=100, height=30, clock=None):
    console, output = _terminal(width, height)
    keys = FakeKeys()
    renderer = StreamRenderer(console, clock=clock or FakeClock())
    assert renderer.start(activity="collapsed", key_listener=keys) is True
    return renderer, output, keys


def test_collapsed_mode_hides_tool_lines_and_logs_them():
    renderer, output, keys = _collapsed()
    try:
        _feed_activity(renderer)
        printed = _printed(output)
        assert not any(word in line for line in printed for word in ("tool:", "done:", "error:"))
        assert "Let me look." in printed  # main text is still printed, as a whole line
        assert renderer.call_log == _EXPECTED_LOG
    finally:
        renderer.stop_view()
    assert renderer.errors == []


def test_collapsed_mode_prints_summary_and_answer_once():
    clock = FakeClock(100.0)
    renderer, output, keys = _collapsed(clock=clock)
    _feed_activity(renderer)
    final = AIMessage(content="All done here.", id="final")
    renderer.feed(_upd("model", [final]))
    final.usage_metadata = None
    clock.now = 172.0
    done = AIMessage(content="All done here.", id="final", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
    renderer.feed(_upd("model", [done]))
    renderer.finish(_snapshot(done))
    lines = _printed(output)
    assert "⏺ Cooked for 1m 12s · 3 calls · /calls to see them" in lines
    assert _text(output).count("All done here.") == 1
    assert any(line.startswith("tokens: 10 in") for line in lines)
    assert lines.index("⏺ Cooked for 1m 12s · 3 calls · /calls to see them") < next(
        i for i, line in enumerate(lines) if line.startswith("tokens:")
    )
    assert keys.events == ["start", "stop"]
    assert renderer.errors == []


def test_final_answer_only_in_the_snapshot_is_printed_after_the_summary():
    renderer, output, _keys = _collapsed()
    renderer.finish(_snapshot(AIMessage(content="late answer", id="late")))
    assert _printed(output) == ["⏺ Cooked for 0s", "late answer"]  # no calls: no count and no hint


def test_summary_counts_one_call_in_the_singular():
    renderer, output, _keys = _collapsed()
    renderer.feed(_upd("model", [_call("ls", {"path": "/"}, "c1")]))
    renderer.finish(_snapshot())
    assert "⏺ Cooked for 0s · 1 call · /calls to see them" in _text(output)


def test_arrow_keys_expand_and_collapse_the_live_region():
    clock = FakeClock(0.0)
    renderer, _output, keys = _collapsed(width=80, height=30, clock=clock)
    try:
        _feed_activity(renderer)
        clock.now = 12.0
        collapsed = [line.plain for line in renderer.status_lines(80)]
        assert len(collapsed) == 1
        status = collapsed[0]
        assert "Lemme cook… 12s · 3 calls · 1 error · " in status
        assert status.endswith("   ↓ show calls")
        assert "[worker] done: grep" in status  # the newest log line
        assert len(status) <= 79
        keys.press("down")
        expanded = [line.plain for line in renderer.status_lines(80)]
        assert expanded[0].endswith("   ↑ hide calls")
        assert expanded[1:] == [line.strip() for line in _EXPECTED_LOG]
        keys.press("up")
        assert [line.plain for line in renderer.status_lines(80)][0].endswith("↓ show calls")
        keys.press("right")
        assert renderer.expanded is True
        keys.press("left")
        assert renderer.expanded is False
        keys.press("x")  # any other key changes nothing
        assert renderer.expanded is False
    finally:
        renderer.stop_view()
    assert renderer.errors == []


def test_expanded_region_shows_only_what_fits_and_never_wraps():
    renderer, _output, keys = _collapsed(width=40, height=10)
    try:
        for index in range(30):
            renderer.feed(_upd("model", [_call("execute", {"command": f"echo {index} " + "x" * 80}, f"c{index}")]))
        keys.press("down")
        lines = renderer.status_lines(40)
        assert len(lines) == 1 + 4  # max(3, min(20, height - 6))
        assert all(line.cell_len <= 39 for line in lines)
        assert lines[-1].plain.startswith("tool: execute(echo 29")
        assert lines[1].plain.startswith("tool: execute(echo 26")
        tiny = StreamRenderer(_terminal(width=40, height=5)[0], clock=FakeClock())
        tiny.call_log.extend(f"line {n}" for n in range(10))
        tiny._expanded = True
        assert len(tiny.status_lines(40)) == 1 + 3
    finally:
        renderer.stop_view()


def test_live_region_draws_and_erases_on_the_terminal():
    renderer, output, keys = _collapsed()
    renderer.feed(_upd("model", [_call("ls", {"path": "/"}, "c1")]))
    keys.press("down")  # a refresh on toggle, drawn at once
    drawn = _text(output)
    assert "Lemme cook…" in drawn and "↑ hide calls" in drawn
    renderer.finish(_snapshot())
    assert "Cooked for" in _text(output)


@pytest.mark.parametrize("make", ["non_terminal", "full_setting"])
def test_non_terminal_and_full_setting_keep_todays_lines(make):
    if make == "non_terminal":
        console, output = _plain_console()
        renderer = StreamRenderer(console, live=False)
        assert renderer.start(activity="collapsed", key_listener=FakeKeys()) is False
    else:
        console, output = _terminal()
        keys = FakeKeys()
        renderer = StreamRenderer(console, live=False)
        assert renderer.start(activity="full", key_listener=keys) is False
        assert keys.events == []
    _feed_activity(renderer)
    renderer.finish(_snapshot())
    lines = _text(output).splitlines()
    for line in _EXPECTED_LOG:
        assert line in lines
    assert "Cooked for" not in _text(output)
    assert "Lemme cook" not in _text(output)
    assert renderer.call_log == _EXPECTED_LOG  # /calls works in the full view too


def test_dumb_terminal_keeps_todays_lines(monkeypatch):
    monkeypatch.setenv("TERM", "dumb")
    console, output = _terminal()
    assert console.is_dumb_terminal
    renderer = StreamRenderer(console, live=False)
    assert renderer.start(activity="collapsed", key_listener=FakeKeys()) is False


def test_live_failure_falls_back_to_full_mode(monkeypatch):
    console, output = _terminal()
    keys = FakeKeys(fail=True)
    renderer = StreamRenderer(console, live=False)
    assert renderer.start(activity="collapsed", key_listener=keys) is False
    assert renderer.collapsed is False
    _feed_activity(renderer)
    renderer.finish(_snapshot())
    text = _text(output)
    assert "  tool: read_file(/src/app.py)" in text.splitlines()
    assert "Cooked for" not in text
    assert renderer.errors == []

    import rich.live

    def broken_start(self, refresh=False):
        raise RuntimeError("no live display")

    monkeypatch.setattr(rich.live.Live, "start", broken_start)
    console, output = _terminal()
    keys = FakeKeys()
    renderer = StreamRenderer(console, live=False)
    assert renderer.start(activity="collapsed", key_listener=keys) is False
    assert keys.events == []  # the listener never started
    _feed_activity(renderer)
    assert "  done: read_file" in _text(output).splitlines()


def test_a_drawing_error_is_recorded_and_shown_once():
    renderer, output, _keys = _collapsed()

    def broken(width):
        raise ValueError("bad draw")

    renderer._status_lines = broken
    assert [line.plain for line in renderer.status_lines(80)] == ["Lemme cook…"]
    renderer.status_lines(80)
    renderer.finish(_snapshot())
    assert len(renderer.errors) >= 1
    assert _text(output).count("display error: ValueError: bad draw") == 1


def test_pause_and_resume_stop_and_restart_the_view():
    renderer, output, keys = _collapsed()
    renderer.feed(_upd("model", [_call("ls", {"path": "/"}, "c1")]))
    with renderer.paused():
        assert keys.events == ["start", "stop"]
        assert renderer._live_region is None
        with renderer.paused():  # nested: still down
            pass
        assert keys.events == ["start", "stop"]
    assert keys.events == ["start", "stop", "start"]
    assert renderer.collapsed is True
    with pytest.raises(KeyboardInterrupt):
        with renderer.paused():
            raise KeyboardInterrupt
    assert keys.events == ["start", "stop", "start", "stop"]  # Ctrl+C at a prompt: stays down
    renderer.interrupted()
    lines = _printed(output)
    assert lines[-2:] == ["⏺ Cooked for 0s · 1 call · /calls to see them", "interrupted"]


def test_format_elapsed():
    assert format_elapsed(0) == "0s"
    assert format_elapsed(12.7) == "12s"
    assert format_elapsed(64) == "1m 04s"
    assert format_elapsed(3723) == "1h 02m 03s"


# --- key decoding -----------------------------------------------------------------


def test_windows_arrow_sequences_decode():
    assert decode_windows("\xe0H\xe0P\xe0K\xe0M") == ["up", "down", "left", "right"]
    assert decode_windows("\x00H\x00P") == ["up", "down"]
    assert decode_windows("abc\r\x1b") == []  # other keys are dropped
    assert decode_windows("\xe0G\x00;") == []  # Home, F1: not arrows
    assert decode_windows("\xe0") == []


def test_posix_arrow_sequences_decode():
    assert decode_posix(b"\x1b[A\x1b[B\x1b[C\x1b[D") == ["up", "down", "right", "left"]
    assert decode_posix(b"\x1bOA\x1bOB") == ["up", "down"]
    assert decode_posix(b"jk\n q") == []
    assert decode_posix(b"\x1b[1;5A\x1b[H") == []  # Ctrl+arrow and Home are not plain arrows


def test_posix_sequence_split_across_reads():
    decoder = PosixDecoder()
    assert decoder.feed(b"\x1b") == []
    assert decoder.feed(b"[") == []
    assert decoder.feed(b"B") == ["down"]
    assert decoder.feed(b"x\x1b[") == []
    assert decoder.feed(b"A") == ["up"]


def test_no_keyboard_means_a_no_op_listener(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    listener = default_key_listener()
    assert isinstance(listener, NoKeys)
    listener.start(lambda _key: None)
    listener.stop()


# --- the loop: approvals and /calls ------------------------------------------------


def _worker():
    worker = scripted_model(lambda _m: AIMessage(content="worker"))
    return role_models({"general-purpose": worker}, default=worker)


def _terminal_repl(tmp_path, responder, *, input_fn=None, **config_fields):
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    console, output = _terminal(width=120)
    from absolutcoder.repl import Repl

    repl = Repl(
        repo,
        Config(main_model="openai:main-test", subagent_model="openai:worker-test", **config_fields),
        chat_model=scripted_model(responder),
        model_factory=_worker(),
        input_fn=input_fn,
        console=console,
    )
    return repl, output


def _git_repo(path):
    import subprocess

    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    return path


def _no_web(monkeypatch):
    from langchain_core.tools import tool

    @tool
    def web_fetch(url: str) -> str:
        """Fetch a page (test double)."""
        return "page"

    @tool
    def web_search(query: str) -> str:
        """Search (test double)."""
        return "results"

    monkeypatch.setattr("absolutcoder.repl.make_web_tools", lambda **_kw: [web_fetch, web_search])


def test_pause_and_resume_around_approval(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    _git_repo(tmp_path / "repo")
    keys = FakeKeys()
    seen_at_prompt: list[list[str]] = []

    def ask(prompt=""):
        seen_at_prompt.append(list(keys.events))
        return "y"

    def responder(messages):
        if "execute" not in tool_names_since_last_human(messages):
            return tool_call("execute", {"command": "git status"}, "call-status")
        return AIMessage(content="status done")

    repl, output = _terminal_repl(tmp_path, responder, input_fn=ask, auto_run=False)
    repl.key_listener = keys
    try:
        assert repl.run_turn("show git status") is not None
    finally:
        repl.close()
    assert seen_at_prompt == [["start", "stop"]]  # stopped before the prompt
    assert keys.events == ["start", "stop", "start", "stop"]  # started again, stopped at the end
    printed = _printed(output)
    assert "Run command? git status" in printed
    assert not any("tool: execute(" in line for line in printed)
    assert printed.count("status done") == 1
    assert printed[-1].startswith("⏺ Cooked for") and printed[-1].endswith("1 call · /calls to see them")


def test_calls_command_prints_last_turn_log(tmp_path, monkeypatch):
    _no_web(monkeypatch)

    def responder(messages):
        if "ls" not in tool_names_since_last_human(messages):
            return tool_call("ls", {"path": "/"}, "call-ls")
        return AIMessage(content="listed")

    repl, output = _terminal_repl(tmp_path, responder, auto_run=True)
    repl.key_listener = FakeKeys()
    try:
        repl.run_turn("list files")
        assert not any("tool: ls" in line for line in _printed(output))
        start = len(output.getvalue())
        repl.handle_line("/calls")
        shown = _ANSI.sub("", output.getvalue()[start:]).splitlines()
        assert shown == ["  tool: ls(/)", "  done: ls"]
    finally:
        repl.close()


def test_calls_command_without_calls(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    repl, output = _terminal_repl(tmp_path, lambda _m: AIMessage(content="hi"), auto_run=True)
    repl.key_listener = FakeKeys()
    try:
        repl.handle_line("/calls")  # before any turn
        repl.run_turn("say hi")
        repl.handle_line("/calls")  # a turn with no calls
    finally:
        repl.close()
    assert _text(output).splitlines().count("No calls in the last turn.") == 2


def test_calls_is_a_builtin_that_no_skill_shadows(tmp_path, monkeypatch):
    from absolutcoder import repl as repl_module

    assert "/calls" in repl_module._BUILTIN_COMMANDS
    assert "/calls" in repl_module._COMMANDS
    assert any(name == "/calls" for name, _ in repl_module._HELP)
    repl, _output = _terminal_repl(tmp_path, lambda _m: AIMessage(content="hi"), auto_run=True)
    try:
        assert repl.skill_for_command("/calls") is None
    finally:
        repl.close()


def test_full_setting_on_a_terminal_prints_every_line(tmp_path, monkeypatch):
    _no_web(monkeypatch)

    def responder(messages):
        if "ls" not in tool_names_since_last_human(messages):
            return tool_call("ls", {"path": "/"}, "call-ls")
        return AIMessage(content="listed")

    repl, output = _terminal_repl(tmp_path, responder, auto_run=True, activity="full")
    keys = FakeKeys()
    repl.key_listener = keys
    try:
        repl.run_turn("list files")
    finally:
        repl.close()
    assert "  tool: ls(/)" in _text(output).splitlines()
    assert keys.events == []
    assert "Cooked for" not in _text(output)


def test_stream_error_in_collapsed_mode_stops_the_view_and_summarises(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    repl, output = _terminal_repl(tmp_path, lambda _m: AIMessage(content="hi"), auto_run=True)
    keys = FakeKeys()
    repl.key_listener = keys

    def stream(*_args, **_kwargs):
        yield _upd("model", [_call("ls", {"path": "/"}, "c1")])
        raise RuntimeError("stream broke")

    repl.agent.stream = stream
    try:
        assert repl.run_turn("go") is None
    finally:
        repl.close()
    lines = _printed(output)
    assert keys.events == ["start", "stop"]
    assert lines[-1] == "Error: stream broke"
    assert lines[-2].startswith("⏺ Cooked for ") and lines[-2].endswith("1 call · /calls to see them")


def test_ctrl_c_in_collapsed_mode_prints_summary_then_interrupted(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    repl, output = _terminal_repl(tmp_path, lambda _m: AIMessage(content="hi"), auto_run=True)
    keys = FakeKeys()
    repl.key_listener = keys

    def stream(*_args, **_kwargs):
        yield _upd("model", [_call("ls", {"path": "/"}, "c1")])
        raise KeyboardInterrupt

    repl.agent.stream = stream
    try:
        assert repl.run_turn("go") is None
    finally:
        repl.close()
    lines = _printed(output)
    assert keys.events == ["start", "stop"]
    assert lines[-1] == "interrupted"
    assert lines[-2].startswith("⏺ Cooked for ")


# --- fix round 1 -------------------------------------------------------------------


def _chunk(text, message_id="m1"):
    from langchain_core.messages import AIMessageChunk

    message = AIMessageChunk(content=text, id=message_id)
    return {"type": "messages", "ns": (), "data": (message, {"langgraph_node": "model"})}


class StopInterrupted(FakeKeys):
    """A listener whose first stop() is cut short by Ctrl+C (inside its join)."""

    def __init__(self):
        super().__init__()
        self.raised = False

    def stop(self):
        super().stop()
        if not self.raised:
            self.raised = True
            raise KeyboardInterrupt


def test_ctrl_c_while_the_listener_stops_still_stops_the_live_region():
    console, output = _terminal()
    keys = StopInterrupted()
    renderer = StreamRenderer(console, clock=FakeClock())
    assert renderer.start(activity="collapsed", key_listener=keys)
    region = renderer._live_region
    assert region.is_started
    with pytest.raises(KeyboardInterrupt):
        renderer.finish(_snapshot())
    assert not region.is_started  # stopped in spite of the interrupt
    assert region._refresh_thread is None
    renderer.interrupted()  # today's interrupt path afterwards: summary, then "interrupted"
    assert _printed(output)[-2:] == ["⏺ Cooked for 0s", "interrupted"]


def test_live_region_is_kept_for_a_retry_when_its_own_stop_is_interrupted():
    renderer, _output, _keys = _collapsed()
    region = renderer._live_region
    real_stop = type(region).stop
    calls = {"n": 0}

    def flaky_stop():
        calls["n"] += 1
        if calls["n"] == 1:
            raise KeyboardInterrupt
        real_stop(region)

    region.stop = flaky_stop
    with pytest.raises(KeyboardInterrupt):
        renderer.stop_view()
    assert renderer._live_region is region  # not forgotten: the next stop finds it
    renderer.stop_view()
    assert renderer._live_region is None and not region.is_started


def test_collapsed_partial_text_is_printed_once_on_interrupt():
    renderer, output, _keys = _collapsed()
    for piece in ("Half an ", "answer"):
        renderer.feed(_chunk(piece))
    assert "Half an answer" not in _printed(output)  # nothing streamed while collapsed
    renderer.interrupted()
    assert _printed(output) == ["⏺ Cooked for 0s", "Half an answer", "interrupted"]


def test_collapsed_partial_text_is_printed_on_a_stream_error():
    renderer, output, _keys = _collapsed()
    renderer.feed(_chunk("cut off"))
    renderer.abort()
    assert _printed(output) == ["⏺ Cooked for 0s", "cut off"]


def test_collapsed_completed_message_prints_its_text_once():
    renderer, output, _keys = _collapsed()
    for piece in ("Whole ", "answer"):
        renderer.feed(_chunk(piece, "m1"))
    renderer.feed(_upd("model", [AIMessage(content="Whole answer", id="final-id")]))
    renderer.interrupted()
    assert _printed(output).count("Whole answer") == 1


def test_a_repeated_drawing_error_is_recorded_once():
    renderer, _output, _keys = _collapsed()

    def broken(width):
        raise ValueError("bad draw")

    renderer._status_lines = broken
    for _ in range(50):
        renderer.status_lines(80)
    renderer.stop_view()
    assert len(renderer.errors) == 1


def test_a_start_time_of_zero_counts():
    clock = FakeClock(0.0)
    renderer, output, _keys = _collapsed(clock=clock)
    clock.now = 75.0
    renderer.finish(_snapshot())
    assert "⏺ Cooked for 1m 15s" in _printed(output)


class FakeMsvcrt:
    """Characters waiting in the console; getwch on an empty queue would block for ever."""

    def __init__(self, chars):
        self.chars = list(chars)

    def kbhit(self):
        return bool(self.chars)

    def getwch(self):
        if not self.chars:
            raise AssertionError("getwch would block: no key is waiting")
        return self.chars.pop(0)


def test_windows_reader_handles_a_lone_prefix_character():
    from absolutcoder.keys import read_windows_keys

    assert read_windows_keys(FakeMsvcrt("à")) == []  # a typed à: no wait for a second character
    assert read_windows_keys(FakeMsvcrt("\xe0P")) == ["down"]
    assert read_windows_keys(FakeMsvcrt("\x00H")) == ["up"]
    assert read_windows_keys(FakeMsvcrt("q")) == []
    queue = FakeMsvcrt("\xe0Ka")
    assert read_windows_keys(queue) == ["left"]
    assert queue.chars == ["a"]


def test_windows_listener_thread_reads_through_an_injected_console():
    import threading

    from absolutcoder.keys import WindowsKeys

    class OneByOne(FakeMsvcrt):
        """à arrives alone; the arrow's two halves arrive together a moment later."""

        def __init__(self):
            super().__init__("")
            self.batches = [["à"], ["\xe0", "M"]]
            self.gap = False  # one empty poll between batches: the next key comes later

        def kbhit(self):
            if not self.chars and self.batches:
                if self.gap:
                    self.gap = False
                    return False
                self.chars = self.batches.pop(0)
                self.gap = True
            return bool(self.chars)

    got: list[str] = []
    seen = threading.Event()
    listener = WindowsKeys(msvcrt=OneByOne())

    def on_key(key):
        got.append(key)
        seen.set()

    listener.start(on_key)
    assert seen.wait(2)
    listener.stop()
    assert got == ["right"]
    assert listener._thread is None


def test_posix_listener_only_in_the_foreground(monkeypatch):
    import absolutcoder.keys as keys_module

    class TTY:
        def isatty(self):
            return True

        def fileno(self):
            return 0

    made = []

    class StubPosix:
        def __init__(self):
            made.append(self)

    monkeypatch.setattr(keys_module.sys, "stdin", TTY())
    monkeypatch.setattr(keys_module.sys, "platform", "linux")
    monkeypatch.setattr(keys_module, "PosixKeys", StubPosix)
    monkeypatch.setattr(keys_module.os, "getpgrp", lambda: 42, raising=False)
    monkeypatch.setattr(keys_module.os, "tcgetpgrp", lambda fd: 42, raising=False)
    assert isinstance(keys_module.default_key_listener(), StubPosix)
    monkeypatch.setattr(keys_module.os, "tcgetpgrp", lambda fd: 7, raising=False)  # `absolut ... &`
    assert isinstance(keys_module.default_key_listener(), NoKeys)

    def no_terminal(fd):
        raise OSError("not a terminal")

    monkeypatch.setattr(keys_module.os, "tcgetpgrp", no_terminal, raising=False)
    assert isinstance(keys_module.default_key_listener(), NoKeys)
    assert len(made) == 1


def test_hitl_rejection_from_after_model_still_prints():
    console, output = _plain_console()
    renderer = StreamRenderer(console, live=False)
    renderer.feed(_upd("model", [_call("execute", {"command": "rm -rf x"}, "c1")]))
    rejected = ToolMessage(content="User rejected the command.", tool_call_id="c1", name="execute", id="r1", status="error")
    renderer.feed(_upd("HumanInTheLoopMiddleware.after_model", [rejected]))
    assert "  error: execute: User rejected the command." in _text(output).splitlines()


def test_a_trimmed_result_is_never_a_result_line_whatever_node_sends_it():
    from absolutcoder.history import IMAGES_TRIMMED_KEY

    console, output = _plain_console()
    renderer = StreamRenderer(console, live=False)
    trimmed = ToolMessage(
        content=[{"type": "text", "text": "[image ...]"}], tool_call_id="old", name="read_file", id="t-old",
        additional_kwargs={IMAGES_TRIMMED_KEY: True},
    )
    renderer.feed(_upd("SeenImagesMiddleware.before_model", [trimmed]))
    renderer.feed(_upd("some.other_node", [trimmed]))
    assert output.getvalue() == "" and renderer.call_log == []


def test_trimmed_image_is_not_new_activity_in_the_next_turn(tmp_path):
    import base64

    from langgraph.checkpoint.memory import InMemorySaver

    from absolutcoder.agent import build_agent

    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "shot.png").write_bytes(png)

    def respond(messages):
        humans = [m for m in messages if getattr(m, "type", None) == "human"]
        if len(humans) == 1 and "read_file" not in tool_names_since_last_human(messages):
            return tool_call("read_file", {"file_path": "/shot.png"}, "read-1")
        return AIMessage(content=f"answer {len(humans)}")

    agent = build_agent(
        repo,
        Config(main_model="openai:main-test", subagent_model="openai:worker-test", auto_run=True),
        InMemorySaver(),
        chat_model=scripted_model(respond),
        model_factory=role_models({}, default=scripted_model(lambda _m: AIMessage(content="ok"))),
    )
    config = {"configurable": {"thread_id": "two-turns"}}
    renderers = []
    for text in ("look at the screenshot", "and now?"):
        console, output = _plain_console()
        renderer = StreamRenderer(console, live=False)
        for part in agent.stream(
            {"messages": [{"role": "user", "content": text}]},
            config,
            stream_mode=["messages", "updates"],
            subgraphs=True,
            version="v2",
        ):
            renderer.feed(part)
        renderer.finish(agent.get_state(config))
        renderers.append((renderer, output))
    first, second = renderers
    assert first[0].call_log == ["  tool: read_file(/shot.png)", "  done: read_file"]
    assert second[0].call_log == []
    assert _text(second[1]).splitlines() == ["answer 2"]
    assert second[0].errors == []
