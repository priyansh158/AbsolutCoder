"""The streaming renderer: main text, tool lines and labelled worker activity."""

from io import StringIO
from types import SimpleNamespace

from langchain_core.messages import AIMessage, AIMessageChunk, RemoveMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from rich.console import Console

from absolutcoder.render import StreamRenderer, summarize_call

WORKER_NS = ("tools:abc",)


def _renderer(live):
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True, width=120)
    return StreamRenderer(console, live=live), output


def _clean(renderer):
    assert renderer.errors == []


def _msg(message, ns=(), node="model", agent=None):
    metadata = {"langgraph_node": node}
    if agent is not None:
        metadata["lc_agent_name"] = agent
    return {"type": "messages", "ns": ns, "data": (message, metadata)}


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


def test_tokens_are_concatenated_per_agent():
    renderer, output = _renderer(True)
    for text in ("Hel", "lo", " there"):
        renderer.feed(_msg(AIMessageChunk(content=text, id="m1")))
    renderer.finish(_snapshot(AIMessage(content="Hello there", id="m1")))
    lines = output.getvalue().splitlines()
    assert "Hello there" in lines
    assert output.getvalue().count("Hello there") == 1
    _clean(renderer)


def test_whole_message_is_printed_once():
    for live in (True, False):
        renderer, output = _renderer(live)
        final = AIMessage(content="the answer", id="final")
        renderer.feed(_msg(final))
        renderer.feed(_upd("model", [final]))
        renderer.finish(_snapshot(final))
        assert output.getvalue().count("the answer") == 1, live
        _clean(renderer)


def test_message_without_id_is_compared_by_text():
    renderer, output = _renderer(False)
    renderer.feed(_upd("model", [AIMessage(content="plain")]))
    renderer.finish(_snapshot(AIMessage(content="plain")))
    assert output.getvalue().count("plain") == 1
    _clean(renderer)


def test_chunks_with_changing_ids_are_matched_by_text():
    renderer, output = _renderer(True)
    for text, chunk_id in (("Hel", "a"), ("lo", "b"), (" there", "c")):
        renderer.feed(_msg(AIMessageChunk(content=text, id=chunk_id)))
    final = AIMessage(content="Hello there")
    final.id = None
    renderer.feed(_upd("model", [final]))
    renderer.finish(_snapshot(AIMessage(content="Hello there", id="assigned-later")))
    assert output.getvalue().count("Hello there") == 1
    _clean(renderer)


def test_narration_before_a_tool_call_is_shown_once():
    call = _call("execute", {"command": "ls"}, "e1", message_id="n1", content="Let me check the config.")
    renderer, output = _renderer(False)
    renderer.feed(_msg(call))
    renderer.feed(_upd("model", [call]))
    renderer.finish(_snapshot(call, AIMessage(content="Let me check the config.", id="n1")))
    text = output.getvalue()
    assert text.count("Let me check the config.") == 1
    assert text.index("Let me check the config.") < text.index("  tool: execute(ls)")
    _clean(renderer)

    renderer, output = _renderer(True)
    for piece in ("Let me check", " the config."):
        renderer.feed(_msg(AIMessageChunk(content=piece, id="n1")))
    renderer.feed(_upd("model", [call]))
    renderer.finish(_snapshot(call))
    text = output.getvalue()
    assert text.count("Let me check the config.") == 1
    assert text.index("Let me check the config.") < text.index("  tool: execute(ls)")
    _clean(renderer)


def test_reused_call_ids_still_render_each_round():
    renderer, output = _renderer(False)
    for round_no in (1, 2):
        call = _call("execute", {"command": f"echo {round_no}"}, "call_0", message_id=f"ai-{round_no}")
        renderer.feed(_upd("model", [call]))
        result = ToolMessage(content="ok", tool_call_id="call_0", name="execute", id=f"tm-{round_no}")
        renderer.feed(_upd("tools", [result]))
    lines = output.getvalue().splitlines()
    assert lines.count("  tool: execute(echo 1)") == 1
    assert lines.count("  tool: execute(echo 2)") == 1
    assert lines.count("  done: execute") == 2
    _clean(renderer)


def test_same_ai_message_twice_prints_one_call_line():
    renderer, output = _renderer(False)
    call = _call("execute", {"command": "ls"}, "call_0", message_id="ai-1")
    renderer.feed(_upd("model", [call]))
    renderer.feed(_upd("model", [call]))
    result = ToolMessage(content="ok", tool_call_id="call_0", name="execute", id="tm-1")
    renderer.feed(_upd("tools", [result]))
    renderer.feed(_upd("tools", [result]))
    lines = output.getvalue().splitlines()
    assert lines.count("  tool: execute(ls)") == 1
    assert lines.count("  done: execute") == 1
    _clean(renderer)


def test_task_role_comes_from_the_latest_call_with_that_id():
    renderer, output = _renderer(False)
    first = _call("task", {"description": "a", "subagent_type": "explore"}, "call_0", message_id="ai-1")
    second = _call("task", {"description": "b", "subagent_type": "plan"}, "call_0", message_id="ai-2")
    renderer.feed(_upd("model", [first]))
    renderer.feed(_upd("tools", [ToolMessage(content="r", tool_call_id="call_0", name="task", id="t1")]))
    renderer.feed(_upd("model", [second]))
    renderer.feed(_upd("tools", [ToolMessage(content="r", tool_call_id="call_0", name="task", id="t2")]))
    lines = output.getvalue().splitlines()
    assert "  task [explore] finished" in lines
    assert "  task [plan] finished" in lines
    _clean(renderer)


def test_subagent_lines_are_labelled():
    renderer, output = _renderer(True)
    renderer.feed(_msg(AIMessage(content="worker thinking", id="w1"), ns=WORKER_NS, agent="explore"))
    renderer.feed(_upd("model", [_call("grep", {"pattern": "TODO"}, "g1")], ns=WORKER_NS))
    renderer.feed(_upd("tools", [ToolMessage(content="x", tool_call_id="g1", name="grep")], ns=WORKER_NS))
    text = output.getvalue()
    assert "  [explore] tool: grep(TODO)" in text.splitlines()
    assert "  [explore] done: grep" in text.splitlines()
    assert "worker thinking" not in text
    _clean(renderer)


def test_unknown_worker_namespace_uses_generic_label():
    renderer, output = _renderer(False)
    renderer.feed(_upd("model", [_call("ls", {"path": "/src"}, "l1")], ns=("tools:zzz",)))
    assert "  [worker] tool: ls(/src)" in output.getvalue().splitlines()
    _clean(renderer)


def test_tool_call_and_result_lines():
    renderer, output = _renderer(False)
    renderer.feed(_upd("model", [_call("execute", {"command": "git status"}, "e1")]))
    renderer.feed(_upd("tools", [ToolMessage(content="clean", tool_call_id="e1", name="execute")]))
    renderer.feed(_upd("model", [_call("execute", {"command": "false"}, "e2")]))
    renderer.feed(
        _upd(
            "tools",
            [ToolMessage(content="boom\nsecond line", tool_call_id="e2", name="execute", status="error")],
        )
    )
    lines = output.getvalue().splitlines()
    assert "  tool: execute(git status)" in lines
    assert "  done: execute" in lines
    assert "  error: execute: boom" in lines
    assert "second line" not in output.getvalue()
    _clean(renderer)


def test_task_lines_name_the_role():
    renderer, output = _renderer(False)
    call = _call("task", {"description": "find\n the  thing", "subagent_type": "explore"}, "t1")
    renderer.feed(_upd("model", [call]))
    renderer.feed(_upd("tools", [ToolMessage(content="report", tool_call_id="t1", name="task")]))
    lines = output.getvalue().splitlines()
    assert "  task [explore]: find the thing" in lines
    assert "  task [explore] finished" in lines
    _clean(renderer)


def test_history_rewrite_outside_tools_node_is_skipped():
    renderer, output = _renderer(False)
    history = [
        RemoveMessage(id=REMOVE_ALL_MESSAGES),
        AIMessage(content="old answer", id="old"),
        ToolMessage(content="x", tool_call_id="old-call", name="execute", id="old-tm"),
    ]
    renderer.feed(_upd("PatchToolCallsMiddleware.before_agent", history))
    assert output.getvalue() == ""
    _clean(renderer)


def test_tools_node_update_with_remove_message_still_shows_results():
    renderer, output = _renderer(False)
    renderer.feed(_upd("model", [_call("execute", {"command": "ls"}, "e1")]))
    update = [
        RemoveMessage(id=REMOVE_ALL_MESSAGES),
        AIMessage(content="replayed", id="r1"),
        ToolMessage(content="ok", tool_call_id="e1", name="execute", id="tm-1"),
    ]
    renderer.feed(_upd("tools", update))
    text = output.getvalue()
    assert "  done: execute" in text.splitlines()
    assert "replayed" not in text
    _clean(renderer)


def test_plain_console_emits_no_control_codes():
    renderer, output = _renderer(False)
    renderer.feed(_msg(AIMessageChunk(content="tok", id="m1")))
    renderer.feed(_upd("model", [_call("execute", {"command": "ls"}, "e1")]))
    renderer.feed(_upd("model", [AIMessage(content="done", id="m2")]))
    renderer.finish(_snapshot(AIMessage(content="done", id="m2")))
    text = output.getvalue()
    assert "\x1b" not in text
    assert "\r" not in text
    _clean(renderer)


def test_render_handles_block_content():
    content = [
        {"type": "reasoning", "text": "secret"},
        {"type": "text", "text": "visible"},
        {"type": "tool_use", "id": "x"},
    ]
    for live in (True, False):
        renderer, output = _renderer(live)
        message = AIMessage(content=content, id="b1")
        renderer.feed(_msg(message))
        renderer.feed(_upd("model", [message]))
        renderer.finish(_snapshot(message))
        text = output.getvalue()
        assert "visible" in text
        assert "secret" not in text
        assert "tool_use" not in text
        assert "{" not in text and "[" not in text
        _clean(renderer)


def test_only_model_node_tokens_render():
    renderer, output = _renderer(True)
    renderer.feed(_msg(AIMessageChunk(content="summary text", id="s1"), node="SummarizationMiddleware.before_model"))
    assert "summary text" not in output.getvalue()
    _clean(renderer)


def test_summarize_call_is_one_short_line():
    command = "echo start\n" + "x" * 500
    summary = summarize_call("execute", {"command": command})
    assert "\n" not in summary
    assert len(summary) <= 80
    assert summary.endswith("…")


def test_feed_ignores_unexpected_shapes():
    renderer, _output = _renderer(True)
    renderer.feed({})
    renderer.feed({"type": "custom", "ns": (), "data": 1})
    renderer.feed({"type": "updates", "ns": (), "data": None})
    renderer.feed({"type": "messages", "ns": (), "data": "x"})
    renderer.feed({"type": "updates", "ns": (), "data": {"model": {"messages": [object()]}}})
    renderer.feed({"type": "updates", "ns": (), "data": {"__interrupt__": (object(),)}})
    _clean(renderer)


def test_renderer_bug_is_recorded_and_reported_once(monkeypatch):
    renderer, output = _renderer(True)
    renderer.feed(_msg(AIMessageChunk(content="half", id="m1")))

    def broken(*_args, **_kwargs):
        raise ValueError("bad call")

    monkeypatch.setattr(renderer, "_on_call", broken)
    renderer.feed(_upd("model", [_call("execute", {"command": "ls"}, "e1")]))
    renderer.feed(_upd("model", [_call("execute", {"command": "pwd"}, "e2")]))
    monkeypatch.undo()
    renderer.feed(_upd("tools", [ToolMessage(content="ok", tool_call_id="e1", name="execute")]))
    lines = output.getvalue().splitlines()
    assert len(renderer.errors) == 2
    assert isinstance(renderer.errors[0], ValueError)
    assert [line for line in lines if line.startswith("display error:")] == [
        "display error: ValueError: bad call"
    ]
    assert lines[0] == "half"
    assert "  done: execute" in lines


def test_interrupted_ends_line_and_says_so():
    renderer, output = _renderer(True)
    renderer.feed(_msg(AIMessageChunk(content="half a", id="m1")))
    renderer.interrupted()
    assert output.getvalue().splitlines() == ["half a", "interrupted"]
    _clean(renderer)
