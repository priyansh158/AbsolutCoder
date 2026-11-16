"""The streaming renderer: main text, tool lines and labelled worker activity."""

from io import StringIO
from types import SimpleNamespace

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from rich.console import Console

from absolutcoder.render import StreamRenderer, summarize_call

WORKER_NS = ("tools:abc",)


def _console():
    output = StringIO()
    return Console(file=output, force_terminal=False, no_color=True, width=120), output


def _msg(message, ns=(), node="model", agent=None):
    metadata = {"langgraph_node": node}
    if agent is not None:
        metadata["lc_agent_name"] = agent
    return {"type": "messages", "ns": ns, "data": (message, metadata)}


def _upd(node, messages, ns=()):
    return {"type": "updates", "ns": ns, "data": {node: {"messages": messages}}}


def _call(name, args, call_id):
    return AIMessage(
        content="",
        id="ai-" + call_id,
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def _snapshot(*messages):
    return SimpleNamespace(values={"messages": list(messages)})


def test_tokens_are_concatenated_per_agent():
    console, output = _console()
    renderer = StreamRenderer(console, live=True)
    for text in ("Hel", "lo", " there"):
        renderer.feed(_msg(AIMessageChunk(content=text, id="m1")))
    renderer.finish(_snapshot(AIMessage(content="Hello there", id="m1")))
    lines = output.getvalue().splitlines()
    assert "Hello there" in lines
    assert output.getvalue().count("Hello there") == 1


def test_whole_message_is_printed_once():
    for live in (True, False):
        console, output = _console()
        renderer = StreamRenderer(console, live=live)
        final = AIMessage(content="the answer", id="final")
        renderer.feed(_msg(final))
        renderer.feed(_upd("model", [final]))
        renderer.finish(_snapshot(final))
        assert output.getvalue().count("the answer") == 1, live


def test_message_without_id_is_compared_by_text():
    console, output = _console()
    renderer = StreamRenderer(console, live=False)
    renderer.feed(_upd("model", [AIMessage(content="plain")]))
    renderer.finish(_snapshot(AIMessage(content="plain")))
    assert output.getvalue().count("plain") == 1


def test_subagent_lines_are_labelled():
    console, output = _console()
    renderer = StreamRenderer(console, live=True)
    renderer.feed(_msg(AIMessage(content="worker thinking", id="w1"), ns=WORKER_NS, agent="explore"))
    renderer.feed(_upd("model", [_call("grep", {"pattern": "TODO"}, "g1")], ns=WORKER_NS))
    renderer.feed(_upd("tools", [ToolMessage(content="x", tool_call_id="g1", name="grep")], ns=WORKER_NS))
    text = output.getvalue()
    assert "  [explore] tool: grep(TODO)" in text.splitlines()
    assert "  [explore] done: grep" in text.splitlines()
    assert "worker thinking" not in text


def test_unknown_worker_namespace_uses_generic_label():
    console, output = _console()
    renderer = StreamRenderer(console, live=False)
    renderer.feed(_upd("model", [_call("ls", {"path": "/src"}, "l1")], ns=("tools:zzz",)))
    assert "  [worker] tool: ls(/src)" in output.getvalue().splitlines()


def test_tool_call_and_result_lines():
    console, output = _console()
    renderer = StreamRenderer(console, live=False)
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


def test_task_lines_name_the_role():
    console, output = _console()
    renderer = StreamRenderer(console, live=False)
    call = _call("task", {"description": "find\n the  thing", "subagent_type": "explore"}, "t1")
    renderer.feed(_upd("model", [call]))
    renderer.feed(_upd("tools", [ToolMessage(content="report", tool_call_id="t1", name="task")]))
    lines = output.getvalue().splitlines()
    assert "  task [explore]: find the thing" in lines
    assert "  task [explore] finished" in lines


def test_plain_console_emits_no_control_codes():
    console, output = _console()
    renderer = StreamRenderer(console, live=False)
    renderer.feed(_msg(AIMessageChunk(content="tok", id="m1")))
    renderer.feed(_upd("model", [_call("execute", {"command": "ls"}, "e1")]))
    renderer.feed(_upd("model", [AIMessage(content="done", id="m2")]))
    renderer.finish(_snapshot(AIMessage(content="done", id="m2")))
    text = output.getvalue()
    assert "\x1b" not in text
    assert "\r" not in text


def test_render_handles_block_content():
    content = [
        {"type": "reasoning", "text": "secret"},
        {"type": "text", "text": "visible"},
        {"type": "tool_use", "id": "x"},
    ]
    for live in (True, False):
        console, output = _console()
        renderer = StreamRenderer(console, live=live)
        message = AIMessage(content=content, id="b1")
        renderer.feed(_msg(message))
        renderer.feed(_upd("model", [message]))
        renderer.finish(_snapshot(message))
        text = output.getvalue()
        assert "visible" in text
        assert "secret" not in text
        assert "tool_use" not in text
        assert "{" not in text and "[" not in text


def test_only_model_node_tokens_render():
    console, output = _console()
    renderer = StreamRenderer(console, live=True)
    renderer.feed(_msg(AIMessageChunk(content="summary text", id="s1"), node="SummarizationMiddleware.before_model"))
    assert "summary text" not in output.getvalue()


def test_summarize_call_is_one_short_line():
    command = "echo start\n" + "x" * 500
    summary = summarize_call("execute", {"command": command})
    assert "\n" not in summary
    assert len(summary) <= 80
    assert summary.endswith("…")


def test_feed_ignores_unexpected_shapes():
    console, _output = _console()
    renderer = StreamRenderer(console, live=True)
    renderer.feed({})
    renderer.feed({"type": "custom", "ns": (), "data": 1})
    renderer.feed({"type": "updates", "ns": (), "data": None})
    renderer.feed({"type": "messages", "ns": (), "data": "x"})
    renderer.feed({"type": "updates", "ns": (), "data": {"model": {"messages": [object()]}}})
    renderer.feed({"type": "updates", "ns": (), "data": {"__interrupt__": (object(),)}})


def test_interrupted_ends_line_and_says_so():
    console, output = _console()
    renderer = StreamRenderer(console, live=True)
    renderer.feed(_msg(AIMessageChunk(content="half a", id="m1")))
    renderer.interrupted()
    assert output.getvalue().splitlines() == ["half a", "interrupted"]
