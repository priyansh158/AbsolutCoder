"""Images the model has already seen are replaced by a note in the stored history."""

import base64

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from absolutcoder.agent import build_agent, build_role_graph
from absolutcoder.config import Config
from absolutcoder.history import (
    PLACEHOLDER,
    SeenImagesMiddleware,
    is_image_block,
    seen_image_update,
    without_images,
)
from absolutcoder.subagents import builtin_agents
from tests.support import role_models, scripted_model, task_call, tool_call, tool_calls

_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
_PNG = base64.b64decode(_PNG_B64)
_CONFIG = Config(main_model="openai:main-test", subagent_model="openai:worker-test", auto_run=True)


def _ok():
    return scripted_model(lambda _m: AIMessage(content="ok"))


def _tool_messages(messages):
    return [m for m in messages if isinstance(m, ToolMessage)]


def _has_image(message) -> bool:
    return isinstance(message.content, list) and any(is_image_block(b) for b in message.content)


def _note(source):
    return PLACEHOLDER.format(source=source)


def _repo_with_shot(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "shot.png").write_bytes(_PNG)
    return repo


def _image_then_more(seen):
    """Read the screenshot, then list the folder, then answer; records what each call saw."""

    def respond(messages):
        seen.append(list(messages))
        done = [m for m in messages if isinstance(m, ToolMessage)]
        if not done:
            return tool_call("read_file", {"file_path": "/shot.png"}, "read-1")
        if len(done) == 1:
            return tool_call("ls", {"path": "/"}, "ls-1")
        return AIMessage(content="looked twice")

    return respond


def test_image_blocks_are_recognised_in_every_shape():
    assert is_image_block({"type": "image", "base64": _PNG_B64, "mime_type": "image/png"})
    assert is_image_block({"type": "image_url", "image_url": {"url": "data:image/png;base64,xx"}})
    assert is_image_block({"type": "file", "base64": _PNG_B64, "mime_type": "image/png"})
    assert is_image_block({"type": "document", "source": {"type": "base64", "media_type": "image/jpeg", "data": "x"}})
    assert is_image_block({"type": "other", "url": "data:image/gif;base64,R0lG"})
    assert not is_image_block({"type": "text", "text": "data:image/png;base64, quoted in text"})
    assert not is_image_block({"type": "file", "base64": "JVBER", "mime_type": "application/pdf"})
    assert not is_image_block("plain string")


def test_text_and_other_blocks_are_kept_and_metadata_survives():
    message = ToolMessage(
        content=[
            {"type": "text", "text": "before"},
            {"type": "image", "base64": _PNG_B64, "mime_type": "image/png"},
            {"type": "image", "base64": _PNG_B64, "mime_type": "image/png"},
            {"type": "text", "text": "after"},
        ],
        tool_call_id="call-1",
        name="screenshot",
        id="tool-msg-1",
        status="success",
        additional_kwargs={"read_file_path": "/shots/a.png"},
    )
    trimmed = without_images(message)
    assert trimmed.content == [
        {"type": "text", "text": "before"},
        {"type": "text", "text": _note("/shots/a.png")},
        {"type": "text", "text": "after"},
    ]
    assert (trimmed.id, trimmed.tool_call_id, trimmed.name, trimmed.status) == (
        "tool-msg-1", "call-1", "screenshot", "success"
    )
    unnamed = ToolMessage(content=[{"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}}], tool_call_id="c", name="snap", id="t")
    assert without_images(unnamed).content == [{"type": "text", "text": _note("snap")}]
    assert without_images(ToolMessage(content="text only", tool_call_id="c", id="t")) is None


def test_only_results_before_the_latest_ai_message_are_trimmed():
    image = ToolMessage(content=[{"type": "image", "base64": _PNG_B64, "mime_type": "image/png"}], tool_call_id="c1", name="read_file", id="t1")
    ask = HumanMessage(content="look", id="h1")
    call = AIMessage(content="", id="a1", tool_calls=[{"name": "read_file", "args": {}, "id": "c1", "type": "tool_call"}])
    assert seen_image_update([ask, call, image]) is None  # not seen yet: the next call shows it
    later = AIMessage(content="seen", id="a2")
    update = seen_image_update([ask, call, image, later, HumanMessage(content="again", id="h2")])
    assert [m.id for m in update["messages"]] == ["t1"]
    assert not _has_image(update["messages"][0])
    assert seen_image_update([ask]) is None
    no_id = image.model_copy(update={"id": None})
    assert seen_image_update([ask, call, no_id, later]) is None


def test_middleware_never_raises():
    middleware = SeenImagesMiddleware()
    assert middleware.before_model({"messages": "not a list"}, None) is None
    assert middleware.before_model(None, None) is None

    class Broken(list):
        def __getitem__(self, index):
            raise RuntimeError("boom")

    assert middleware.before_model({"messages": Broken([1, 2])}, None) is None


def test_main_graph_shows_an_image_once_then_keeps_a_note(tmp_path):
    repo = _repo_with_shot(tmp_path)
    seen: list = []
    saver = InMemorySaver()
    agent = build_agent(
        repo,
        _CONFIG,
        saver,
        chat_model=scripted_model(_image_then_more(seen)),
        model_factory=role_models({}, default=_ok()),
    )
    config = {"configurable": {"thread_id": "t-main"}}
    agent.invoke({"messages": [{"role": "user", "content": "look at the screenshot"}]}, config)
    assert len(seen) == 3
    first_result = _tool_messages(seen[1])[0]
    assert first_result.name == "read_file"
    assert _has_image(first_result)  # the call right after the read sees the image
    later = _tool_messages(seen[2])[0]
    assert not _has_image(later)
    assert later.content == [{"type": "text", "text": _note("/shot.png")}]
    assert _PNG_B64 not in str(seen[2])
    stored = agent.get_state(config).values["messages"]
    assert _PNG_B64 not in str(stored)  # the checkpoint itself no longer holds the data
    stored_result = next(m for m in stored if isinstance(m, ToolMessage) and m.name == "read_file")
    assert stored_result.id == first_result.id and stored_result.tool_call_id == "read-1"
    assert sum(isinstance(m, ToolMessage) and m.name == "read_file" for m in stored) == 1  # replaced, not added


def test_text_next_to_an_image_survives_in_the_graph(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    @tool(response_format="content")
    def screenshot() -> list:
        """Take a screenshot (test double)."""
        return [
            {"type": "text", "text": "window 800x600"},
            {"type": "image", "base64": _PNG_B64, "mime_type": "image/png"},
        ]

    seen: list = []

    def respond(messages):
        seen.append(list(messages))
        done = _tool_messages(messages)
        if not done:
            return tool_call("screenshot", {}, "shot-1")
        if len(done) == 1:
            return tool_call("ls", {"path": "/"}, "ls-1")
        return AIMessage(content="done")

    agent = build_agent(
        repo,
        _CONFIG,
        InMemorySaver(),
        chat_model=scripted_model(respond),
        model_factory=role_models({}, default=_ok()),
        extra_tools=[screenshot],
    )
    config = {"configurable": {"thread_id": "t-text"}}
    agent.invoke({"messages": [{"role": "user", "content": "shoot"}]}, config)
    assert _has_image(_tool_messages(seen[1])[0])
    trimmed = _tool_messages(seen[2])[0]
    assert trimmed.content == [
        {"type": "text", "text": "window 800x600"},
        {"type": "text", "text": _note("screenshot")},
    ]
    assert _PNG_B64 not in str(agent.get_state(config).values["messages"])


def test_worker_graph_trims_images_it_has_seen(tmp_path):
    repo = _repo_with_shot(tmp_path)
    worker_seen: list = []

    def main(messages):
        if any(isinstance(m, ToolMessage) and m.name == "task" for m in messages):
            return AIMessage(content="main done")
        return tool_calls(task_call("general-purpose", "look at shot.png", "t1"))

    agent = build_agent(
        repo,
        _CONFIG,
        InMemorySaver(),
        chat_model=scripted_model(main),
        model_factory=role_models(
            {"general-purpose": scripted_model(_image_then_more(worker_seen))}, default=_ok()
        ),
    )
    agent.invoke({"messages": [{"role": "user", "content": "delegate"}]}, {"configurable": {"thread_id": "t-w"}})
    assert len(worker_seen) == 3
    assert _has_image(_tool_messages(worker_seen[1])[0])
    assert not _has_image(_tool_messages(worker_seen[2])[0])
    assert _PNG_B64 not in str(worker_seen[2])


def test_explore_worker_and_background_role_graph_trim_too(tmp_path):
    repo = _repo_with_shot(tmp_path)
    roles = {d.name: d for d in builtin_agents()}
    for name in ("explore", "general-purpose"):
        seen: list = []
        saver = InMemorySaver()
        graph = build_role_graph(
            repo,
            _CONFIG,
            saver,
            roles[name],
            model_factory=role_models({name: scripted_model(_image_then_more(seen))}),
        )
        config = {"configurable": {"thread_id": f"bg-{name}"}}
        graph.invoke({"messages": [{"role": "user", "content": "look"}]}, config)
        assert len(seen) == 3, name
        assert _has_image(_tool_messages(seen[1])[0]), name
        assert not _has_image(_tool_messages(seen[2])[0]), name
        assert _PNG_B64 not in str(graph.get_state(config).values["messages"]), name
