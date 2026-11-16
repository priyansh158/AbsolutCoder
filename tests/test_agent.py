"""The deep agent is built with one worker, project memory, and image blocks."""

import base64

import pytest
from langchain_core.messages import AIMessage

from absolutcoder.agent import MEMORY_PATH, build_agent
from absolutcoder.attachments import build_user_content
from absolutcoder.config import Config
from absolutcoder.models import ProviderMissing
from tests.support import role_models, scripted_model, tool_call

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _ok_model():
    return scripted_model(lambda _messages: AIMessage(content="ok"))


def test_build_agent_uses_memory_and_image_blocks(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    config = Config(main_model="gpt-main", subagent_model="worker-test", auto_run=False)
    agent = build_agent(
        repo,
        config,
        None,
        chat_model=_ok_model(),
        model_factory=role_models({"general-purpose": _ok_model()}),
    )

    assert _memory_sources(agent) == [MEMORY_PATH]
    assert "TodoListMiddleware.after_model" in agent.nodes
    assert "HumanInTheLoopMiddleware.after_model" in agent.nodes

    image = tmp_path / "shot.png"
    image.write_bytes(_PNG)
    blocks = build_user_content("look at this", [image])
    image_blocks = [block for block in blocks if block["type"] == "image"]
    assert image_blocks
    assert image_blocks[0]["mime_type"] == "image/png"
    assert str(image.resolve()) in image_blocks[0]["path"]

    notes = tmp_path / "notes.log"
    notes.write_text("log", encoding="utf-8")
    other = build_user_content("read this", [notes])
    assert any(block["type"] == "text" and str(notes.resolve()) in block["text"] for block in other)
    assert not any(block["type"] == "image" for block in other)


def _memory_sources(agent) -> list[str]:
    middleware = agent.nodes["MemoryMiddleware.before_agent"].bound.func.__self__
    return list(middleware.sources)



def test_subagent_model_comes_from_factory(tmp_path):
    seen: list = []
    config = Config(main_model="openai:main-test", subagent_model="openai:worker-test")
    build_agent(
        tmp_path,
        config,
        None,
        chat_model=_ok_model(),
        model_factory=role_models({"general-purpose": _ok_model()}, seen=seen),
    )
    assert seen == [("general-purpose", "openai:worker-test")]


def test_agent_models_override_worker_model(tmp_path):
    seen: list = []
    config = Config(
        main_model="openai:main-test",
        subagent_model="openai:worker-test",
        agent_models={"general-purpose": "anthropic:w"},
    )
    build_agent(
        tmp_path,
        config,
        None,
        chat_model=_ok_model(),
        model_factory=role_models({"general-purpose": _ok_model()}, seen=seen),
    )
    assert seen == [("general-purpose", "anthropic:w")]


def test_main_delegates_to_worker_offline(tmp_path):
    def main(messages):
        if any(getattr(m, "type", None) == "tool" and m.name == "task" for m in messages):
            return AIMessage(content="main done")
        return tool_call(
            "task", {"description": "say hi", "subagent_type": "general-purpose"}, "call-task"
        )

    config = Config(
        main_model="openai:main-test", subagent_model="openai:worker-test", auto_run=True
    )
    agent = build_agent(
        tmp_path,
        config,
        None,
        chat_model=scripted_model(main),
        model_factory=role_models(
            {"general-purpose": scripted_model(lambda _m: AIMessage(content="worker says hi"))}
        ),
    )
    result = agent.invoke({"messages": [{"role": "user", "content": "delegate"}]})
    messages = result["messages"]
    task_messages = [m for m in messages if getattr(m, "type", None) == "tool" and m.name == "task"]
    assert len(task_messages) == 1
    assert "worker says hi" in str(task_messages[0].content)
    assert messages[-1].content == "main done"


_GROQ_ERROR = ImportError(
    "Initializing ChatGroq requires the langchain-groq package. "
    "Please install it with `pip install langchain-groq`"
)


def test_missing_provider_package_becomes_hint(tmp_path):
    def factory(role, model_id):
        raise _GROQ_ERROR

    config = Config(main_model="groq:x", subagent_model="groq:x")
    with pytest.raises(ProviderMissing) as caught:
        build_agent(tmp_path, config, None, model_factory=factory)
    assert str(caught.value) == (
        "The langchain-groq package for this model is not installed. "
        "Install it with: pip install langchain-groq"
    )


def test_unknown_provider_becomes_provider_missing(tmp_path):
    def factory(role, model_id):
        raise ValueError("Unable to infer model provider for model='antropic:x'. Please specify.")

    config = Config(main_model="antropic:x", subagent_model="antropic:x")
    with pytest.raises(ProviderMissing, match="Unknown model provider in 'antropic:x'"):
        build_agent(tmp_path, config, None, model_factory=factory)
    # A string id that langchain cannot resolve is converted the same way.
    with pytest.raises(ProviderMissing, match="Unknown model provider in 'antropic:x'"):
        build_agent(tmp_path, config, None)


def test_other_factory_errors_become_provider_missing(tmp_path):
    def factory(role, model_id):
        raise RuntimeError("Missing credentials")

    config = Config(main_model="openai:x", subagent_model="openai:x")
    with pytest.raises(ProviderMissing, match="Could not set up openai:x: Missing credentials"):
        build_agent(tmp_path, config, None, model_factory=factory)
