"""The deep agent is built with one worker, project memory, and image blocks."""

import base64

from absolutcoder.agent import MEMORY_PATH, build_agent
from absolutcoder.attachments import build_user_content
from absolutcoder.config import Config

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def test_build_agent_uses_worker_memory_and_image_blocks(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    repo = tmp_path / "repo"
    repo.mkdir()
    from tests.support import scripted_model
    from langchain_core.messages import AIMessage

    config = Config(main_model="gpt-main", subagent_model="worker-test", auto_run=False)
    agent = build_agent(repo, config, None, chat_model=scripted_model(lambda _messages: AIMessage(content="ok")))

    worker = _worker_model(agent)
    assert worker.model_name == "worker-test"
    assert type(worker).__name__ == "ChatOpenAI"
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


def _worker_model(agent):
    tools = agent.nodes["tools"].bound.func.__self__
    task = tools.tools_by_name["task"].func
    found = []

    def walk(obj, depth: int, seen: set[int]) -> None:
        if obj is None or depth > 12 or id(obj) in seen:
            return
        seen.add(id(obj))
        if type(obj).__name__ == "ChatOpenAI":
            found.append(obj)
            return
        if isinstance(obj, dict):
            for value in obj.values():
                walk(value, depth + 1, seen)
            return
        if isinstance(obj, (list, tuple, set, frozenset)):
            for value in obj:
                walk(value, depth + 1, seen)
            return
        closure = getattr(obj, "__closure__", None)
        if closure:
            for cell in closure:
                try:
                    walk(cell.cell_contents, depth + 1, seen)
                except ValueError:
                    pass
        for attr in ("bound", "func", "nodes"):
            if hasattr(obj, attr):
                walk(getattr(obj, attr), depth + 1, seen)

    walk(task, 0, set())
    assert found, "worker model was not compiled onto the task tool"
    return found[0]
