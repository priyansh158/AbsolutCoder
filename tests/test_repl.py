"""Slash commands update the session and /resume restores the checkpoint."""

import base64
import io
import subprocess
from pathlib import Path

from langchain_core.messages import AIMessage
from rich.console import Console

from absolutcoder.config import Config
from absolutcoder.repl import Repl
from absolutcoder.sessions import list_sessions, load_session
from tests.support import scripted_model, tool_call, tool_names_since_last_human

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, no_color=True, width=100)


def test_model_command_updates_session_on_the_same_thread(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    repo = tmp_path / "repo"
    repo.mkdir()
    config = Config(main_model="openai:main-test", subagent_model="openai:worker-test", auto_run=True)
    repl = Repl(
        repo,
        config,
        chat_model=scripted_model(lambda _messages: AIMessage(content="ok")),
        console=_console(),
    )
    thread_id = repl.thread_id
    repl.handle_line("/model gpt-other")
    loaded = load_session(repl.session.id)
    assert loaded.main_model == "openai:gpt-other"
    assert loaded.subagent_model == "openai:worker-test"
    assert repl.thread_id == thread_id == loaded.id


def test_shell_interrupt_asks_before_execute(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    repo = _git_repo(tmp_path / "repo")
    prompts: list[str] = []

    def ask(prompt=""):
        prompts.append(prompt)
        return "y"

    def responder(messages):
        if "execute" not in tool_names_since_last_human(messages):
            return tool_call("execute", {"command": "git status"}, "call-status")
        return AIMessage(content="status done")

    repl = Repl(
        repo,
        Config(main_model="openai:main-test", subagent_model="openai:worker-test", auto_run=False),
        chat_model=scripted_model(responder),
        input_fn=ask,
        console=_console(),
    )
    snapshot = repl.run_turn("show git status")
    assert prompts
    assert any("git status" in entry["command"] and entry["allowed"] for entry in repl.turn_vet_log)
    assert "status done" in _flatten(snapshot.values.get("messages", []))


def _git_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    return path


def test_stream_exception_stays_in_the_session(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    repo = tmp_path / "repo"
    repo.mkdir()
    output = io.StringIO()
    lines = iter(["build the game", "continue", "/exit"])
    repl = Repl(
        repo,
        Config(main_model="openai:main-test", subagent_model="openai:worker-test", auto_run=True),
        chat_model=scripted_model(lambda _messages: AIMessage(content="ok")),
        input_fn=lambda _prompt="": next(lines),
        console=Console(file=output, force_terminal=False, no_color=True, width=100),
    )
    thread_id = repl.thread_id
    real_stream = repl.agent.stream
    state = {"raised": False}

    def stream(*args, **kwargs):
        if not state["raised"]:
            state["raised"] = True
            raise RuntimeError("tool exploded")
        return real_stream(*args, **kwargs)

    repl.agent.stream = stream
    repl.loop()
    text = output.getvalue()
    assert "tool exploded" in text
    assert "Traceback" not in text
    assert "ok" in text
    assert repl.thread_id == thread_id
    assert state["raised"] is True


def test_resume_restores_checkpoint_image_path_and_models(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    repo = tmp_path / "repo"
    repo.mkdir()
    image = tmp_path / "shot.png"
    image.write_bytes(_PNG)
    fake = scripted_model(lambda _messages: AIMessage(content="seen"))
    original = Repl(
        repo,
        Config(main_model="openai:main-test", subagent_model="openai:worker-test", auto_run=True),
        chat_model=fake,
        console=_console(),
    )
    original.run_turn("look at this", paths=[image])
    original_id = original.session.id

    def choose(_prompt=""):
        for index, session in enumerate(list_sessions(repo), start=1):
            if session.id == original_id:
                return str(index)
        raise AssertionError("original session was not listed")

    resumed = Repl(
        repo,
        Config(main_model="openai:other-main", subagent_model="openai:other-worker", auto_run=False),
        chat_model=fake,
        input_fn=choose,
        console=_console(),
    )
    resumed.handle_line("/resume")
    assert resumed.thread_id == original_id
    assert resumed.session.main_model == "openai:main-test"
    assert resumed.session.subagent_model == "openai:worker-test"
    state = resumed.agent.get_state({"configurable": {"thread_id": resumed.thread_id}})
    blob = _flatten(state.values.get("messages", []))
    assert str(image.resolve()) in blob


def _flatten(messages) -> str:
    parts: list[str] = []
    for message in messages:
        content = getattr(message, "content", message)
        if isinstance(content, str):
            parts.append(content)
            continue
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    parts.append(str(block.get("text", "")))
                    parts.append(str(block.get("path", "")))
                else:
                    parts.append(str(block))
            continue
        parts.append(str(content))
    return "\n".join(parts)
