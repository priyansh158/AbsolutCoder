"""Slash commands update the session and /resume restores the checkpoint."""

import base64
import io
import subprocess
from pathlib import Path

from langchain_core.messages import AIMessage
from rich.console import Console

from dataclasses import replace

import pytest

from absolutcoder.config import Config, config_path, save_config
from absolutcoder.models import ProviderMissing
from absolutcoder.repl import Repl
from absolutcoder.sessions import create_session, list_sessions, load_session
from tests.support import role_models, scripted_model, tool_call, tool_names_since_last_human

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _worker():
    return role_models({"general-purpose": scripted_model(lambda _m: AIMessage(content="worker"))})


def _console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, no_color=True, width=100)


def test_model_command_updates_session_on_the_same_thread(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    config = Config(main_model="openai:main-test", subagent_model="openai:worker-test", auto_run=True)
    repl = Repl(
        repo,
        config,
        chat_model=scripted_model(lambda _messages: AIMessage(content="ok")),
        model_factory=_worker(),
        console=_console(),
    )
    thread_id = repl.thread_id
    repl.handle_line("/model gpt-other")
    loaded = load_session(repl.session.id)
    assert loaded.main_model == "openai:gpt-other"
    assert loaded.subagent_model == "openai:worker-test"
    assert repl.thread_id == thread_id == loaded.id


def test_shell_interrupt_asks_before_execute(tmp_path):
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
        model_factory=_worker(),
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


def test_stream_exception_stays_in_the_session(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    output = io.StringIO()
    lines = iter(["build the game", "continue", "/exit"])
    repl = Repl(
        repo,
        Config(main_model="openai:main-test", subagent_model="openai:worker-test", auto_run=True),
        chat_model=scripted_model(lambda _messages: AIMessage(content="ok")),
        model_factory=_worker(),
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


def test_resume_restores_checkpoint_image_path_and_models(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    image = tmp_path / "shot.png"
    image.write_bytes(_PNG)
    fake = scripted_model(lambda _messages: AIMessage(content="seen"))
    original = Repl(
        repo,
        Config(main_model="openai:main-test", subagent_model="openai:worker-test", auto_run=True),
        chat_model=fake,
        model_factory=_worker(),
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
        model_factory=_worker(),
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


def test_resume_with_unavailable_provider_prints_hint(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    missing = ImportError(
        "Initializing ChatGroq requires the langchain-groq package. "
        "Please install it with `pip install langchain-groq`"
    )
    fine = _worker()

    def factory(role, model_id):
        if model_id.startswith("groq:"):
            raise missing
        return fine(role, model_id)

    output = io.StringIO()
    repl = Repl(
        repo,
        Config(main_model="openai:main-test", subagent_model="openai:worker-test", auto_run=True),
        chat_model=scripted_model(lambda _messages: AIMessage(content="ok")),
        model_factory=factory,
        console=Console(file=output, force_terminal=False, no_color=True, width=100),
    )
    original_id = repl.thread_id
    other = create_session(repo, "groq:x", "groq:x")
    ids = [session.id for session in list_sessions(repo)]
    repl.input_fn = lambda _prompt="": str(ids.index(other.id) + 1)
    repl.handle_line("/resume")

    text = output.getvalue()
    assert "pip install langchain-groq" in text
    assert "Traceback" not in text
    assert repl.thread_id == original_id
    assert repl.session.id == original_id
    assert repl.config.main_model == "openai:main-test"
    snapshot = repl.run_turn("still there?")
    assert snapshot is not None
    assert "ok" in _flatten(snapshot.values.get("messages", []))


_ANTHROPIC_ERROR = ImportError(
    "Initializing ChatAnthropic requires the langchain-anthropic package. "
    "Please install it with `pip install langchain-anthropic`"
)


def _factory_repl(tmp_path, fail, *, auto_run=True):
    """A Repl whose main model also comes from the factory; ``fail(model_id)`` may raise."""

    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    models = role_models(
        {
            "main": scripted_model(lambda _m: AIMessage(content="ok")),
            "general-purpose": scripted_model(lambda _m: AIMessage(content="worker")),
        }
    )

    def factory(role, model_id):
        fail(model_id)
        return models(role, model_id)

    output = io.StringIO()
    config = Config(main_model="openai:main-test", subagent_model="openai:worker-test", auto_run=auto_run)
    repl = Repl(
        repo,
        config,
        model_factory=factory,
        console=Console(file=output, force_terminal=False, no_color=True, width=100),
    )
    return repl, output


def _assert_unchanged(repl, output, before, config_text):
    text = output.getvalue()
    assert "Traceback" not in text
    assert repl.thread_id == before["thread_id"]
    assert repl.session == before["session"]
    assert repl.config == before["config"]
    assert config_path().read_text(encoding="utf-8") == config_text
    snapshot = repl.run_turn("still there?")
    assert snapshot is not None
    assert "ok" in _flatten(snapshot.values.get("messages", []))


def _snapshot(repl):
    save_config(config_path(), repl.config)
    return (
        {"thread_id": repl.thread_id, "session": repl.session, "config": repl.config},
        config_path().read_text(encoding="utf-8"),
    )


def test_install_hint_keeps_its_extra_in_console(tmp_path):
    def fail(model_id):
        if model_id.startswith("anthropic:"):
            raise _ANTHROPIC_ERROR

    repl, output = _factory_repl(tmp_path, fail)
    repl.handle_line("/model anthropic:x")
    text = output.getvalue()
    assert 'pip install "absolutcoder[anthropic]"' in text
    assert "langchain-anthropic package" in text


def test_model_switch_with_unknown_provider_keeps_session(tmp_path):
    def fail(model_id):
        if model_id.startswith("antropic:"):
            raise ValueError("Unable to infer model provider for model='antropic:x'. Specify it.")

    repl, output = _factory_repl(tmp_path, fail)
    before, text = _snapshot(repl)
    repl.handle_line("/model antropic:x")
    repl.handle_line("/subagent antropic:x")
    assert "Unknown model provider" in output.getvalue()
    _assert_unchanged(repl, output, before, text)


def test_model_switch_with_invalid_id_keeps_session(tmp_path):
    repl, output = _factory_repl(tmp_path, lambda model_id: None)
    before, text = _snapshot(repl)
    repl.handle_line("/model a b:x")
    repl.handle_line("/subagent a b:x")
    assert "Invalid model id" in output.getvalue()
    _assert_unchanged(repl, output, before, text)


def test_resume_with_missing_key_prints_variable(tmp_path, monkeypatch):
    repl, output = _factory_repl(tmp_path, lambda model_id: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    repl.model_factory = None
    result = repl._try_build(replace(repl.config, main_model="anthropic:x"))
    assert result is None
    assert "ANTHROPIC_API_KEY is not set" in output.getvalue()


def test_auto_does_not_report_success_when_rebuild_fails(tmp_path):
    state = {"fail": False}

    def fail(model_id):
        if state["fail"]:
            raise _ANTHROPIC_ERROR

    repl, output = _factory_repl(tmp_path, fail, auto_run=False)
    before, text = _snapshot(repl)
    state["fail"] = True
    repl.handle_line("/auto on")
    assert "auto_run on" not in output.getvalue()
    state["fail"] = False
    _assert_unchanged(repl, output, before, text)


def test_failed_start_leaves_no_session_row(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def factory(role, model_id):
        raise ImportError(
            "Initializing ChatGroq requires the langchain-groq package. "
            "Please install it with `pip install langchain-groq`"
        )

    with pytest.raises(ProviderMissing, match="langchain-groq"):
        Repl(
            repo,
            Config(main_model="groq:x", subagent_model="groq:x"),
            model_factory=factory,
            console=_console(),
        )
    assert list_sessions(repo) == []


def test_agent_model_ids_are_normalised(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    repl = Repl(
        repo,
        Config(main_model="a", subagent_model="b", agent_models={"general-purpose": "c"}),
        chat_model=scripted_model(lambda _m: AIMessage(content="ok")),
        model_factory=_worker(),
        console=_console(),
    )
    assert repl.config.agent_models == {"general-purpose": "openai:c"}


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
