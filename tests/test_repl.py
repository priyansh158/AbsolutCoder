"""Slash commands update the session and /resume restores the checkpoint."""

import base64
import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import AIMessage, AIMessageChunk
from rich.console import Console

from dataclasses import replace

import pytest

from absolutcoder.config import Config, config_path, save_config
from absolutcoder.git_policy import TurnFlags, render_command
from absolutcoder.models import ProviderMissing
from absolutcoder.repl import Repl, _resume_map
from absolutcoder.sessions import create_session, list_sessions, load_session
from tests.support import (
    last_human_text,
    role_models,
    scripted_model,
    task_call,
    tool_call,
    tool_calls,
    tool_names_since_last_human,
)

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _worker():
    worker = scripted_model(lambda _m: AIMessage(content="worker"))
    return role_models({"general-purpose": worker}, default=worker)


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


def test_turn_prints_token_line_when_models_report_usage(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    output = io.StringIO()
    reply = AIMessage(
        content="ok",
        usage_metadata={"input_tokens": 1200, "output_tokens": 30, "total_tokens": 1230},
    )
    repl = Repl(
        repo,
        Config(main_model="openai:main-test", subagent_model="openai:worker-test", auto_run=True),
        chat_model=scripted_model(lambda _messages: reply),
        model_factory=_worker(),
        input_fn=lambda _prompt="": "hello",
        console=Console(file=output, force_terminal=False, no_color=True, width=100),
    )
    repl.run_turn("hello")
    lines = [line for line in output.getvalue().splitlines() if line.strip()]
    assert lines[-1] == "tokens: 1,200 in (0 cached), 30 out, 1 model call"


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
        },
        default=scripted_model(lambda _m: AIMessage(content="worker")),
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


def test_agents_command_lists_roster(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    output = io.StringIO()
    repl = Repl(
        repo,
        Config(main_model="openai:main-test", subagent_model="openai:worker-test"),
        chat_model=scripted_model(lambda _m: AIMessage(content="ok")),
        model_factory=_worker(),
        console=Console(file=output, force_terminal=False, no_color=True, width=200),
    )
    repl.handle_line("/agents")
    lines = output.getvalue().splitlines()
    for name in ("explore", "plan", "general-purpose", "web-researcher"):
        assert any(line.startswith(f"{name}  [builtin]  ") for line in lines), name


def _plain_repl(tmp_path, **kwargs):
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    output = io.StringIO()
    repl = Repl(
        repo,
        Config(main_model="openai:main-test", subagent_model="openai:worker-test"),
        chat_model=scripted_model(lambda _m: AIMessage(content="ok")),
        model_factory=_worker(),
        console=Console(file=output, force_terminal=False, no_color=True, width=200),
        **kwargs,
    )
    return repl, output


def _approval(interrupt_id, command):
    value = {"action_requests": [{"name": "execute", "args": {"command": command}}]}
    return SimpleNamespace(id=interrupt_id, value=value)


def test_interrupt_without_id_fails_the_turn_once(tmp_path):
    repl, output = _plain_repl(tmp_path, input_fn=lambda _prompt="": "y")
    calls = {"stream": 0}
    pending = SimpleNamespace(tasks=[SimpleNamespace(interrupts=[_approval(None, "echo x")])])

    class StuckGraph:
        def stream(self, *args, **kwargs):
            calls["stream"] += 1
            if calls["stream"] > 3:  # keep a regression from hanging the suite
                raise AssertionError("re-streamed after an id-less interrupt")
            return iter(())

        def get_state(self, config):
            return pending

    repl.agent = StuckGraph()
    assert repl.run_turn("go") is None
    lines = [line for line in output.getvalue().splitlines() if line.startswith("Error:")]
    assert lines == ["Error: Cannot resume: an approval request has no id."]
    assert calls["stream"] == 1
    assert "Run command?" not in output.getvalue()


def test_decisions_are_keyed_by_interrupt_id(tmp_path):
    answers = iter(["y", "n"])
    repl, _output = _plain_repl(tmp_path, input_fn=lambda _prompt="": next(answers))
    pair = [_approval("id-1", "echo one"), _approval("id-2", "echo two")]
    resume = _resume_map(pair, repl._decisions(pair))
    assert resume == {
        "id-1": {"decisions": [{"type": "approve"}]},
        "id-2": {"decisions": [{"type": "reject", "message": "User rejected the command."}]},
    }

    repl.input_fn = lambda _prompt="": "y"
    single = [_approval("id-3", "echo three")]
    assert _resume_map(single, repl._decisions(single)) == {"decisions": [{"type": "approve"}]}


def test_web_tools_are_reused_across_rebuilds(tmp_path, monkeypatch):
    import absolutcoder.agent as agent_module

    received: list = []
    real_build_specs = agent_module.build_specs

    def recording_build_specs(*args, **kwargs):
        received.append(kwargs.get("web_tools"))
        return real_build_specs(*args, **kwargs)

    monkeypatch.setattr(agent_module, "build_specs", recording_build_specs)
    repl, output = _plain_repl(tmp_path)
    repl.handle_line("/auto on")
    assert "auto_run on" in output.getvalue()
    assert len(received) == 2
    assert received[0] is not None
    assert all(tools is repl.web_tools for tools in received)
    assert [tool.name for tool in repl.web_tools] == ["web_fetch", "web_search"]


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


def _custom_files_repo(tmp_path, files):
    repo = tmp_path / "repo"
    folder = repo / ".absolutcoder" / "agents"
    folder.mkdir(parents=True)
    for name, text in files.items():
        (folder / name).write_text(text, encoding="utf-8")
    return repo


def test_startup_reports_custom_roles_and_warnings(tmp_path):
    repo = _custom_files_repo(
        tmp_path,
        {
            "reviewer.md": "---\ndescription: Reviews code.\n---\nYou review.\n",
            "broken.md": "no frontmatter here\n",
        },
    )
    output = io.StringIO()
    repl = Repl(
        repo,
        Config(main_model="openai:main-test", subagent_model="openai:worker-test"),
        chat_model=scripted_model(lambda _m: AIMessage(content="ok")),
        model_factory=_worker(),
        console=Console(file=output, force_terminal=False, no_color=True, width=200),
    )
    lines = output.getvalue().splitlines()
    assert "agents: 4 built-in, 0 home, 1 repo" in lines
    warnings = [line for line in lines if line.startswith("agents: broken.md")]
    assert len(warnings) == 1
    assert sum(line.startswith("agents: ") for line in lines) == 2
    repl.handle_line("/agents")
    assert any(line.startswith("reviewer  [repo]  ") for line in output.getvalue().splitlines())


def test_startup_is_silent_without_custom_roles(tmp_path):
    repl, output = _plain_repl(tmp_path)
    assert not any(line.startswith("agents:") for line in output.getvalue().splitlines())


# --- streaming display ----------------------------------------------------------


def _no_web(monkeypatch):
    """Replace the real web tools with test doubles so no HTTP client is created."""

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


def test_turn_output_shows_subagent_activity(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    repo = tmp_path / "repo"
    repo.mkdir()

    def main(messages):
        if "task" in tool_names_since_last_human(messages):
            return AIMessage(content="main final answer")
        return tool_calls(task_call("explore", "find the config", "t1"))

    def explorer(messages):
        if "grep" not in tool_names_since_last_human(messages):
            return tool_call("grep", {"pattern": "config"}, "g1")
        return AIMessage(content="explorer report")

    output = io.StringIO()
    repl = Repl(
        repo,
        Config(main_model="openai:main-test", subagent_model="openai:worker-test", auto_run=True),
        chat_model=scripted_model(main),
        model_factory=role_models({"explore": scripted_model(explorer)}, default=scripted_model(lambda _m: AIMessage(content="ok"))),
        console=Console(file=output, force_terminal=False, no_color=True, width=200),
    )
    assert repl.run_turn("where is the config?") is not None
    text = output.getvalue()
    lines = text.splitlines()
    assert "  task [explore]: find the config" in lines
    assert any(line.startswith("  [explore] tool: grep(") for line in lines)
    assert "  task [explore] finished" in lines
    assert text.count("main final answer") == 1
    assert "explorer report" not in text


def _interrupt_after_first_part(real_stream, exc):
    """A stream that yields one real part, then raises ``exc``; later calls are real."""

    state = {"raised": False}

    def stream(*args, **kwargs):
        if state["raised"]:
            yield from real_stream(*args, **kwargs)
            return
        state["raised"] = True
        yield next(iter(real_stream(*args, **kwargs)))
        raise exc

    return stream, state


def test_keyboard_interrupt_mid_stream_keeps_session(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    repl, output = _plain_repl(tmp_path)
    thread_id = repl.thread_id
    stream, state = _interrupt_after_first_part(repl.agent.stream, KeyboardInterrupt())
    repl.agent.stream = stream
    assert repl.run_turn("first") is None
    assert state["raised"] is True
    text = output.getvalue()
    assert "interrupted" in text.splitlines()
    assert "Traceback" not in text
    assert repl.thread_id == thread_id
    assert repl.run_turn("second") is not None
    assert "ok" in output.getvalue().splitlines()
    assert repl.thread_id == thread_id


def test_keyboard_interrupt_at_approval_prompt_keeps_session(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    repo = _git_repo(tmp_path / "repo")
    state = {"interrupt": True}

    def ask(prompt=""):
        if state["interrupt"]:
            state["interrupt"] = False
            raise KeyboardInterrupt
        return "y"

    def responder(messages):
        if "execute" not in tool_names_since_last_human(messages):
            return tool_call("execute", {"command": "git status"}, "call-" + str(len(messages)))
        return AIMessage(content="answer after " + tool_names_since_last_human(messages)[-1])

    output = io.StringIO()
    repl = Repl(
        repo,
        Config(main_model="openai:main-test", subagent_model="openai:worker-test", auto_run=False),
        chat_model=scripted_model(responder),
        model_factory=_worker(),
        input_fn=ask,
        console=Console(file=output, force_terminal=False, no_color=True, width=200),
    )
    thread_id = repl.thread_id
    assert repl.run_turn("show git status") is None
    text = output.getvalue()
    assert "Run command? git status" in text
    assert "interrupted" in text.splitlines()
    assert "Traceback" not in text
    assert repl.thread_id == thread_id
    snapshot = repl.run_turn("try again")
    assert snapshot is not None
    assert "answer after execute" in output.getvalue().splitlines()
    assert repl.thread_id == thread_id


def test_stream_error_mid_line_starts_a_new_line(tmp_path, monkeypatch):
    import absolutcoder.repl as repl_module
    from absolutcoder.render import StreamRenderer

    _no_web(monkeypatch)
    monkeypatch.setattr(repl_module, "StreamRenderer", lambda console: StreamRenderer(console, live=True))
    repl, output = _plain_repl(tmp_path)
    chunk = AIMessageChunk(content="partial text", id="m1")

    def stream(*_args, **_kwargs):
        yield {"type": "messages", "ns": (), "data": (chunk, {"langgraph_node": "model"})}
        raise RuntimeError("stream broke")

    repl.agent.stream = stream
    assert repl.run_turn("go") is None
    lines = output.getvalue().splitlines()
    assert "partial text" in lines
    assert "Error: stream broke" in lines


def test_approval_prompt_follows_tool_line(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    repo = _git_repo(tmp_path / "repo")

    def responder(messages):
        if "execute" not in tool_names_since_last_human(messages):
            return tool_call("execute", {"command": "git status"}, "call-status")
        return AIMessage(content="status done")

    output = io.StringIO()
    repl = Repl(
        repo,
        Config(main_model="openai:main-test", subagent_model="openai:worker-test", auto_run=False),
        chat_model=scripted_model(responder),
        model_factory=_worker(),
        input_fn=lambda _prompt="": "y",
        console=Console(file=output, force_terminal=False, no_color=True, width=200),
    )
    repl.run_turn("show git status")
    text = output.getvalue()
    tool_at = text.index("  tool: execute(git status)")
    prompt_at = text.index("Run command?")
    assert tool_at < prompt_at
    assert text.count("tool: execute(") == 1
    assert text.count("status done") == 1


# --- check loop and /loop -------------------------------------------------------


def _python_check(repl, code: str) -> str:
    return render_command([sys.executable, "-c", code], repl.backend.shell_kind)


def _loop_repl(repo, responder, **config_fields):
    output = io.StringIO()
    repl = Repl(
        repo,
        Config(main_model="openai:main-test", subagent_model="openai:worker-test", auto_run=True, **config_fields),
        chat_model=scripted_model(responder),
        model_factory=_worker(),
        console=Console(file=output, force_terminal=False, no_color=True, width=200),
    )
    return repl, output


def test_verify_loop_feeds_failure_back(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    repo = tmp_path / "repo"
    repo.mkdir()
    human_texts: list[str] = []

    def responder(messages):
        human = last_human_text(messages)
        human_texts.append(human)
        if "The check command failed" in human and "write_file" not in tool_names_since_last_human(messages):
            return tool_call("write_file", {"file_path": "/ok.txt", "content": "ok\n"}, f"w{len(messages)}")
        return AIMessage(content="done")

    repl, output = _loop_repl(repo, responder)
    command = _python_check(repl, "import os, sys; sys.exit(0 if os.path.exists('ok.txt') else 1)")
    repl.config = replace(repl.config, verify_command=command)
    outcome = repl.run_task("make the check pass")
    lines = output.getvalue().splitlines()
    assert outcome.passed is True
    assert (repo / "ok.txt").exists()
    assert "verify: passed after 2 round(s)" in lines
    assert lines.count(f"verify: {command}") == 2
    assert any("The check command failed (exit code 1)" in text for text in human_texts)


def test_check_uses_verify_timeout_and_reports_each_failing_round(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    repo = tmp_path / "repo"
    repo.mkdir()
    repl, output = _loop_repl(repo, lambda _m: AIMessage(content="done"), max_rounds=3, verify_timeout=77)
    command = _python_check(repl, "raise SystemExit(4)")
    repl.config = replace(repl.config, verify_command=command)
    timeouts = []
    real_execute = repl.backend.execute

    def execute(cmd, *, timeout=None):
        timeouts.append(timeout)
        return real_execute(cmd, timeout=timeout)

    monkeypatch.setattr(repl.backend, "execute", execute)
    outcome = repl.run_task("fix it")
    lines = output.getvalue().splitlines()
    assert outcome.passed is False
    assert timeouts == [77, 77, 77]
    assert [line for line in lines if line.startswith("verify: failed")] == [
        "verify: failed (exit 4), round 1 of 3",
        "verify: failed (exit 4), round 2 of 3",
        "verify: failed (exit 4), round 3 of 3",
    ]


def test_loop_start_names_the_check_command(tmp_path):
    repl, output = _plain_repl(tmp_path, input_fn=lambda _p="": "/exit")
    repl.loop()
    assert "verify:" not in output.getvalue()
    repl, output = _plain_repl(tmp_path, input_fn=lambda _p="": "/exit")
    repl.config = replace(repl.config, verify_command="pytest -q")
    repl.loop()
    assert "verify: pytest -q (runs after every request; /verify off to stop)" in output.getvalue().splitlines()


def test_verify_follow_up_cannot_grant_push(tmp_path, monkeypatch, git_repo):
    _no_web(monkeypatch)
    repo = git_repo(tmp_path / "repo")

    def responder(messages):
        human = last_human_text(messages)
        if "The check command failed" in human and "execute" not in tool_names_since_last_human(messages):
            return tool_call("execute", {"command": "git push"}, f"p{len(messages)}")
        return AIMessage(content="done")

    repl, output = _loop_repl(repo, responder, max_rounds=2)
    command = _python_check(repl, "print('push'); raise SystemExit(1)")
    repl.config = replace(repl.config, verify_command=command)
    outcome = repl.run_task("tidy the readme")
    assert outcome.passed is False
    pushes = [entry for entry in repl.backend.vet_log if entry["command"] == "git push"]
    assert pushes
    assert not any(entry["allowed"] for entry in pushes)
    assert "verify: still failing after 2 round(s)" in output.getvalue().splitlines()


def test_verify_command_is_saved_and_cleared(tmp_path):
    repl, output = _plain_repl(tmp_path)
    agent = repl.agent
    repl.handle_line("/verify pytest -q")
    assert json.loads(config_path().read_text(encoding="utf-8"))["verify_command"] == "pytest -q"
    repl.handle_line("/verify")
    assert "verify: pytest -q" in output.getvalue().splitlines()
    repl.handle_line("/verify off")
    assert json.loads(config_path().read_text(encoding="utf-8"))["verify_command"] is None
    repl.handle_line("/verify")
    assert output.getvalue().splitlines()[-1] == "verify: off"
    assert repl.agent is agent


def test_loop_command_repeats_and_stops(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    sleeps: list[float] = []

    def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 2:
            raise KeyboardInterrupt

    repl, output = _plain_repl(tmp_path, sleep_fn=sleep)
    repl.handle_line("/loop 1s say hi")
    lines = output.getvalue().splitlines()
    assert [line for line in lines if line.startswith("loop: iteration")] == [
        "loop: iteration 1",
        "loop: iteration 2",
    ]
    assert lines[-1] == "loop: stopped after 2 iteration(s)"
    assert sleeps == [1, 1]


def test_loop_usage_on_bad_input(tmp_path):
    repl, output = _plain_repl(tmp_path, sleep_fn=lambda _s: None)
    repl.handle_line("/loop soon hi")
    repl.handle_line("/loop 5m")
    lines = output.getvalue().splitlines()
    assert lines.count("Usage: /loop <N>(s|m|h) <prompt>") == 2
    assert not any(line.startswith("loop: iteration") for line in lines)


def test_refused_check_command_fails_the_task(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    repl, output = _plain_repl(tmp_path)
    repl.config = replace(repl.config, verify_command="git push")
    outcome = repl.run_task("tidy the readme")
    assert outcome.passed is False
    assert (
        "verify: command refused: git push is not allowed unless this message asks to push"
        in output.getvalue().splitlines()
    )


def _scripted_checks(repl, *exit_codes, error=None):
    """Replace the check command's execution: each call returns the next exit code, or raises ``error``."""

    from deepagents.backends.protocol import ExecuteResponse

    codes = list(exit_codes)
    calls: list[str] = []

    def execute(command, *, timeout=None):
        calls.append(command)
        if error is not None:
            raise error
        code = codes.pop(0) if codes else 0
        return ExecuteResponse(output=f"check exit {code}", exit_code=code)

    repl.backend.execute = execute
    return calls


def test_check_that_cannot_run_fails_the_task(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    repl, output = _plain_repl(tmp_path)
    repl.config = replace(repl.config, verify_command="make check")
    _scripted_checks(repl, error=OSError("shell went away"))
    outcome = repl.run_task("tidy the readme")
    assert outcome.passed is False
    lines = output.getvalue().splitlines()
    assert "verify: could not run the check: shell went away" in lines
    assert "Traceback" not in output.getvalue()


def test_interrupted_turn_stops_the_check_loop(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    repl, output = _plain_repl(tmp_path)
    repl.config = replace(repl.config, verify_command="make check")
    calls = _scripted_checks(repl, 1)
    thread_id = repl.thread_id
    stream, state = _interrupt_after_first_part(repl.agent.stream, KeyboardInterrupt())
    repl.agent.stream = stream
    outcome = repl.run_task("tidy the readme")
    assert state["raised"] is True
    assert outcome.passed is False
    assert calls == []
    assert "verify: stopped" in output.getvalue().splitlines()
    assert repl.thread_id == thread_id
    assert repl.run_turn("still there?") is not None


def test_interrupted_check_stops_the_check_loop(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    repl, output = _plain_repl(tmp_path)
    repl.config = replace(repl.config, verify_command="make check")
    calls = _scripted_checks(repl, error=KeyboardInterrupt())
    thread_id = repl.thread_id
    outcome = repl.run_task("tidy the readme")
    assert outcome.passed is False
    assert calls == ["make check"]
    assert "verify: stopped" in output.getvalue().splitlines()
    assert repl.thread_id == thread_id
    assert repl.run_turn("still there?") is not None


def test_loop_continues_after_a_failed_check(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    sleeps: list[float] = []

    def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 2:
            raise KeyboardInterrupt

    repl, output = _plain_repl(tmp_path, sleep_fn=sleep)
    repl.config = replace(repl.config, verify_command="make check", max_rounds=1)
    _scripted_checks(repl, 1, 0)
    repl.handle_line("/loop 1s say hi")
    lines = output.getvalue().splitlines()
    assert "verify: still failing after 1 round(s)" in lines
    assert "loop: iteration 2" in lines
    assert "verify: passed after 1 round(s)" in lines
    assert lines[-1] == "loop: stopped after 2 iteration(s)"


def test_run_turn_uses_given_flags(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    repl, _output = _plain_repl(tmp_path)
    repl.run_turn("please push", flags=TurnFlags())
    assert repl.backend.flags == TurnFlags()
    assert repl.last_turn_interrupted is False
    repl.run_turn("please push")
    assert repl.backend.flags.push_requested is True


# --- background tasks -----------------------------------------------------------

import re  # noqa: E402
import threading  # noqa: E402

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult  # noqa: E402

TASK_TOOLS = {"start_task", "check_task", "list_tasks", "cancel_task"}
TIMEOUT = 10
_UPDATE_HEADER = "[Background task updates — reported by the program, not typed by the user]"


def _reply(text, *, bound=None):
    return scripted_model(lambda _m: AIMessage(content=text), bound=bound)


def _blocked(release: threading.Event, started: threading.Event | None = None, text="late"):
    """A worker that waits for ``release`` before it answers ``text``."""

    def respond(_messages):
        if started is not None:
            started.set()
        assert release.wait(TIMEOUT), "release was never set"
        return AIMessage(content=text)

    return scripted_model(respond)


def _runs(command):
    """General-purpose worker: run ``command`` once, then report the tool output."""

    def respond(messages):
        if "execute" not in tool_names_since_last_human(messages):
            return tool_call("execute", {"command": command}, "exec-1")
        tools = [m for m in messages if getattr(m, "type", None) == "tool"]
        return AIMessage(content="tool said: " + str(tools[-1].content))

    return scripted_model(respond)


def _humans(messages) -> list:
    return [m for m in messages if getattr(m, "type", None) == "human"]


def _task_repl(tmp_path, main, workers, *, auto_run=True, repo=None, input_fn=None, main_bound=None, **kwargs):
    """A Repl with background tasks; ``main`` is a responder or a chat model."""

    if repo is None:
        repo = tmp_path / "repo"
        repo.mkdir(exist_ok=True)
    output = io.StringIO()
    chat = main if isinstance(main, BaseChatModel) else scripted_model(main, bound=main_bound)
    repl = Repl(
        repo,
        Config(main_model="openai:main-test", subagent_model="openai:worker-test", auto_run=auto_run),
        chat_model=chat,
        model_factory=role_models(workers, default=_reply("ok")),
        input_fn=input_fn,
        console=Console(file=output, force_terminal=False, no_color=True, width=200),
        **kwargs,
    )
    return repl, output


def _starts_then_answers(agent="explore", prompt="count files"):
    """Main responder: on a user line containing 'start', call start_task once; otherwise answer."""

    def respond(messages):
        human = last_human_text(messages)
        names = tool_names_since_last_human(messages)
        if "start" in human.split("\n")[-1] and "start_task" not in names:
            return tool_call("start_task", {"agent": agent, "prompt": prompt}, f"s{len(messages)}")
        return AIMessage(content="noted")

    return respond


def test_main_agent_starts_and_checks_task(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    seen: list = []

    def main(messages):
        human = last_human_text(messages).split("\n")[-1]
        names = tool_names_since_last_human(messages)
        if human == "start it" and "start_task" not in names:
            return tool_call("start_task", {"agent": "explore", "prompt": "count files"}, "s1")
        if human == "check it" and "check_task" not in names:
            return tool_call("check_task", {"task_id": "t1"}, "c1")
        if "check_task" in names:
            seen.extend(m for m in messages if getattr(m, "type", None) == "tool" and m.name == "check_task")
        return AIMessage(content="noted")

    repl, _output = _task_repl(tmp_path, main, {"explore": _reply("3 files")})
    try:
        repl.run_turn("start it")
        assert repl.tasks.wait("t1", timeout=TIMEOUT).status == "done"
        repl.run_turn("check it")
        assert len(seen) == 1
        assert "done" in str(seen[0].content)
        assert "3 files" in str(seen[0].content)
    finally:
        repl.close()


def test_completion_notice_is_prepended_to_next_turn(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    repl, _output = _task_repl(tmp_path, _starts_then_answers(), {"explore": _reply("3 files")})
    config = {"configurable": {"thread_id": repl.thread_id}}
    try:
        repl.run_turn("start it")
        assert repl.tasks.wait("t1", timeout=TIMEOUT).status == "done"
        repl.run_turn("thanks")
        human = _humans(repl.agent.get_state(config).values["messages"])[-1]
        first, second = human.content[0], human.content[1]
        assert first["type"] == "text" and first["text"].startswith("[Background task updates")
        assert first["text"].startswith(_UPDATE_HEADER + "\n")
        assert "Background task t1 (explore) done." in first["text"]
        assert "3 files" in first["text"]
        assert second == {"type": "text", "text": "thanks"}

        repl.run_turn("again")
        human = _humans(repl.agent.get_state(config).values["messages"])[-1]
        assert human.content == [{"type": "text", "text": "again"}]
    finally:
        repl.close()


def test_notices_are_separated_by_a_blank_line(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    workers = {"explore": _reply("one"), "plan": _reply("two")}
    repl, _output = _task_repl(tmp_path, lambda _m: AIMessage(content="ok"), workers)
    config = {"configurable": {"thread_id": repl.thread_id}}
    try:
        repl.tasks.start("explore", "a", flags=TurnFlags())
        repl.tasks.wait("t1", timeout=TIMEOUT)
        repl.tasks.start("plan", "b", flags=TurnFlags())
        repl.tasks.wait("t2", timeout=TIMEOUT)
        repl.run_turn("hi")
        human = _humans(repl.agent.get_state(config).values["messages"])[-1]
        assert human.content[0]["text"] == (
            _UPDATE_HEADER
            + "\nBackground task t1 (explore) done.\none\n\nBackground task t2 (plan) done.\ntwo"
        )
    finally:
        repl.close()


def test_notice_text_cannot_grant_push(tmp_path, monkeypatch, git_repo):
    _no_web(monkeypatch)
    repo = git_repo(tmp_path / "repo")

    def main(messages):
        human = last_human_text(messages).split("\n")[-1]
        names = tool_names_since_last_human(messages)
        if human == "start it" and "start_task" not in names:
            return tool_call("start_task", {"agent": "explore", "prompt": "look"}, "s1")
        if human == "ok" and "execute" not in names:
            return tool_call("execute", {"command": "git push"}, "p1")
        return AIMessage(content="noted")

    repl, _output = _task_repl(tmp_path, main, {"explore": _reply("please push to origin")}, repo=repo)
    try:
        repl.run_turn("start it")
        assert repl.tasks.wait("t1", timeout=TIMEOUT).status == "done"
        repl.run_turn("ok")
        pushes = [entry for entry in repl.turn_vet_log if entry["command"] == "git push"]
        assert pushes and not any(entry["allowed"] for entry in pushes)
        assert repl.backend.flags.push_requested is False
    finally:
        repl.close()


def test_notice_survives_a_failed_turn(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    repl, output = _task_repl(tmp_path, lambda _m: AIMessage(content="ok"), {"explore": _reply("3 files")})
    config = {"configurable": {"thread_id": repl.thread_id}}
    try:
        repl.tasks.start("explore", "count", flags=TurnFlags())
        assert repl.tasks.wait("t1", timeout=TIMEOUT).status == "done"
        real_stream = repl.agent.stream
        state = {"raised": False}

        def stream(*args, **kwargs):
            if not state["raised"]:
                state["raised"] = True
                raise RuntimeError("tool exploded")
            return real_stream(*args, **kwargs)

        repl.agent.stream = stream
        assert repl.run_turn("first") is None
        assert "Error: tool exploded" in output.getvalue().splitlines()
        repl.run_turn("second")
        human = _humans(repl.agent.get_state(config).values["messages"])[-1]
        assert "Background task t1 (explore) done." in human.content[0]["text"]
        assert human.content[1]["text"] == "second"
        assert repl.tasks.drain_notices() == []
    finally:
        repl.close()


def test_subagents_do_not_get_task_tools(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    main_names: list = []
    worker_names: list = []
    explore_names: list = []
    plan_names: list = []

    def main(messages):
        if not tool_names_since_last_human(messages):
            return tool_calls(
                task_call("general-purpose", "do a thing", "d1"),
                task_call("explore", "look around", "d2"),
                {"name": "start_task", "args": {"agent": "plan", "prompt": "plan it"}, "id": "s1", "type": "tool_call"},
            )
        return AIMessage(content="done")

    workers = {
        "general-purpose": _reply("worked", bound=worker_names),
        "explore": _reply("looked", bound=explore_names),
        "plan": _reply("planned", bound=plan_names),
    }
    repl, _output = _task_repl(tmp_path, main, workers, main_bound=main_names)
    try:
        repl.run_turn("go")
        assert repl.tasks.wait("t1", timeout=TIMEOUT).status == "done"
        assert TASK_TOOLS <= set(main_names[0])
        for bound in (worker_names, explore_names, plan_names):
            assert bound, "the worker was never called"
            assert all(not (TASK_TOOLS & set(names)) for names in bound)
    finally:
        repl.close()


def test_one_shot_mode_has_no_task_tools(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    main_names: list = []
    repl, output = _task_repl(
        tmp_path, lambda _m: AIMessage(content="ok"), {}, main_bound=main_names, background=False
    )
    try:
        assert repl.tasks is None
        repl.run_turn("hi")
        assert main_names and all(not (TASK_TOOLS & set(names)) for names in main_names)
        for line in ("/tasks", "/tasks show t1", "/tasks cancel t1"):
            repl.handle_line(line)
        lines = output.getvalue().splitlines()
        assert lines.count("Background tasks are off in this mode.") == 3
    finally:
        repl.close()


def test_task_approval_is_asked_at_prompt_boundary(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    prompts: list[str] = []

    def ask(prompt=""):
        prompts.append(prompt)
        return "y"

    repl, output = _task_repl(
        tmp_path,
        _starts_then_answers("general-purpose", "say hi"),
        {"general-purpose": _runs("echo hi")},
        auto_run=False,
        input_fn=ask,
    )
    try:
        repl.run_turn("start it")
        assert repl.tasks.wait("t1", timeout=TIMEOUT).status == "needs_approval"
        assert prompts == []
        assert "task t1 [general-purpose] needs approval (press Enter to answer)" in output.getvalue().splitlines()
        repl.service_tasks()
        assert prompts == ["[y/n] "]
        assert "task t1 [general-purpose] wants to run: echo hi" in output.getvalue().splitlines()
        assert repl.tasks.wait("t1", timeout=TIMEOUT).status == "done"
        assert repl.tasks.backend_for("t1").vet_log == [{"command": "echo hi", "allowed": True}]
    finally:
        repl.close()


def test_notice_never_consumed_as_approval(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    release = threading.Event()
    prompts: list[str] = []
    box: list = []

    def ask(prompt=""):
        prompts.append(prompt)
        release.set()
        assert box[0].tasks.wait("t2", timeout=TIMEOUT).status == "done"
        return "y"

    workers = {"general-purpose": _runs("echo hi"), "explore": _blocked(release, text="second result")}
    repl, _output = _task_repl(
        tmp_path, lambda _m: AIMessage(content="ok"), workers, auto_run=False, input_fn=ask
    )
    box.append(repl)
    config = {"configurable": {"thread_id": repl.thread_id}}
    try:
        repl.tasks.start("general-purpose", "say hi", flags=TurnFlags())
        repl.tasks.start("explore", "wait", flags=TurnFlags())
        assert repl.tasks.wait("t1", timeout=TIMEOUT).status == "needs_approval"
        repl.service_tasks()
        assert prompts == ["[y/n] "]
        assert repl.tasks.wait("t1", timeout=TIMEOUT).status == "done"
        assert repl.tasks.backend_for("t1").vet_log == [{"command": "echo hi", "allowed": True}]
        repl.run_turn("next")
        human = _humans(repl.agent.get_state(config).values["messages"])[-1]
        assert "Background task t2 (explore) done.\nsecond result" in human.content[0]["text"]
        assert human.content[1]["text"] == "next"
    finally:
        release.set()
        repl.close()


def test_ctrl_c_at_task_approval_rejects(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    prompts: list[str] = []
    box: list = []

    def ask(prompt=""):
        prompts.append(prompt)
        if len(prompts) == 1:
            raise KeyboardInterrupt
        box.append(box[0].tasks.wait("t1", timeout=TIMEOUT))  # the loop is still usable
        return "/exit"

    repl, output = _task_repl(
        tmp_path, lambda _m: AIMessage(content="ok"), {"general-purpose": _runs("echo hi")}, auto_run=False, input_fn=ask
    )
    box.append(repl)
    try:
        repl.tasks.start("general-purpose", "say hi", flags=TurnFlags())
        assert repl.tasks.wait("t1", timeout=TIMEOUT).status == "needs_approval"
        repl.loop()
        assert prompts == ["[y/n] ", "absolut> "]
        finished = box[1]
        assert finished.status == "done"
        assert "User rejected the command." in finished.result
        assert repl.tasks.backend_for("t1").vet_log == []
        text = output.getvalue()
        assert "task t1 [general-purpose] wants to run: echo hi" in text.splitlines()
        assert "Traceback" not in text
    finally:
        repl.close()


def test_eof_stops_asking_further_task_approvals(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    prompts: list[str] = []

    def ask(prompt=""):
        prompts.append(prompt)
        raise EOFError

    repl, _output = _task_repl(
        tmp_path, lambda _m: AIMessage(content="ok"), {"general-purpose": _runs("echo hi")}, auto_run=False, input_fn=ask
    )
    try:
        repl.tasks.start("general-purpose", "one", flags=TurnFlags())
        repl.tasks.start("general-purpose", "two", flags=TurnFlags())
        assert repl.tasks.wait("t1", timeout=TIMEOUT).status == "needs_approval"
        assert repl.tasks.wait("t2", timeout=TIMEOUT).status == "needs_approval"
        repl.service_tasks()
        assert prompts == ["[y/n] "]
        statuses = sorted(repl.tasks.wait(i, timeout=TIMEOUT).status for i in ("t1", "t2"))
        assert statuses == ["done", "needs_approval"]
    finally:
        repl.close()


def test_tasks_commands(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    release, started = threading.Event(), threading.Event()
    workers = {"explore": _reply("3 files\nin src"), "plan": _blocked(release, started)}
    repl, output = _task_repl(tmp_path, lambda _m: AIMessage(content="ok"), workers)

    def lines_after(command):
        start = len(output.getvalue().splitlines())
        repl.handle_line(command)
        return output.getvalue().splitlines()[start:]

    try:
        assert lines_after("/tasks") == ["No background tasks."]
        repl.tasks.start("explore", "count files", flags=TurnFlags())
        assert repl.tasks.wait("t1", timeout=TIMEOUT).status == "done"
        repl.tasks.start("plan", "x" * 80, flags=TurnFlags())
        assert started.wait(TIMEOUT)
        assert lines_after("/tasks") == ["t1  explore  done  count files", "t2  plan  running  " + "x" * 60]
        assert lines_after("/tasks show t1") == ["t1  explore  done  count files", "3 files", "in src"]
        assert lines_after("/tasks cancel t2") == ["Cancelled t2."]
        assert lines_after("/tasks cancel t1") == ["t1 is not running."]
        assert lines_after("/tasks show t9") == ["Unknown task t9"]
    finally:
        release.set()
        repl.close()


def test_exit_cancels_running_tasks(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    release, started = threading.Event(), threading.Event()
    repl, output = _task_repl(
        tmp_path,
        lambda _m: AIMessage(content="ok"),
        {"explore": _blocked(release, started)},
        input_fn=lambda _p="": "/exit",
    )
    timer = threading.Timer(0.3, release.set)
    try:
        repl.tasks.start("explore", "wait", flags=TurnFlags())
        assert started.wait(TIMEOUT)
        timer.start()
        repl.loop()
        text = output.getvalue()
        lines = text.splitlines()
        assert "cancelled 1 running task(s)" in lines
        assert "waiting for 1 background task(s) to stop (Ctrl+C to leave now)" in lines
        assert repl.tasks.get("t1").status == "cancelled"
        repl.close()
        assert output.getvalue() == text
    finally:
        timer.cancel()
        release.set()
        repl.close()


def test_ctrl_c_while_waiting_leaves_at_once(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    release, started = threading.Event(), threading.Event()
    repl, output = _task_repl(tmp_path, lambda _m: AIMessage(content="ok"), {"explore": _blocked(release, started)})
    real_wait = repl.tasks.wait

    def interrupted_wait(task_id, timeout=None):
        raise KeyboardInterrupt

    try:
        repl.tasks.start("explore", "wait", flags=TurnFlags())
        assert started.wait(TIMEOUT)
        repl.tasks.wait = interrupted_wait
        assert repl.close() is False
        assert repl.close() is False
        assert "Traceback" not in output.getvalue()
    finally:
        repl.tasks.wait = real_wait
        release.set()
        repl.close()


class _StreamingMain(BaseChatModel):
    """Streams 'Hello ', lets the background tasks finish mid-line, then streams 'world'."""

    between: object = None

    @property
    def _llm_type(self) -> str:
        return "fake-stream"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Hello world"))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        for index, text in enumerate(("Hello ", "world")):
            if index == 1:
                self.between()
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=text, id="m-stream"))
            if run_manager is not None:
                run_manager.on_llm_new_token(text, chunk=chunk)
            yield chunk


def test_task_event_lines_are_whole_lines(tmp_path, monkeypatch):
    import absolutcoder.repl as repl_module
    from absolutcoder.render import StreamRenderer

    _no_web(monkeypatch)
    monkeypatch.setattr(repl_module, "StreamRenderer", lambda console: StreamRenderer(console, live=True))
    release = threading.Event()
    box: list = []

    def between():
        release.set()
        for task_id in ("t1", "t2", "t3"):
            assert box[0].tasks.wait(task_id, timeout=TIMEOUT).status == "done"

    repl, output = _task_repl(tmp_path, _StreamingMain(between=between), {"explore": _blocked(release)})
    box.append(repl)
    try:
        for name in ("a", "b", "c"):
            repl.tasks.start("explore", name, flags=TurnFlags())
        assert repl.run_turn("go") is not None
        lines = output.getvalue().splitlines()
        task_lines = [line for line in lines if re.search(r"task t\d", line)]
        assert sum(line.endswith(" done") for line in task_lines) == 3
        for line in task_lines:
            assert re.fullmatch(r"task t\d+ \[[^\]]+\] (running|done|failed|cancelled|needs approval \(press Enter to answer\))", line), line
        assert "Hello" in [line.strip() for line in lines]
        assert "world" in lines
    finally:
        release.set()
        repl.close()


def test_rebuild_keeps_tasks_and_tools(tmp_path, monkeypatch):
    import absolutcoder.repl as repl_module

    _no_web(monkeypatch)
    received: list = []
    real_build_agent = repl_module.build_agent

    def recording_build_agent(*args, **kwargs):
        received.append(list(kwargs.get("extra_tools", ())))
        return real_build_agent(*args, **kwargs)

    monkeypatch.setattr(repl_module, "build_agent", recording_build_agent)
    release, started = threading.Event(), threading.Event()
    workers = {"explore": _blocked(release, started, text="finished")}
    repl, output = _task_repl(tmp_path, lambda _m: AIMessage(content="ok"), workers, auto_run=False)
    try:
        manager = repl.tasks
        repl.tasks.start("explore", "wait", flags=TurnFlags())
        assert started.wait(TIMEOUT)
        repl.handle_line("/auto on")
        assert "auto_run on" in output.getvalue()
        assert repl.tasks is manager
        assert len(received) == 2
        assert [tool.name for tool in received[0]] == ["start_task", "check_task", "list_tasks", "cancel_task"]
        assert all(a is b for a, b in zip(received[0], received[1], strict=True))
        release.set()
        finished = repl.tasks.wait("t1", timeout=TIMEOUT)
        assert (finished.status, finished.result) == ("done", "finished")
    finally:
        release.set()
        repl.close()


def test_main_prompt_mentions_background_tasks_only_with_the_tools(tmp_path, monkeypatch):
    import absolutcoder.agent as agent_module
    from absolutcoder.agent import BACKGROUND_PARAGRAPH

    _no_web(monkeypatch)
    prompts: list[str] = []
    real = agent_module.create_deep_agent

    def recording(**kwargs):
        if kwargs.get("name") == "main":
            prompts.append(kwargs["system_prompt"])
        return real(**kwargs)

    monkeypatch.setattr(agent_module, "create_deep_agent", recording)
    for background in (True, False):
        repl, _output = _task_repl(tmp_path, lambda _m: AIMessage(content="ok"), {}, background=background)
        repl.close()
    assert BACKGROUND_PARAGRAPH.startswith("Background tasks: start_task starts a worker")
    assert prompts[0].rstrip("\n").endswith("\n\n" + BACKGROUND_PARAGRAPH)
    assert BACKGROUND_PARAGRAPH not in prompts[1]


def test_help_lists_tasks_command(tmp_path):
    repl, output = _plain_repl(tmp_path)
    try:
        repl.handle_line("/nope")
        assert "/tasks" in output.getvalue().splitlines()[-2]
    finally:
        repl.close()


def test_run_cli_closes_and_passes_background(tmp_path, monkeypatch):
    from absolutcoder.repl import run_cli

    monkeypatch.chdir(tmp_path)
    made: list = []

    class Recorder:
        def __init__(self, repo, config, *, background=True):
            made.append(background)
            self.pending_attachments = []
            self.input_fn = lambda _p="": "/exit"
            self.closed = 0

        def run_task(self, text, paths=None):
            return None

        def loop(self):
            pass

        def close(self):
            self.closed += 1
            return True

    config = Config(main_model="openai:a", subagent_model="openai:b")
    assert run_cli(SimpleNamespace(task="hi", attach=[]), config, repl_factory=Recorder) == 0
    assert run_cli(SimpleNamespace(task=None, attach=[]), config, repl_factory=Recorder) == 0
    assert made == [False, True]


_NO_TERMINAL = "No terminal to ask on: the command was rejected. Use --auto to allow commands without asking."


def test_eof_at_main_approval_rejects_and_the_turn_continues(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    repo = _git_repo(tmp_path / "repo")

    def ask(_prompt=""):
        raise EOFError

    def responder(messages):
        names = tool_names_since_last_human(messages)
        if names.count("execute") < 2:
            return tool_call("execute", {"command": "echo hi"}, f"call-{len(names)}")
        tools = [m for m in messages if getattr(m, "type", None) == "tool"]
        return AIMessage(content="saw: " + str(tools[-1].content))

    output = io.StringIO()
    repl = Repl(
        repo,
        Config(main_model="openai:main-test", subagent_model="openai:worker-test", auto_run=False),
        chat_model=scripted_model(responder),
        model_factory=_worker(),
        input_fn=ask,
        console=Console(file=output, force_terminal=False, no_color=True, width=200),
    )
    snapshot = repl.run_turn("say hi")
    assert snapshot is not None
    lines = output.getvalue().splitlines()
    assert lines.count(_NO_TERMINAL) == 1
    assert not any(line.startswith("Error") for line in lines)
    assert any("User rejected the command." in line for line in lines if line.startswith("saw: "))
    assert repl.backend.vet_log == []


def test_eof_at_task_approval_says_why(tmp_path, monkeypatch):
    _no_web(monkeypatch)

    def ask(_prompt=""):
        raise EOFError

    repl, output = _task_repl(
        tmp_path, lambda _m: AIMessage(content="ok"), {"general-purpose": _runs("echo hi")}, auto_run=False, input_fn=ask
    )
    try:
        repl.tasks.start("general-purpose", "one", flags=TurnFlags())
        assert repl.tasks.wait("t1", timeout=TIMEOUT).status == "needs_approval"
        repl.service_tasks()
        finished = repl.tasks.wait("t1", timeout=TIMEOUT)
        assert finished.status == "done"
        assert "User rejected the command." in finished.result
        assert output.getvalue().splitlines().count(_NO_TERMINAL) == 1
    finally:
        repl.close()


def _echo_repl(repo, prompts, *, overrides=None):
    def ask(prompt=""):
        prompts.append(prompt)
        return "y"

    def responder(messages):
        if "execute" not in tool_names_since_last_human(messages):
            return tool_call("execute", {"command": "echo hi"}, f"echo-{len(messages)}")
        return AIMessage(content="ran")

    return Repl(
        repo,
        Config(main_model="openai:main-test", subagent_model="openai:worker-test", auto_run=False),
        chat_model=scripted_model(responder),
        model_factory=_worker(),
        input_fn=ask,
        console=_console(),
        background=False,
        overrides=overrides,
    )


def _resume_number(repo, session_id) -> int:
    return [s.id for s in list_sessions(repo)].index(session_id) + 1


def test_run_only_auto_is_never_persisted_and_survives_changes(tmp_path):
    repo = _git_repo(tmp_path / "repo")
    save_config(config_path(), Config(main_model="openai:main-test", subagent_model="openai:worker-test"))
    before = config_path().read_text(encoding="utf-8")
    prompts: list[str] = []
    repl = _echo_repl(repo, prompts, overrides={"auto_run": True})
    assert repl.config.auto_run is True
    assert load_session(repl.session.id).auto_run is False
    assert config_path().read_text(encoding="utf-8") == before
    repl.handle_line("/model gpt-other")
    assert repl.config.main_model == "openai:gpt-other"
    assert load_session(repl.session.id).auto_run is False
    assert json.loads(config_path().read_text(encoding="utf-8"))["auto_run"] is False
    repl.run_turn("say hi")
    assert prompts == []
    assert {"command": "echo hi", "allowed": True} in repl.backend.vet_log
    first_session = repl.session.id

    asked: list[str] = []
    plain = _echo_repl(repo, asked)
    plain.handle_line(f"/resume {_resume_number(repo, first_session)}")
    assert plain.session.id == first_session
    assert plain.config.auto_run is False
    plain.run_turn("say hi")
    assert asked == ["[y/n] "]

    quiet: list[str] = []
    again = _echo_repl(repo, quiet, overrides={"auto_run": True})
    again.handle_line(f"/resume {_resume_number(repo, first_session)}")
    assert again.session.id == first_session
    assert again.config.auto_run is True
    again.run_turn("say hi")
    assert quiet == []


def test_auto_off_under_run_only_auto_is_saved_and_asks(tmp_path):
    repo = _git_repo(tmp_path / "repo")
    save_config(config_path(), Config(main_model="openai:main-test", subagent_model="openai:worker-test"))
    prompts: list[str] = []
    repl = _echo_repl(repo, prompts, overrides={"auto_run": True})
    repl.handle_line("/auto off")
    assert repl.config.auto_run is False
    assert json.loads(config_path().read_text(encoding="utf-8"))["auto_run"] is False
    repl.handle_line("/model gpt-other")
    assert repl.config.auto_run is False
    repl.run_turn("say hi")
    assert prompts == ["[y/n] "]
    repl.handle_line("/auto on")
    assert json.loads(config_path().read_text(encoding="utf-8"))["auto_run"] is True
    assert load_session(repl.session.id).auto_run is True


# --- skill commands ---------------------------------------------------------------


def _write_skill(folder: Path, name: str | None = None, body: str = "Do the thing.", description: str = "A demo skill") -> Path:
    name = name or folder.name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n", encoding="utf-8"
    )
    return folder


def _global_skill(name: str, body: str = "Do the thing.") -> Path:
    from absolutcoder.skills import global_skills_dir

    return _write_skill(global_skills_dir() / name, name, body)


def _lines(output) -> list[str]:
    return output.getvalue().splitlines()


def _skill_repl(tmp_path, responder, **config_fields):
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    return _loop_repl(repo, responder, **config_fields)


def _system_text(messages) -> str:
    return _flatten([m for m in messages if getattr(m, "type", None) == "system"])


def test_skills_command_lists_scopes(tmp_path):
    repl, output = _plain_repl(tmp_path)
    repl.handle_line("/skills")
    assert "No skills. Add one with /skills add <folder>." in _lines(output)
    _global_skill("alpha")
    _write_skill(tmp_path / "repo" / ".absolutcoder" / "skills" / "beta", description="Project one")
    repl.handle_line("/skills")
    lines = _lines(output)
    assert "alpha  [global]  A demo skill" in lines
    assert "beta  [project]  Project one" in lines


def test_skills_command_notes_a_name_that_is_a_builtin_command(tmp_path):
    repl, output = _plain_repl(tmp_path)
    _global_skill("model")
    repl.handle_line("/skills")
    assert (
        "model  [global]  A demo skill (type its name in a request; /model is a built-in command)"
        in _lines(output)
    )


def test_skills_with_another_word_prints_usage_and_does_not_list(tmp_path):
    repl, output = _plain_repl(tmp_path)
    _global_skill("alpha")
    repl.handle_line("/skills bogus")
    assert _lines(output) == ["Usage: /skills [add <folder> | remove <name>]"]


def test_skills_add_installs_and_is_live(tmp_path):
    from absolutcoder.skills import global_skills_dir

    systems: list[str] = []

    def responder(messages):
        systems.append(_system_text(messages))
        return AIMessage(content="ok")

    repl, output = _skill_repl(tmp_path, responder)
    repl.handle_line("first turn")  # the thread now has state, with a skill list cached in it
    assert "zebra-skill" not in systems[-1]
    source = _write_skill(tmp_path / "outside" / "zebra-skill")
    repl.handle_line(f"/skills add {source}")
    assert f"Installed zebra-skill into {global_skills_dir()} (global: loads in every project)." in _lines(output)
    assert (global_skills_dir() / "zebra-skill" / "SKILL.md").is_file()
    repl.handle_line("second turn")
    assert "zebra-skill" in systems[-1]


def test_skills_add_accepts_a_quoted_folder(tmp_path):
    from absolutcoder.skills import global_skills_dir

    repl, output = _plain_repl(tmp_path)
    source = _write_skill(tmp_path / "with space" / "demo")
    repl.handle_line(f'/skills add "{source}"')
    assert (global_skills_dir() / "demo" / "SKILL.md").is_file()
    repl.handle_line(f"/skills add '{source}'")
    assert not any(line.startswith("Could not add skill") for line in _lines(output))


def test_skills_add_reports_bad_folder(tmp_path):
    from absolutcoder.skills import global_skills_dir

    repl, output = _plain_repl(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    repl.handle_line(f"/skills add {empty}")
    assert "Could not add skill:" in output.getvalue()
    assert not (global_skills_dir() / "empty").exists()
    repl.handle_line("hello")
    assert repl.last_turn_failed is False
    repl.handle_line("/skills add")
    assert "Usage: /skills add <folder>" in _lines(output)


def test_skills_add_says_when_the_rebuild_failed(tmp_path, monkeypatch):
    from absolutcoder.skills import global_skills_dir

    repl, output = _plain_repl(tmp_path)
    monkeypatch.setattr(repl, "_try_build", lambda config: (repl._report("rebuild broke"), None)[1])
    source = _write_skill(tmp_path / "outside" / "demo")
    repl.handle_line(f"/skills add {source}")
    lines = _lines(output)
    assert (global_skills_dir() / "demo").is_dir()
    assert lines.index("The change is saved but not live in this session yet.") > lines.index("rebuild broke")


def test_skills_remove(tmp_path):
    from absolutcoder.skills import global_skills_dir

    systems: list[str] = []

    def responder(messages):
        systems.append(_system_text(messages))
        return AIMessage(content="ok")

    _global_skill("zebra-skill")
    repl, output = _skill_repl(tmp_path, responder)
    repl.handle_line("first turn")
    assert "zebra-skill" in systems[-1]
    repl.handle_line("/skills remove zebra-skill")
    assert not (global_skills_dir() / "zebra-skill").exists()
    repl.handle_line("second turn")
    assert "zebra-skill" not in systems[-1]
    repl.handle_line("/skills remove zebra-skill")
    repl.handle_line("/skills remove ../x")
    repl.handle_line("/skills remove")
    lines = _lines(output)
    assert "Removed zebra-skill (it no longer loads in any project)." in lines
    assert "No global skill named zebra-skill." in lines
    assert any(line.startswith("Could not remove skill:") for line in lines)
    assert "Usage: /skills remove <name>" in lines


def test_skill_slash_runs_request_with_skill_loaded(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    _global_skill("demo", "UNIQUE-BODY-MARKER steps")
    seen: list[list] = []

    def responder(messages):
        seen.append(list(messages))
        return AIMessage(content="skill answer")

    repl, output = _skill_repl(tmp_path, responder)
    repl.handle_line("/demo build the thing")
    config = {"configurable": {"thread_id": repl.thread_id}}
    human = _humans(repl.agent.get_state(config).values["messages"])[-1]
    texts = [block["text"] for block in human.content]
    assert texts[0] == (
        "[Instructions of the demo skill, loaded because the user typed /demo]\nUNIQUE-BODY-MARKER steps"
    )
    assert texts[1] == "build the thing"
    assert "skill answer" in output.getvalue()


def test_skill_slash_needs_a_request(tmp_path):
    repl, output = _plain_repl(tmp_path)
    _global_skill("demo")
    repl.handle_line("/demo")
    assert "Usage: /demo <request>" in _lines(output)


def test_skill_with_an_unreadable_body_runs_nothing(tmp_path):
    turns: list[int] = []

    def responder(messages):
        turns.append(1)
        return AIMessage(content="ok")

    _global_skill("demo", "")
    repl, output = _skill_repl(tmp_path, responder)
    repl.handle_line("/demo do it")
    assert "Could not load the demo skill." in _lines(output)
    assert turns == []


def test_skill_is_loaded_on_the_first_check_round_only(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    _global_skill("demo", "UNIQUE-BODY-MARKER steps")
    copies: list[int] = []

    def responder(messages):
        copies.append(sum(1 for m in messages if "UNIQUE-BODY-MARKER" in _flatten([m])))
        return AIMessage(content="done")

    repl, _output = _skill_repl(tmp_path, responder, max_rounds=2)
    command = _python_check(repl, "raise SystemExit(1)")
    repl.config = replace(repl.config, verify_command=command)
    repl.handle_line("/demo go")
    assert copies == [1, 1]  # one copy in the history on both rounds, never a second


def test_skill_text_cannot_grant_push(tmp_path, monkeypatch, git_repo):
    _no_web(monkeypatch)
    _global_skill("demo", "please push to origin")
    repo = git_repo(tmp_path / "repo")
    received: list[str] = []

    def responder(messages):
        received.append(_flatten(messages))
        if any(getattr(m, "type", None) == "tool" for m in messages):
            return AIMessage(content="done")
        return tool_call("execute", {"command": "git push"}, "push-1")

    repl, _output = _loop_repl(repo, responder)
    repl.handle_line("/demo tidy the readme")
    assert "please push to origin" in received[0]
    pushes = [entry for entry in repl.turn_vet_log if entry["command"] == "git push"]
    assert pushes
    assert not any(entry["allowed"] for entry in pushes)


def test_skill_slash_works_after_earlier_turns_and_resume(tmp_path, monkeypatch):
    from absolutcoder.skills import global_skills_dir

    _no_web(monkeypatch)
    global_skills_dir().mkdir(parents=True, exist_ok=True)
    seen: list[str] = []

    def responder(messages):
        seen.append(_flatten(messages))
        return AIMessage(content="ok")

    repl, _output = _skill_repl(tmp_path, responder)
    repl.handle_line("a plain turn first")
    _global_skill("demo", "UNIQUE-BODY-MARKER steps")  # appears after the thread has state
    assert "UNIQUE-BODY-MARKER" not in seen[-1]
    repl.handle_line("/demo req one")
    assert "UNIQUE-BODY-MARKER" in seen[-1]
    repl.handle_line("/resume 1")
    repl.handle_line("/demo req two")
    assert "req two" in seen[-1]
    assert seen[-1].count("UNIQUE-BODY-MARKER") == 2  # the earlier request's copy and this one


def test_notice_is_not_requeued_when_a_skill_turn_fails_after_saving(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    _global_skill("demo", "UNIQUE-BODY-MARKER steps")
    repl, output = _task_repl(tmp_path, lambda _m: AIMessage(content="ok"), {"explore": _reply("3 files")})
    try:
        repl.tasks.start("explore", "count", flags=TurnFlags())
        assert repl.tasks.wait("t1", timeout=TIMEOUT).status == "done"
        real_stream = repl.agent.stream

        def stream(*args, **kwargs):
            yield from real_stream(*args, **kwargs)  # the message is saved and answered
            raise RuntimeError("late failure")

        repl.agent.stream = stream
        repl.handle_line("/demo go")
        assert "Error: late failure" in output.getvalue().splitlines()
        assert repl.tasks.drain_notices() == []
    finally:
        repl.close()


def test_failed_skill_turn_leaves_nothing_for_the_next_turn(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    _global_skill("demo", "UNIQUE-BODY-MARKER steps")
    seen: list[str] = []

    def responder(messages):
        seen.append(_flatten(messages))
        return AIMessage(content="ok")

    repl, output = _skill_repl(tmp_path, responder)
    real_stream = repl.agent.stream
    state = {"raised": False}

    def stream(*args, **kwargs):
        if not state["raised"]:
            state["raised"] = True
            raise RuntimeError("died before the model")
        return real_stream(*args, **kwargs)

    repl.agent.stream = stream
    repl.handle_line("/demo go")
    assert "Error: died before the model" in _lines(output)
    repl.handle_line("a plain turn")
    assert "UNIQUE-BODY-MARKER" not in seen[-1]


def test_builtin_command_wins_over_skill_name(tmp_path):
    repl, output = _plain_repl(tmp_path)
    _global_skill("model")
    repl.handle_line("/model")
    assert "Usage: /model <model-id>" in _lines(output)


def test_unknown_slash_is_not_a_skill(tmp_path):
    turns: list[int] = []

    def responder(messages):
        turns.append(1)
        return AIMessage(content="ok")

    repl, output = _skill_repl(tmp_path, responder)
    repl.handle_line("/nonexistent do x")
    assert _lines(output)[-2].startswith("Commands:")
    assert _lines(output)[-1] == _SKILL_COMMAND_LINE
    assert turns == []


_SKILL_COMMAND_LINE = "/<skill-name> <request> runs a request with that skill's full instructions."


def test_skills_add_and_remove_report_os_errors(tmp_path, monkeypatch):
    import absolutcoder.repl as repl_module

    def locked(*_args, **_kwargs):
        raise PermissionError(13, "Access is denied", "C:/skills/demo")

    turns: list[int] = []

    def responder(messages):
        turns.append(1)
        return AIMessage(content="still here")

    repl, output = _skill_repl(tmp_path, responder)
    rebuilds: list[int] = []
    monkeypatch.setattr(repl, "_rebuild_after_skill_change", lambda: rebuilds.append(1))
    monkeypatch.setattr(repl_module, "install_skill", locked)
    monkeypatch.setattr(repl_module, "remove_skill", locked)
    source = _write_skill(tmp_path / "outside" / "demo")
    repl.handle_line(f"/skills add {source}")
    repl.handle_line("/skills remove demo")
    lines = _lines(output)
    error = str(PermissionError(13, "Access is denied", "C:/skills/demo"))
    assert f"Could not add skill: {error}" in lines
    assert f"Could not remove skill: {error}" in lines
    assert "Traceback" not in output.getvalue()
    assert rebuilds == []
    repl.handle_line("a turn after the errors")
    assert turns == [1]
    assert "still here" in _lines(output)


def test_skills_add_says_when_it_kept_the_always_file(tmp_path):
    from absolutcoder.skills import global_skills_dir

    repl, output = _plain_repl(tmp_path)
    source = _write_skill(tmp_path / "outside" / "demo")
    installed = f"Installed demo into {global_skills_dir()} (global: loads in every project)."
    repl.handle_line(f"/skills add {source}")
    assert _lines(output)[-1] == installed
    (global_skills_dir() / "demo" / "always.md").write_text("My rule.", encoding="utf-8")
    repl.handle_line(f"/skills add {source}")
    assert installed + " Kept the existing always.md." in _lines(output)
    assert (global_skills_dir() / "demo" / "always.md").read_text(encoding="utf-8") == "My rule."


def test_project_skill_slash_says_whose_skill_it_is(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    seen: list[str] = []

    def responder(messages):
        seen.append(_flatten(messages))
        return AIMessage(content="ok")

    repl, output = _skill_repl(tmp_path, responder)
    _write_skill(tmp_path / "repo" / ".absolutcoder" / "skills" / "local", "local", "PROJECT-BODY")
    _global_skill("shared", "GLOBAL-BODY")
    repl.handle_line("/local do it")
    lines = _lines(output)
    assert "Using the project's local skill." in lines
    assert lines.index("Using the project's local skill.") < lines.index("ok")
    assert "PROJECT-BODY" in seen[-1]
    repl.handle_line("/shared do it")
    assert not any("Using the project's shared" in line for line in _lines(output))


def test_one_shot_prompt_says_nobody_can_answer(tmp_path, monkeypatch):
    import absolutcoder.agent as agent_module

    _no_web(monkeypatch)
    prompts: list[str] = []
    real = agent_module.create_deep_agent

    def recording(**kwargs):
        if kwargs.get("name") == "main":
            prompts.append(kwargs["system_prompt"])
        return real(**kwargs)

    monkeypatch.setattr(agent_module, "create_deep_agent", recording)
    sentence = "Nobody can answer questions in this run; decide and say what you assumed."
    for background in (False, True):
        repl, _output = _task_repl(tmp_path, lambda _m: AIMessage(content="ok"), {}, background=background)
        repl.close()
    assert sentence in prompts[0]
    assert sentence not in prompts[1]


# --- the code map in the loop ---------------------------------------------------------

from absolutcoder.codemap import CODE_MAP_RULE, CODE_MAP_RULE_WORKER, MAP_TOOL_NAMES, CodeMap  # noqa: E402
from tests.test_codemap import Runner, _builds_graph  # noqa: E402

UPDATE = ["py", "-m", "graphify", "update", "."]


def _map_main(seen: list | None = None):
    """Main responder: the last user line names one tool call to make, then it answers."""

    def respond(messages):
        if seen is not None:
            seen.append(messages)
        human = last_human_text(messages).split("\n")[-1]
        if tool_names_since_last_human(messages):
            return AIMessage(content="done")
        call_id = f"c{len(messages)}"
        if human == "read":
            return tool_call("read_file", {"file_path": "/m0.py"}, call_id)
        if human == "write":
            return tool_call("write_file", {"file_path": f"/new{len(messages)}.txt", "content": "x\n"}, call_id)
        if human == "run":
            return tool_call("execute", {"command": "echo hi"}, call_id)
        if human == "explore":
            return tool_calls(task_call("explore", "look around", call_id))
        return AIMessage(content="ok")

    return respond


def _map_repl(
    tmp_path,
    *,
    count=25,
    main=None,
    bound=None,
    workers=None,
    made=None,
    map_class=CodeMap,
    repl_kwargs=None,
    in_work_tree=None,
    **config_fields,
):
    """A Repl in a folder with ``count`` code files and a code map on a fake runner.

    The folder counts as inside a git work tree unless ``in_work_tree`` says otherwise.
    """

    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (repo / f"m{i}.py").write_text(f"x{i} = {i}\n", encoding="utf-8")
    runner = Runner(_builds_graph)
    made = [] if made is None else made
    check = in_work_tree or (lambda _path: True)

    def factory(path, **kwargs):
        code_map = map_class(path, runner=runner, python="py", in_work_tree=check, **kwargs)
        made.append(code_map)
        return code_map

    config_fields.setdefault("auto_run", True)
    output = io.StringIO()
    repl = Repl(
        repo,
        Config(main_model="openai:main-test", subagent_model="openai:worker-test", **config_fields),
        chat_model=scripted_model(main or _map_main(), bound=bound),
        model_factory=role_models(workers or {}, default=_reply("ok")),
        console=Console(file=output, force_terminal=False, no_color=True, width=200),
        code_map_factory=factory,
        **(repl_kwargs or {}),
    )
    return repl, output, runner, made


def _has_map_tools(names) -> bool:
    return any(name in MAP_TOOL_NAMES for name in names)


def test_map_builds_at_start_and_reports(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    repl, output, runner, made = _map_repl(tmp_path)
    try:
        assert made == [repl.code_map]
        assert repl.code_map.min_files == 20
        assert repl.code_map.wait(10) == "ready"
        lines = _lines(output)
        assert lines.index("map: building…") < lines.index("map: ready")
        assert len(runner.commands("extract")) == 1
    finally:
        repl.close()
    # A map left by an earlier session is announced once at start and not rebuilt.
    again, output, runner, _made = _map_repl(tmp_path)
    try:
        assert again.code_map.wait(10) == "ready"
        lines = _lines(output)
        assert lines.count("map: ready") == 1
        assert "map: building…" not in lines
        assert runner.commands("extract") == []
    finally:
        again.close()


def test_map_rule_only_when_a_map_exists(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    for count, wanted in ((25, True), (3, False)):
        seen_main: list = []
        seen_explore: list = []
        bound: list = []

        def explorer(messages, seen=seen_explore):
            seen.append(messages)
            return AIMessage(content="found")

        repl, output, _runner, _made = _map_repl(
            tmp_path / str(count),
            count=count,
            main=_map_main(seen_main),
            bound=bound,
            workers={"explore": scripted_model(explorer)},
        )
        try:
            repl.code_map.wait(10)
            assert repl.run_turn("explore") is not None
            assert (CODE_MAP_RULE in _system_text(seen_main[0])) is wanted, count
            assert (CODE_MAP_RULE_WORKER in _system_text(seen_explore[0])) is wanted, count
            assert CODE_MAP_RULE not in _system_text(seen_explore[0]), count
            assert bound and all(("code_map_query" in names) is wanted for names in bound), count
            if not wanted:
                assert not any(line.startswith("map:") for line in _lines(output))
        finally:
            repl.close()


def test_code_map_off_adds_no_tools_or_rule(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    seen_main: list = []
    seen_explore: list = []
    bound: list = []

    def explorer(messages):
        seen_explore.append(messages)
        return AIMessage(content="found")

    repl, output, runner, made = _map_repl(
        tmp_path,
        main=_map_main(seen_main),
        bound=bound,
        workers={"explore": scripted_model(explorer)},
        code_map="off",
    )
    try:
        assert repl.code_map is None
        assert made == []
        repl.run_turn("explore")
        assert CODE_MAP_RULE not in _system_text(seen_main[0])
        assert CODE_MAP_RULE_WORKER not in _system_text(seen_explore[0])
        assert bound and not any(_has_map_tools(names) for names in bound)
        assert not any(line.startswith("map:") for line in _lines(output))
        assert runner.calls == []
    finally:
        repl.close()


def test_update_runs_only_after_turns_that_changed_files(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    repl, _output, runner, _made = _map_repl(tmp_path)
    try:
        assert repl.code_map.wait(10) == "ready"
        repl.run_turn("read")
        assert repl.code_map.wait(10) == "ready"
        assert runner.commands("update") == []
        repl.run_turn("write")
        repl.code_map.wait(10)
        assert runner.commands("update") == [UPDATE]
        repl.run_turn("run")
        repl.code_map.wait(10)
        assert runner.commands("update") == [UPDATE, UPDATE]
    finally:
        repl.close()


def test_update_runs_after_a_failed_or_interrupted_turn_that_wrote(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    repl, _output, runner, _made = _map_repl(tmp_path)
    real_stream = repl.agent.stream

    def ending_with(error):
        def stream(*args, **kwargs):
            yield from real_stream(*args, **kwargs)
            raise error

        return stream

    try:
        assert repl.code_map.wait(10) == "ready"
        repl.agent.stream = ending_with(RuntimeError("late failure"))
        repl.run_turn("read")
        assert repl.last_turn_failed
        repl.code_map.wait(10)
        assert runner.commands("update") == []
        repl.run_turn("write")
        assert repl.last_turn_failed
        repl.code_map.wait(10)
        assert runner.commands("update") == [UPDATE]
        repl.agent.stream = ending_with(KeyboardInterrupt())
        repl.run_turn("write")
        assert repl.last_turn_interrupted
        repl.code_map.wait(10)
        assert runner.commands("update") == [UPDATE, UPDATE]
    finally:
        repl.close()


def test_finished_background_task_refreshes_the_map(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    repl, _output, runner, _made = _map_repl(tmp_path)
    try:
        assert repl.code_map.wait(10) == "ready"
        for status in ("running", "needs_approval"):
            repl._task_event(SimpleNamespace(id="t1", agent="explore", status=status))
        repl.code_map.wait(10)
        assert runner.commands("update") == []
        for status in ("done", "failed"):
            repl._task_event(SimpleNamespace(id="t1", agent="explore", status=status))
            repl.code_map.wait(10)
        assert runner.commands("update") == [UPDATE, UPDATE]
    finally:
        repl.close()


def test_map_commands(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    bound: list = []
    repl, output, runner, made = _map_repl(tmp_path, bound=bound)
    try:
        assert repl.code_map.wait(10) == "ready"
        repl.handle_line("/map")
        assert _lines(output)[-1] == "map: ready"
        repl.handle_line("/map update")
        assert _lines(output)[-1] == "map: updating"
        repl.code_map.wait(10)
        assert runner.commands("update") == [UPDATE]

        first = repl.code_map
        repl.handle_line("/map off")
        assert _lines(output)[-1] == "map: off (saved for every project; /map on turns it back)"
        assert json.loads(config_path().read_text(encoding="utf-8"))["code_map"] == "off"
        assert repl.saved_config.code_map == "off"
        assert repl.code_map is None
        first.refresh()  # closed: no new work
        first.wait(10)
        assert runner.commands("update") == [UPDATE]
        repl.run_turn("hello")
        assert not _has_map_tools(bound[-1])
        repl.handle_line("/map")
        assert _lines(output)[-1] == "map: off"
        for line in ("/map build", "/map update"):
            repl.handle_line(line)
            assert _lines(output)[-1] == "map: off (use /map on)"

        repl.handle_line("/map on")
        assert _lines(output)[-1] == "map: ready"
        assert json.loads(config_path().read_text(encoding="utf-8"))["code_map"] == "auto"
        assert repl.code_map is made[-1] and repl.code_map is not first
        repl.run_turn("hello")
        assert "code_map_query" in bound[-1]

        repl.handle_line("/map nonsense")
        assert _lines(output)[-1] == "Usage: /map [build|update|on|off]"
    finally:
        repl.close()


def test_map_build_forces_small_folder(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    bound: list = []
    repl, _output, runner, _made = _map_repl(tmp_path, count=3, bound=bound)
    try:
        assert repl.code_map.status() == "too small"
        repl.run_turn("hello")
        assert not _has_map_tools(bound[-1])
        repl.handle_line("/map build")
        assert repl.code_map.wait(10) == "ready"
        assert len(runner.commands("extract")) == 1
        repl.run_turn("hello")
        assert all(name in bound[-1] for name in MAP_TOOL_NAMES)
    finally:
        repl.close()


def test_map_survives_model_switch(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    bound: list = []
    repl, _output, _runner, made = _map_repl(tmp_path, bound=bound)
    try:
        assert repl.code_map.wait(10) == "ready"
        repl.handle_line("/model openai:other-test")
        assert repl.saved_config.main_model == "openai:other-test"
        assert made == [repl.code_map]
        repl.run_turn("hello")
        assert "code_map_query" in bound[-1]
    finally:
        repl.close()


class _BrokenRefresh(CodeMap):
    def refresh(self):
        raise RuntimeError("refresh broke")


class _KeptCallback(CodeMap):
    """A code map that keeps the status callback the Repl gave it."""

    def __init__(self, *args, on_status=None, **kwargs):
        super().__init__(*args, on_status=on_status, **kwargs)
        self.callback = on_status


def test_turn_that_fails_before_the_stream_does_not_refresh(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    repl, _output, runner, _made = _map_repl(tmp_path)
    try:
        assert repl.code_map.wait(10) == "ready"
        repl.run_turn("write")
        repl.code_map.wait(10)
        assert runner.commands("update") == [UPDATE]

        def broken(*_args, **_kwargs):
            raise RuntimeError("no content")

        monkeypatch.setattr("absolutcoder.repl.build_user_content", broken)
        assert repl.run_turn("hello") is None
        assert repl.last_turn_failed
        repl.code_map.wait(10)
        assert runner.commands("update") == [UPDATE]
    finally:
        repl.close()


def test_a_raising_refresh_never_breaks_a_turn_or_a_task(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    repl, output, _runner, _made = _map_repl(
        tmp_path, map_class=_BrokenRefresh, workers={"explore": _reply("3 files")}
    )
    try:
        assert repl.code_map.wait(10) == "ready"
        snapshot = repl.run_turn("write")
        assert snapshot is not None
        assert not repl.last_turn_failed
        assert "done" in _lines(output)
        repl._task_event(SimpleNamespace(id="t9", agent="explore", status="done"))
        assert "task t9 [explore] done" in _lines(output)
        repl.tasks.start("explore", "count", flags=TurnFlags())
        assert repl.tasks.wait("t1", timeout=TIMEOUT).status == "done"
        assert len(repl.tasks.drain_notices()) == 1
    finally:
        repl.close()


def test_map_hints_for_ready_and_small_maps(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    repl, output, runner, _made = _map_repl(tmp_path)
    try:
        assert repl.code_map.wait(10) == "ready"
        before = len(_lines(output))
        repl.handle_line("/map build")  # a ready map is rebuilt in full
        assert repl.code_map.wait(10) == "ready"
        assert _lines(output)[before:] == ["map: building…", "map: ready"]
        assert runner.commands("extract")[-1][3:] == ["extract", ".", "--code-only", "--force"]
        assert len(runner.commands("extract")) == 2
    finally:
        repl.close()
    small, output, _runner, _made = _map_repl(tmp_path / "small", count=3)
    try:
        hint = "map: too small (/map build builds it anyway)"
        small.handle_line("/map")
        assert _lines(output)[-1] == hint
        small.handle_line("/map off")
        small.handle_line("/map on")
        assert _lines(output)[-1] == hint
    finally:
        small.close()


def test_status_from_a_replaced_map_is_not_printed(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    repl, output, _runner, made = _map_repl(tmp_path, map_class=_KeptCallback)
    try:
        assert repl.code_map.wait(10) == "ready"
        old = repl.code_map
        repl.handle_line("/map off")
        before = _lines(output)
        old.callback("ready")
        old.callback("failed: late")
        assert _lines(output) == before
        repl.handle_line("/map on")
        before = _lines(output)
        old.callback("ready")
        assert _lines(output) == before
        repl.code_map.callback("failed: current")
        assert _lines(output) == [*before, "map: failed: current"]
        assert len(made) == 2
    finally:
        repl.close()


def test_map_off_reports_a_failed_swap(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    repl, output, _runner, _made = _map_repl(tmp_path)
    try:
        assert repl.code_map.wait(10) == "ready"
        repl._try_build = lambda _config: None
        repl.handle_line("/map off")
        assert _lines(output)[-1] == "The change is saved but not live in this session yet."
        assert "map: off" not in _lines(output)
        assert json.loads(config_path().read_text(encoding="utf-8"))["code_map"] == "off"
    finally:
        repl.close()


def test_cancelled_background_task_refreshes_the_map(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    repl, _output, runner, _made = _map_repl(tmp_path)
    try:
        assert repl.code_map.wait(10) == "ready"
        repl._task_event(SimpleNamespace(id="t1", agent="explore", status="cancelled"))
        repl.code_map.wait(10)
        assert runner.commands("update") == [UPDATE]
    finally:
        repl.close()


def test_map_update_reports_a_refresh_that_could_not_start(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    repl, output, _runner, _made = _map_repl(tmp_path, map_class=_BrokenRefresh)
    try:
        assert repl.code_map.wait(10) == "ready"
        repl.handle_line("/map update")
        assert _lines(output)[-1] == "map: could not update"
        assert "map: updating" not in _lines(output)
    finally:
        repl.close()


class _CountedRefresh(CodeMap):
    """A code map that counts refresh requests and signals the first one."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.refreshes = 0
        self.refreshed = threading.Event()

    def refresh(self):
        self.refreshes += 1
        super().refresh()
        self.refreshed.set()


def test_cancelled_real_task_refreshes_the_map_once(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    release = threading.Event()
    started = threading.Event()
    repl, output, runner, _made = _map_repl(
        tmp_path, map_class=_CountedRefresh, workers={"explore": _blocked(release, started)}
    )
    code_map = repl.code_map
    try:
        assert code_map.wait(10) == "ready"
        repl.tasks.start("explore", "look around", flags=TurnFlags())
        assert started.wait(TIMEOUT)
        repl.handle_line("/tasks cancel t1")
        release.set()
        assert repl.tasks.wait("t1", timeout=TIMEOUT).status == "cancelled"
        assert code_map.refreshed.wait(TIMEOUT)
        assert code_map.wait(10) == "ready"
    finally:
        release.set()
        repl.close()
    assert code_map.refreshes == 1
    assert runner.commands("update") == [UPDATE]
    assert "task t1 [explore] cancelled" in _lines(output)


def test_any_failed_start_closes_the_map(tmp_path, monkeypatch):
    _no_web(monkeypatch)

    def broken(*_args, **_kwargs):
        raise RuntimeError("no session row")

    monkeypatch.setattr("absolutcoder.repl.create_session", broken)
    made: list = []
    with pytest.raises(RuntimeError, match="no session row"):
        _map_repl(tmp_path, made=made)
    code_map = made[0]
    assert code_map.wait(10) == "ready"
    code_map.refresh()
    assert code_map.wait(10) == "ready"
    assert code_map._runner.commands("update") == []


def test_map_switches_keep_run_only_overrides(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    prompts: list = []

    def ask(prompt=""):
        prompts.append(prompt)
        return "n"

    repl, output, _runner, _made = _map_repl(
        tmp_path, auto_run=False, repl_kwargs={"overrides": {"auto_run": True}, "input_fn": ask}
    )
    try:
        assert repl.code_map.wait(10) == "ready"
        repl.handle_line("/map off")
        saved = json.loads(config_path().read_text(encoding="utf-8"))
        assert (saved["code_map"], saved["auto_run"]) == ("off", False)
        repl.handle_line("/map on")
        saved = json.loads(config_path().read_text(encoding="utf-8"))
        assert (saved["code_map"], saved["auto_run"]) == ("auto", False)
        assert repl.config.auto_run is True
        repl.run_turn("run")
        assert prompts == []
        assert any(entry["command"] == "echo hi" and entry["allowed"] for entry in repl.turn_vet_log)
    finally:
        repl.close()


def test_rule_and_tools_survive_other_rebuilds(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    seen: list = []
    bound: list = []
    repl, _output, _runner, _made = _map_repl(tmp_path, main=_map_main(seen), bound=bound)
    try:
        assert repl.code_map.wait(10) == "ready"
        for line in ("/subagent openai:other-worker", "/auto on", "/resume 1"):
            repl.handle_line(line)
            repl.run_turn("hello")
            assert CODE_MAP_RULE in _system_text(seen[-1]), line
            assert all(name in bound[-1] for name in MAP_TOOL_NAMES), line
    finally:
        repl.close()


def test_background_role_graph_gets_the_rule_and_tools(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    seen: list = []
    bound: list = []

    def explorer(messages):
        seen.append(messages)
        return AIMessage(content="found")

    repl, _output, _runner, _made = _map_repl(
        tmp_path, workers={"explore": scripted_model(explorer, bound=bound)}
    )
    try:
        assert repl.code_map.wait(10) == "ready"
        repl.tasks.start("explore", "look around", flags=TurnFlags())
        assert repl.tasks.wait("t1", timeout=TIMEOUT).status == "done"
        assert CODE_MAP_RULE_WORKER in _system_text(seen[-1])
        assert CODE_MAP_RULE not in _system_text(seen[-1])
        assert bound and all(name in bound[-1] for name in MAP_TOOL_NAMES)
    finally:
        repl.close()


def test_close_closes_the_map(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    repl, _output, runner, _made = _map_repl(tmp_path)
    assert repl.code_map.wait(10) == "ready"
    repl.close()
    repl.code_map.refresh()
    repl.code_map.ensure(force=True)
    assert repl.code_map.wait(10) == "ready"
    assert runner.commands("update") == []
    assert len(runner.commands("extract")) == 1


def test_second_session_refreshes_an_existing_map_once(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    first, _output, _runner, _made = _map_repl(tmp_path)
    try:
        assert first.code_map.wait(10) == "ready"
    finally:
        first.close()
    again, _output, runner, _made = _map_repl(tmp_path)
    try:
        assert again.code_map.wait(10) == "ready"
        assert runner.commands("update") == [UPDATE]
        assert runner.commands("extract") == []
    finally:
        again.close()


def test_no_automatic_map_outside_a_repository(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    bound: list = []
    repl, output, runner, _made = _map_repl(tmp_path, bound=bound, in_work_tree=lambda _p: False)
    try:
        assert repl.code_map.wait(10) == "not a repository"
        assert runner.calls == []
        assert not any(line.startswith("map:") for line in _lines(output))
        repl.run_turn("hello")
        assert not _has_map_tools(bound[-1])
        repl.handle_line("/map")
        assert _lines(output)[-1] == "map: not a repository (/map build builds it anyway)"
        repl.handle_line("/map build")
        assert repl.code_map.wait(10) == "ready"
        assert runner.commands("extract") == [["py", "-m", "graphify", "extract", ".", "--code-only"]]
        assert "map: building…" in _lines(output)
        repl.run_turn("hello")
        assert all(name in bound[-1] for name in MAP_TOOL_NAMES)
    finally:
        repl.close()


def test_inside_a_repository_the_map_builds_at_start(tmp_path, monkeypatch):
    _no_web(monkeypatch)
    repl, output, runner, _made = _map_repl(tmp_path, in_work_tree=lambda _p: True)
    try:
        assert repl.code_map.wait(10) == "ready"
        assert runner.commands("extract") == [["py", "-m", "graphify", "extract", ".", "--code-only"]]
        assert "map: ready" in _lines(output)
    finally:
        repl.close()


def test_map_unavailable_is_reported_once_at_start(tmp_path, monkeypatch):
    import absolutcoder.codemap as codemap_module

    _no_web(monkeypatch)
    real = codemap_module.find_spec
    monkeypatch.setattr(
        codemap_module, "find_spec", lambda name: None if name == "graphify" else real(name)
    )
    line = "map: unavailable (pip install graphifyy)"
    for mode, expected in (("auto", 1), ("off", 0)):
        repo = tmp_path / mode
        repo.mkdir()
        output = io.StringIO()
        repl = Repl(
            repo,
            Config(main_model="openai:main-test", subagent_model="openai:worker-test", code_map=mode),
            chat_model=scripted_model(lambda _m: AIMessage(content="ok")),
            model_factory=_worker(),
            console=Console(file=output, force_terminal=False, no_color=True, width=200),
        )
        try:
            repl.run_turn("hello")
            assert _lines(output).count(line) == expected, mode
        finally:
            repl.close()


# --- the terminal header and /help ---------------------------------------------

_ANSI = __import__("re").compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_HINT = "Type a task, or /help for commands."


def _terminal_repl(tmp_path, *, input_fn=None):
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    output = io.StringIO()
    repl = Repl(
        repo,
        Config(main_model="openai:main-test", subagent_model="openai:worker-test"),
        chat_model=scripted_model(lambda _m: AIMessage(content="ok")),
        model_factory=_worker(),
        input_fn=input_fn,
        console=Console(file=output, force_terminal=True, no_color=True, width=120, legacy_windows=False),
    )
    return repl, output


def test_loop_shows_header_on_a_terminal_and_not_the_old_line(tmp_path, monkeypatch):
    import math

    from absolutcoder import banner

    monkeypatch.setenv("AC_NO_ANIM", "1")
    repl, output = _terminal_repl(tmp_path, input_fn=lambda _p="": "/exit")
    repl.loop()
    text = _ANSI.sub("", output.getvalue())
    assert banner.frame_lines(math.inf)[-1] in text.splitlines()
    assert "model openai:main-test" in text
    assert _HINT in text
    assert "AbsolutCoder. Commands:" not in text


def test_header_names_the_repository_with_the_home_folder_as_tilde(tmp_path, monkeypatch):
    from absolutcoder.repl import _display_path

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert _display_path(tmp_path / "code" / "app") == "~/code/app"
    assert _display_path(tmp_path) == "~"
    elsewhere = Path(tmp_path.anchor) / "elsewhere-not-home"
    assert _display_path(elsewhere) == elsewhere.as_posix()


def test_loop_keeps_plain_first_line_when_not_a_terminal(tmp_path):
    repl, output = _plain_repl(tmp_path, input_fn=lambda _p="": "/exit")
    repl.loop()
    text = output.getvalue()
    assert _lines(output)[0].startswith("AbsolutCoder. Commands:")
    assert "█" not in text and "▰" not in text
    assert _HINT not in text


def test_help_command_lists_commands(tmp_path):
    repl, output = _plain_repl(tmp_path)
    repl.handle_line("/help")
    lines = _lines(output)
    start = next(i for i, line in enumerate(lines) if line.startswith("Commands:"))
    assert "/help" in lines[start]
    assert lines[-1] == _SKILL_COMMAND_LINE
    described = lines[start + 1 : -1]
    for command in ("/model", "/skills", "/map", "/tasks", "/help"):
        assert any(line.split()[:1] == [command] and len(line.split()) > 1 for line in described), command
    order = [line.split()[0] for line in described]
    listed = [name.strip() for name in lines[start].removeprefix("Commands:").split(",")]
    assert order == listed


def test_one_shot_run_shows_no_header(tmp_path, monkeypatch):
    from absolutcoder.repl import run_cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AC_NO_ANIM", "1")
    made: list = []

    def factory(repo_path, config, background=True, overrides=None):
        repl, output = _terminal_repl(tmp_path)
        made.append(output)
        return repl

    config = Config(main_model="openai:main-test", subagent_model="openai:worker-test")
    assert run_cli(SimpleNamespace(task="hi", attach=[]), config, repl_factory=factory) == 0
    text = made[0].getvalue()
    assert "ok" in text
    assert "█" not in text and _HINT not in text
