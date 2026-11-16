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
