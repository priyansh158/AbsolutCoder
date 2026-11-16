"""Each command runs as one process in a known shell, with a per-call environment."""

import os
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from absolutcoder import shell as shell_module
from absolutcoder.agent import build_agent
from absolutcoder.config import Config
from absolutcoder.repl import run_cli
from absolutcoder.shell import GuardedShellBackend, find_shell
from tests.support import role_models, scripted_model, task_call, tool_calls

POSIX = find_shell()[0] != "cmd"
posix_only = pytest.mark.skipif(not POSIX, reason="needs a POSIX shell")


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)
    return completed.stdout


def test_cd_persists_across_and_chain(tmp_path):
    (tmp_path / "sub").mkdir()
    backend = GuardedShellBackend(root_dir=tmp_path)
    show = "pwd" if backend.shell_kind != "cmd" else "cd"
    result = backend.execute(f"cd sub && {show}")
    assert result.exit_code == 0
    assert result.output.strip().replace("\\", "/").endswith("/sub")


@posix_only
def test_exported_variable_persists_in_one_command(tmp_path):
    backend = GuardedShellBackend(root_dir=tmp_path)
    result = backend.execute('export X=5; echo "$X"')
    assert result.output.strip() == "5"


@posix_only
def test_multiline_loop_runs_as_one_script(tmp_path):
    backend = GuardedShellBackend(root_dir=tmp_path)
    result = backend.execute("for i in 1 2 3\ndo\n  echo line$i\ndone")
    assert result.output.splitlines() == ["line1", "line2", "line3"]


@posix_only
def test_heredoc_is_not_split(tmp_path):
    backend = GuardedShellBackend(root_dir=tmp_path)
    result = backend.execute("cat <<'EOF'\nline one\nline two\nEOF")
    assert result.output.splitlines() == ["line one", "line two"]


def test_plain_command_is_passed_through_unchanged(tmp_path, monkeypatch):
    backend = GuardedShellBackend(root_dir=tmp_path)
    calls = []

    def record(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(shell_module.subprocess, "run", record)
    command = "echo a && echo b"
    result = backend.execute(command)
    assert result.output == "ok\n"
    assert len(calls) == 1
    args, kwargs = calls[0]
    if backend.shell_kind == "cmd":
        assert args.endswith(' "' + command + '"')
    else:
        assert args[-1] == command
    assert kwargs["env"] is not backend._env


def test_commit_env_is_per_call(tmp_path, git_repo):
    repo = git_repo(tmp_path / "repo")
    backend = GuardedShellBackend(root_dir=repo)
    backend._env["GIT_AUTHOR_NAME"] = "Not Ada"
    backend._env["GIT_COMMITTER_EMAIL"] = "not@example.com"
    before = dict(backend._env)
    (repo / "f.txt").write_text("f\n", encoding="utf-8")
    result = backend.execute('git add f.txt && git commit -m "feat: f"')
    assert result.exit_code == 0, result.output
    assert dict(backend._env) == before
    assert _git(repo, "log", "-1", "--format=%an <%ce>").strip() == "Ada Fixture <ada@example.com>"


def test_commit_inside_chain_still_scrubbed(tmp_path, git_repo):
    repo = git_repo(tmp_path / "repo")
    backend = GuardedShellBackend(root_dir=repo)
    (repo / "f.txt").write_text("f\n", encoding="utf-8")
    result = backend.execute(
        'cd . && git add f.txt && git commit -m "feat: x" -m "Co-authored-by: A <a@b.c>"'
    )
    assert result.exit_code == 0, result.output
    assert _git(repo, "rev-list", "--count", "HEAD").strip() == "1"
    assert _git(repo, "log", "-1", "--format=%B").strip() == "feat: x"


def test_multiline_commit_message_survives(tmp_path, git_repo):
    repo = git_repo(tmp_path / "repo")
    backend = GuardedShellBackend(root_dir=repo)
    (repo / "f.txt").write_text("f\n", encoding="utf-8")
    result = backend.execute(
        'git add f.txt && git commit -m "feat: subject\n\nSecond paragraph explains it."'
    )
    assert result.exit_code == 0, result.output
    assert _git(repo, "log", "-1", "--format=%B").strip() == (
        "feat: subject\n\nSecond paragraph explains it."
    )


@posix_only
def test_context_file_unstaged_between_add_and_commit(tmp_path, git_repo):
    repo = git_repo(tmp_path / "repo")
    backend = GuardedShellBackend(root_dir=repo)
    (repo / "f.txt").write_text("f\n", encoding="utf-8")
    subprocess.run(["git", "add", "-f", "AbsolutCoder.md"], cwd=repo, check=True, capture_output=True)
    result = backend.execute('git add -A && git commit -m "x"')
    assert result.exit_code == 0, result.output
    committed = _git(repo, "show", "--name-only", "--format=", "HEAD").splitlines()
    assert "f.txt" in committed
    assert "AbsolutCoder.md" not in committed
    assert "AbsolutCoder.md" not in _git(repo, "diff", "--cached", "--name-only")


def _commit_in_threads(repo: Path, backends: list[GuardedShellBackend], names: list[str]) -> list:
    barrier = threading.Barrier(len(names))
    results: list = [None] * len(names)

    def work(index: int) -> None:
        name = names[index]
        (repo / f"{name}.txt").write_text(name + "\n", encoding="utf-8")
        barrier.wait()
        results[index] = backends[index].execute(f'git add {name}.txt && git commit -m "add {name}"')

    threads = [threading.Thread(target=work, args=(i,)) for i in range(len(names))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results


def test_concurrent_commits_both_land(tmp_path, git_repo):
    repo = git_repo(tmp_path / "repo")
    shared = GuardedShellBackend(root_dir=repo)
    results = _commit_in_threads(repo, [shared, shared], ["one", "two"])
    results += _commit_in_threads(
        repo, [GuardedShellBackend(root_dir=repo), GuardedShellBackend(root_dir=repo)], ["three", "four"]
    )
    for result in results:
        assert result.exit_code == 0, result.output
        assert "index.lock" not in result.output
    subjects = set(_git(repo, "log", "--format=%s").splitlines())
    assert subjects == {"add one", "add two", "add three", "add four"}


def test_nonzero_exit_and_stderr_are_reported(tmp_path):
    backend = GuardedShellBackend(root_dir=tmp_path)
    if backend.shell_kind == "cmd":
        command = "echo oops 1>&2 & exit /b 3"
    else:
        command = "echo oops 1>&2; exit 3"
    result = backend.execute(command)
    assert result.exit_code == 3
    assert "[stderr] oops" in result.output
    assert result.output.endswith("\n\nExit code: 3")


def test_timeout_is_reported_like_the_base_class(tmp_path):
    backend = GuardedShellBackend(root_dir=tmp_path)
    command = "ping -n 3 127.0.0.1 >nul" if backend.shell_kind == "cmd" else "sleep 2"
    result = backend.execute(command, timeout=1)
    assert result.exit_code == 124
    assert result.output == (
        "Error: Command timed out after 1 seconds (custom timeout). "
        "The command may be stuck or require more time."
    )
    with pytest.raises(ValueError, match="timeout must be positive"):
        backend.execute("echo hi", timeout=0)


def test_vet_log_is_consistent_under_threads(tmp_path):
    backend = GuardedShellBackend(root_dir=tmp_path)
    threads = [threading.Thread(target=backend.execute, args=(f"echo {i}",)) for i in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(backend.vet_log) == 20
    assert all(entry["allowed"] for entry in backend.vet_log)


def test_shell_line_reaches_main_and_shell_roles(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    backend = GuardedShellBackend(root_dir=repo)
    line = (
        f"Shell commands run with {backend.shell_kind}; each command is one {backend.shell_kind} "
        "script, so cd and variables last until the command ends."
    )
    systems: dict[str, str] = {}

    def recorder(role, reply):
        def respond(messages):
            systems.setdefault(role, messages[0].text if messages[0].type == "system" else "")
            return reply(messages)

        return scripted_model(respond)

    def main(messages):
        if any(getattr(m, "type", None) == "tool" for m in messages):
            return AIMessage(content="done")
        return tool_calls(
            task_call("general-purpose", "look", "t1"), task_call("explore", "look", "t2")
        )

    graph = build_agent(
        repo,
        Config(main_model="openai:main-test", subagent_model="openai:worker-test", auto_run=True),
        InMemorySaver(),
        chat_model=recorder("main", main),
        model_factory=role_models(
            {
                "general-purpose": recorder("general-purpose", lambda _m: AIMessage(content="ok")),
                "explore": recorder("explore", lambda _m: AIMessage(content="ok")),
            },
            default=scripted_model(lambda _m: AIMessage(content="ok")),
        ),
        backend=backend,
    )
    graph.invoke({"messages": [{"role": "user", "content": "go"}]}, {"configurable": {"thread_id": "t"}})
    assert systems["main"].count(line) == 1
    assert systems["general-purpose"].count(line) == 1
    assert line not in systems["explore"]


def test_bad_shell_setting_is_a_config_error(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    config = Config(main_model="openai:m", subagent_model="openai:s", shell="definitely-missing-shell")
    assert run_cli(SimpleNamespace(task="hi", attach=[]), config) == 1
    err = capsys.readouterr().err
    assert err.startswith("Config error: shell:")
    assert "Traceback" not in err


def test_find_shell_preferences():
    if os.name == "nt":
        assert find_shell("cmd")[0] == "cmd"
        kind, argv = find_shell("auto")
        if kind != "cmd":
            # never WSL's C:\Windows\System32\bash.exe
            assert "system32" not in argv[0].replace("/", "\\").lower().split("\\")
    with pytest.raises(ValueError):
        find_shell("definitely-missing-shell")
