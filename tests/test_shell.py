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


def _identity(repo: Path) -> str:
    return _git(repo, "log", "-1", "--format=%an <%ae> / %cn <%ce>").strip()


_REPO_IDENTITY = "Ada Fixture <ada@example.com> / Ada Fixture <ada@example.com>"


@posix_only
def test_identity_set_by_unlisted_form_does_not_reach_commit(tmp_path, git_repo):
    repo = git_repo(tmp_path / "repo")
    backend = GuardedShellBackend(root_dir=repo)
    (repo / "f.txt").write_text("f\n", encoding="utf-8")
    result = backend.execute(
        "set -a; printf -v GIT_AUTHOR_NAME evil; printf -v GIT_AUTHOR_EMAIL e@vil; "
        'git add f.txt && git commit -m "x"'
    )
    assert result.exit_code == 0, result.output
    assert _identity(repo) == _REPO_IDENTITY

    (tmp_path / "ident.sh").write_text(
        "export GIT_COMMITTER_NAME=evil\nexport GIT_COMMITTER_EMAIL=e@vil\n", encoding="utf-8"
    )
    (repo / "g.txt").write_text("g\n", encoding="utf-8")
    result = backend.execute('. ../ident.sh; git add g.txt && git commit -m "y"')
    assert result.exit_code == 0, result.output
    assert _git(repo, "rev-list", "--count", "HEAD").strip() == "2"
    assert _identity(repo) == _REPO_IDENTITY


@posix_only
def test_shell_function_named_git_cannot_replace_commit(tmp_path, git_repo):
    repo = git_repo(tmp_path / "repo")
    backend = GuardedShellBackend(root_dir=repo)
    (repo / "f.txt").write_text("f\n", encoding="utf-8")
    subprocess.run(["git", "add", "f.txt"], cwd=repo, check=True, capture_output=True)
    result = backend.execute(
        'git() { echo fake; }; git commit -m "feat: x" -m "Co-authored-by: A <a@b.c>"'
    )
    assert result.exit_code == 0, result.output
    assert _git(repo, "rev-list", "--count", "HEAD").strip() == "1"
    assert _git(repo, "log", "-1", "--format=%B").strip() == "feat: x"


def _finishes_while_locked(repo: Path, backend: GuardedShellBackend, command: str, wait: float) -> bool:
    """Run ``command`` in another thread while this thread holds the repo lock."""

    done = threading.Event()
    with shell_module.repo_lock(repo):
        worker = threading.Thread(target=lambda: (backend.execute(command), done.set()))
        worker.start()
        finished = done.wait(timeout=wait)
    worker.join(timeout=30)
    assert done.is_set()  # it always finishes once the lock is free
    return finished


def test_non_git_command_does_not_take_the_repo_lock(tmp_path, git_repo):
    repo = git_repo(tmp_path / "repo")
    backend = GuardedShellBackend(root_dir=repo)
    assert _finishes_while_locked(repo, backend, "echo git is mentioned", wait=10)


def test_git_command_waits_for_the_repo_lock(tmp_path, git_repo):
    repo = git_repo(tmp_path / "repo")
    backend = GuardedShellBackend(root_dir=repo)
    assert not _finishes_while_locked(repo, backend, "git status", wait=2)


def test_backends_in_one_repo_share_a_lock(tmp_path, git_repo):
    repo = git_repo(tmp_path / "repo")
    (repo / "sub").mkdir()
    root = GuardedShellBackend(root_dir=repo)
    sub = GuardedShellBackend(root_dir=repo / "sub")
    assert root.git_lock is sub.git_lock
    assert root.git_lock is shell_module.repo_lock(repo)
    plain = GuardedShellBackend(root_dir=tmp_path / "not-a-repo")
    assert plain.git_lock is shell_module.repo_lock(tmp_path / "not-a-repo")


@posix_only
def test_trailing_comment_keeps_the_group_valid(tmp_path, git_repo):
    repo = git_repo(tmp_path / "repo")
    backend = GuardedShellBackend(root_dir=repo)
    (repo / "f.txt").write_text("f\n", encoding="utf-8")
    result = backend.execute('git add f.txt && git commit -m "x" # note')
    assert result.exit_code == 0, result.output
    assert _git(repo, "rev-list", "--count", "HEAD").strip() == "1"
    assert _git(repo, "log", "-1", "--format=%B").strip() == "x"


@posix_only
def test_backgrounded_commit_runs_in_the_foreground(tmp_path, git_repo):
    repo = git_repo(tmp_path / "repo")
    backend = GuardedShellBackend(root_dir=repo)
    (repo / "f.txt").write_text("f\n", encoding="utf-8")
    subprocess.run(["git", "add", "f.txt"], cwd=repo, check=True, capture_output=True)
    result = backend.execute('git commit -m "x" &')
    assert result.exit_code == 0, result.output
    assert _git(repo, "rev-list", "--count", "HEAD").strip() == "1"
    assert _git(repo, "log", "-1", "--format=%B").strip() == "x"


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


# --- A1: here-document bodies stay byte-identical when the command is rewritten


_APP_BODY = "def f():\n    return 1\n\nx = 2\n"


@posix_only
def test_heredoc_body_survives_a_rewritten_add(tmp_path, git_repo):
    repo = git_repo(tmp_path / "repo")
    backend = GuardedShellBackend(root_dir=repo)
    result = backend.execute("cat > app.py <<'EOF'\n" + _APP_BODY + "EOF\ngit add app.py")
    assert result.exit_code == 0, result.output
    assert (repo / "app.py").read_bytes() == _APP_BODY.encode()
    assert _git(repo, "diff", "--cached", "--name-only").splitlines() == ["app.py"]


@posix_only
def test_heredoc_body_survives_a_rewritten_commit(tmp_path, git_repo):
    repo = git_repo(tmp_path / "repo")
    backend = GuardedShellBackend(root_dir=repo)
    result = backend.execute("cat > app.py <<'EOF'\n" + _APP_BODY + "EOF\ngit add app.py\ngit commit -m x")
    assert result.exit_code == 0, result.output
    assert _git(repo, "show", "HEAD:app.py") == _APP_BODY
    assert _git(repo, "log", "-1", "--format=%B").strip() == "x"


@posix_only
def test_heredoc_body_with_shell_text_is_written_verbatim(tmp_path, git_repo):
    repo = git_repo(tmp_path / "repo")
    backend = GuardedShellBackend(root_dir=repo)
    (repo / "other.txt").write_text("other\n", encoding="utf-8")
    body = "git add other.txt\na && b; c | d\ntrailing space   \n\n  indented\n"
    result = backend.execute("cat > notes.txt <<'EOF' && git add notes.txt\n" + body + "EOF")
    assert result.exit_code == 0, result.output
    assert (repo / "notes.txt").read_bytes() == body.encode()
    assert _git(repo, "diff", "--cached", "--name-only").splitlines() == ["notes.txt"]


@posix_only
def test_commit_with_heredoc_message_still_scrubbed(tmp_path, git_repo):
    repo = git_repo(tmp_path / "repo")
    backend = GuardedShellBackend(root_dir=repo)
    (repo / "f.txt").write_text("f\n", encoding="utf-8")
    result = backend.execute(
        "git add f.txt && git commit -F - <<'EOF'\nfeat: x\n\n    indented body\nCo-authored-by: A <a@b.c>\nEOF"
    )
    assert result.exit_code == 0, result.output
    assert _git(repo, "log", "-1", "--format=%B").strip() == "feat: x\n\n    indented body"


# --- B: stored commit messages under a POSIX shell ---------------------------


def _staged_repo(tmp_path, git_repo):
    repo = git_repo(tmp_path / "repo")
    (repo / "f.txt").write_text("f\n", encoding="utf-8")
    subprocess.run(["git", "add", "f.txt"], cwd=repo, check=True, capture_output=True)
    return repo, GuardedShellBackend(root_dir=repo)


@posix_only
def test_substituted_heredoc_message_is_scrubbed(tmp_path, git_repo, monkeypatch):
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Not Ada")
    repo, backend = _staged_repo(tmp_path, git_repo)
    result = backend.execute(
        "git commit -m \"$(cat <<'EOF'\nfeat: x\n\nbody line\nCo-authored-by: A <a@b.c>\nEOF\n)\""
    )
    assert result.exit_code == 0, result.output
    assert _git(repo, "log", "-1", "--format=%B").strip() == "feat: x\n\nbody line"
    assert _identity(repo) == _REPO_IDENTITY


@posix_only
def test_variable_in_message_is_expanded(tmp_path, git_repo):
    repo, backend = _staged_repo(tmp_path, git_repo)
    result = backend.execute('NAME=world; git commit -m "hello $NAME" -m "Generated-by: AbsolutCoder"')
    assert result.exit_code == 0, result.output
    assert _git(repo, "log", "-1", "--format=%B").strip() == "hello world"


@posix_only
def test_glued_message_is_stored_whole(tmp_path, git_repo):
    repo, backend = _staged_repo(tmp_path, git_repo)
    result = backend.execute('git commit -m"fix algorithm"')
    assert result.exit_code == 0, result.output
    assert _git(repo, "log", "-1", "--format=%B").strip() == "fix algorithm"


@posix_only
def test_am_commits_tracked_changes(tmp_path, git_repo):
    repo, backend = _staged_repo(tmp_path, git_repo)
    assert backend.execute('git commit -m "first"').exit_code == 0
    (repo / "f.txt").write_text("changed\n", encoding="utf-8")
    result = backend.execute('git commit -am "x"')
    assert result.exit_code == 0, result.output
    assert _git(repo, "log", "-1", "--format=%B").strip() == "x"
    assert _git(repo, "show", "HEAD:f.txt") == "changed\n"
    assert _identity(repo) == _REPO_IDENTITY


@posix_only
def test_attribution_trailer_option_is_dropped(tmp_path, git_repo):
    repo, backend = _staged_repo(tmp_path, git_repo)
    result = backend.execute(
        'git commit -m "x" --trailer "Co-authored-by: A <a@b.c>" --trailer "Reviewed-by: B <b@c.d>"'
    )
    assert result.exit_code == 0, result.output
    body = _git(repo, "log", "-1", "--format=%B")
    assert "Co-authored-by" not in body
    assert "Reviewed-by: B <b@c.d>" in body


@posix_only
def test_merge_message_is_scrubbed(tmp_path, git_repo):
    repo, backend = _staged_repo(tmp_path, git_repo)
    assert backend.execute('git commit -m "base"').exit_code == 0
    assert backend.execute("git switch -c side").exit_code == 0
    (repo / "g.txt").write_text("g\n", encoding="utf-8")
    assert backend.execute('git add g.txt && git commit -m "side work"').exit_code == 0
    assert backend.execute("git switch -").exit_code == 0
    result = backend.execute('git merge --no-ff -m "merge\n\nGenerated-by: tool" side')
    assert result.exit_code == 0, result.output
    assert _git(repo, "log", "-1", "--format=%B").strip() == "merge"
    assert _identity(repo) == _REPO_IDENTITY


def _side_branch(tmp_path, git_repo, *, conflict=False):
    repo, backend = _staged_repo(tmp_path, git_repo)
    assert backend.execute('git commit -m "base"').exit_code == 0
    assert backend.execute("git switch -c side").exit_code == 0
    (repo / "f.txt").write_text("side\n", encoding="utf-8")
    assert backend.execute('git commit -am "side work"').exit_code == 0
    assert backend.execute("git switch -").exit_code == 0
    if conflict:
        (repo / "f.txt").write_text("main\n", encoding="utf-8")
        assert backend.execute('git commit -am "main work"').exit_code == 0
    return repo, backend


def _untracked(repo):
    return _git(repo, "status", "--porcelain", "--untracked-files=all").splitlines()


@posix_only
def test_merge_with_run_time_message_is_scrubbed(tmp_path, git_repo):
    repo, backend = _side_branch(tmp_path, git_repo)
    result = backend.execute('B=side; git merge --no-ff -m "Merge $B" -m "Generated-by: tool" side')
    assert result.exit_code == 0, result.output
    assert _git(repo, "log", "-1", "--format=%B").strip() == "Merge side"
    assert _git(repo, "rev-list", "--parents", "-1", "HEAD").count(" ") == 2
    assert _identity(repo) == _REPO_IDENTITY


@posix_only
def test_merge_with_message_file_is_scrubbed(tmp_path, git_repo):
    repo, backend = _side_branch(tmp_path, git_repo)
    (tmp_path / "msg.txt").write_text(
        "Merge the side work\n\nBody line.\nCo-authored-by: A <a@b.c>\n", encoding="utf-8"
    )
    result = backend.execute("git merge --no-ff -F ../msg.txt side")
    assert result.exit_code == 0, result.output
    assert _git(repo, "log", "-1", "--format=%B").strip() == "Merge the side work\n\nBody line."


@posix_only
def test_failing_merge_keeps_its_exit_code_and_leaves_no_file(tmp_path, git_repo):
    repo, backend = _side_branch(tmp_path, git_repo, conflict=True)
    result = backend.execute('B=side; git merge --no-ff -m "Merge $B" side && echo merged')
    assert result.exit_code != 0, result.output
    assert "merged" not in result.output
    leftovers = [line for line in _untracked(repo) if line.startswith("??")]
    assert leftovers == []
    assert not list((repo / ".git").glob("absolut-commit-*"))
