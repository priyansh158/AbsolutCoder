"""Shell guard, commit scrubbing, and the untracked memory file."""

import os
import shlex
import subprocess
from pathlib import Path

import pytest

from absolutcoder.git_policy import (
    PolicyDenied,
    TurnFlags,
    execute_needs_approval,
    protect_context_file,
    render_command,
    scrub_commit_message,
    sets_commit_identity,
    unwrap_shell,
    vet,
)
from absolutcoder.shell import GuardedShellBackend


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def test_push_and_pull_require_flags(tmp_path, git_repo):
    repo = git_repo(tmp_path / "repo")
    backend = GuardedShellBackend(root_dir=repo)
    denied = TurnFlags()
    for command in (
        "git push",
        "git push --force",
        "git push -f",
        "git pull",
        "git fetch origin",
        "git status && git push",
        "git status && git pull",
    ):
        with pytest.raises(PolicyDenied):
            vet(command, denied)
        with pytest.raises(PolicyDenied):
            backend.execute(command)
    vet("git push", TurnFlags(push_requested=True))
    vet("git push --force", TurnFlags(push_requested=True))
    vet("git pull", TurnFlags(pull_requested=True))
    vet("git fetch", TurnFlags(pull_requested=True))
    assert not (repo / "should-not-exist").exists()


def test_git_config_reset_and_clean_require_the_destructive_flag(tmp_path, git_repo):
    repo = git_repo(tmp_path / "repo")
    backend = GuardedShellBackend(root_dir=repo)
    for command in ("git config user.name Mallory", "git reset --hard", "git clean -fd"):
        with pytest.raises(PolicyDenied):
            vet(command, TurnFlags())
        with pytest.raises(PolicyDenied):
            backend.execute(command)
    assert _git(repo, "config", "user.name").strip() == "Ada Fixture"
    allowed = TurnFlags(destructive_requested=True)
    vet("git config user.name Mallory", allowed)
    vet("git reset --hard", allowed)
    vet("git clean -fd", allowed)


def test_commit_drops_trailers_and_keeps_repo_identity(tmp_path, monkeypatch, git_repo):
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Not Ada")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "not@example.com")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Not Ada")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "not@example.com")
    repo = git_repo(tmp_path / "repo")
    backend = GuardedShellBackend(root_dir=repo)
    (repo / "file.txt").write_text("x\n", encoding="utf-8")
    message = (
        "feat: add file\n\n"
        "Co-authored-by: Cursor <cursoragent@cursor.com>\n"
        "Assisted-by: Claude\n"
        "Generated-by: AbsolutCoder\n"
    )
    backend.execute("git add file.txt && git commit -m " + shlex.quote(message))
    body = _git(repo, "log", "-1", "--format=%B")
    assert "feat: add file" in body
    assert "Co-authored-by" not in body
    assert "Assisted-by" not in body
    assert "Generated-by" not in body
    assert "AbsolutCoder" not in body
    assert _git(repo, "log", "-1", "--format=%an").strip() == "Ada Fixture"
    assert _git(repo, "log", "-1", "--format=%cn").strip() == "Ada Fixture"
    assert _git(repo, "config", "user.name").strip() == "Ada Fixture"
    assert os.environ["GIT_AUTHOR_NAME"] == "Not Ada"


def test_scrub_commit_message_removes_attribution():
    scrubbed = scrub_commit_message(
        "feat: work\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n"
        "Assisted-by: someone\nGenerated-by: AbsolutCoder\n"
    )
    assert scrubbed == "feat: work"


def test_context_file_is_excluded_and_unstaged(tmp_path, git_repo):
    repo = git_repo(tmp_path / "repo")
    protect_context_file(repo)
    (repo / "tracked.txt").write_text("y\n", encoding="utf-8")
    subprocess.run(["git", "add", "-f", "AbsolutCoder.md"], cwd=repo, check=True, capture_output=True)
    backend = GuardedShellBackend(root_dir=repo)
    backend.execute("git add -A")
    staged = _git(repo, "diff", "--cached", "--name-only").splitlines()
    assert "tracked.txt" in staged
    assert "AbsolutCoder.md" not in staged
    exclude = (repo / ".git" / "info" / "exclude").read_text(encoding="utf-8")
    assert "AbsolutCoder.md" in exclude
    with pytest.raises(PolicyDenied):
        backend.execute("git add AbsolutCoder.md")
    with pytest.raises(PolicyDenied):
        backend.execute("git add -f AbsolutCoder.md")


def test_exported_identity_is_denied():
    for command in (
        "export GIT_AUTHOR_NAME=x",
        "GIT_AUTHOR_NAME=x",
        "set GIT_AUTHOR_NAME=x",
        "setx GIT_AUTHOR_NAME x",
        '$env:GIT_AUTHOR_NAME = "x"',
        "declare -x GIT_COMMITTER_EMAIL=x",
        "env GIT_AUTHOR_NAME=x git commit -m y",
        "env GIT_AUTHOR_NAME=x make",
        "export GIT_COMMITTER_NAME=x; git commit -m y",
    ):
        assert sets_commit_identity(command.split(";")[0]), command
        with pytest.raises(PolicyDenied, match="commit identity variables cannot be set"):
            vet(command, TurnFlags())
    assert not sets_commit_identity("GIT_AUTHOR_NAME=x git commit -m y")
    vet("GIT_AUTHOR_NAME=x git commit -m y", TurnFlags())
    vet("echo $GIT_AUTHOR_NAME && git log --author=Ada", TurnFlags())


def test_render_command_quotes_for_posix_shells():
    tokens = ["git", "commit", "-m", "two words", "it's", "$HOME and $(id)", 'say "hi"', ""]
    for kind in ("bash", "sh"):
        assert shlex.split(render_command(tokens, kind)) == tokens


def test_turn_flags_follow_the_user_message():
    assert TurnFlags.from_message("please push the branch").push_requested is True
    assert TurnFlags.from_message("don't push").push_requested is False
    assert TurnFlags.from_message("pull the latest and fetch tags").pull_requested is True
    assert TurnFlags.from_message("git reset --hard HEAD").destructive_requested is True
    assert TurnFlags.from_message("commit the fix").push_requested is False


_ALL_FLAGS = TurnFlags(push_requested=True, pull_requested=True, destructive_requested=True)


@pytest.mark.parametrize(
    "command",
    [
        'sh -c "git push"',
        "bash -lc 'git push origin main'",
        "cmd /c git push",
        'cmd.exe /d /s /c "git push"',
        'powershell -Command "git push"',
        'eval "git push"',
        "env X=1 git push",
        "nohup git push",
        "bash -c \"sh -c 'git push'\"",
        '/bin/sh -c "git push"',
        'C:\\Windows\\System32\\cmd.exe /c "git push"',
        'if sh -c "git push"; then echo ok; fi',
        "sudo -u root sh -c 'git push'",
        "echo $(sh -c 'git push')",
    ],
)
def test_wrapped_git_push_is_denied(command):
    with pytest.raises(PolicyDenied):
        vet(command, TurnFlags())
    vet(command, TurnFlags(push_requested=True))


def test_wrapped_destructive_git_is_denied():
    with pytest.raises(PolicyDenied):
        vet('sh -c "git reset --hard"', TurnFlags())


@pytest.mark.parametrize(
    "command",
    [
        'sh -c "git commit -m x"',
        "bash -lc 'git add -A && git commit -m x'",
        'eval "git commit -m x"',
        "env X=1 git add f.txt",
        "echo $(git commit -m x)",
    ],
)
def test_wrapped_commit_and_add_are_refused(command):
    for flags in (TurnFlags(), _ALL_FLAGS):
        with pytest.raises(PolicyDenied, match="run git commit and git add directly"):
            vet(command, flags)
    vet("git commit -m x", TurnFlags())
    vet("git add f.txt", TurnFlags())


def test_wrapper_always_needs_approval():
    assert execute_needs_approval('sh -c "git commit -m x"') is True
    assert execute_needs_approval("git commit -m x") is False


@pytest.mark.parametrize("command", ["git status", 'python -c "print(1)"', "ls -la"])
def test_unwrap_shell_returns_none_for_plain_commands(command):
    assert unwrap_shell(shlex.split(command)) is None


def test_unwrap_shell_returns_the_inner_command():
    assert unwrap_shell(["bash", "-ec", "git status"]) == "git status"
    assert unwrap_shell(["cmd", "/C", "git", "push"]) == "git push"
    assert unwrap_shell(["pwsh.exe", "-NoProfile", "-c", "git", "push"]) == "git push"
    assert unwrap_shell(["eval", "git", "push"]) == "git push"
    assert shlex.split(unwrap_shell(["env", "-u", "HOME", "X=1", "sh", "-c", "git push"])) == [
        "sh",
        "-c",
        "git push",
    ]
    assert unwrap_shell(["X=1", "git", "push"]) == "git push"
    assert unwrap_shell(["sudo", "-u", "root", "git", "push"]) == "git push"


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("push this logic into a helper", False),
        ("push the button handler down into the base class", False),
        ("push the value onto the stack", False),
        ("please push", True),
        ("push it", True),
        ("commit and push the changes", True),
        ("git push", True),
        ("run git push into prod", True),
        ("then push to origin", True),
        ("don't push", False),
        ("commit the fix", False),
        ("fix the bug, then push.", True),
    ],
)
def test_push_as_ordinary_verb_does_not_unlock(message, expected):
    assert TurnFlags.from_message(message).push_requested is expected


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("pull this out into a function", False),
        ("open a pull request", False),
        ("pull the latest changes", True),
        ("fetch origin", True),
        ("git pull", True),
    ],
)
def test_pull_as_ordinary_verb_does_not_unlock(message, expected):
    assert TurnFlags.from_message(message).pull_requested is expected
