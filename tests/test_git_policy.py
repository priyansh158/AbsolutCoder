"""Shell guard, commit scrubbing, and the untracked memory file."""

import os
import shlex
import subprocess
from pathlib import Path

import pytest

from absolutcoder.git_policy import PolicyDenied, TurnFlags, protect_context_file, scrub_commit_message, vet
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


def init_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    _git(path, "config", "user.name", "Ada Fixture")
    _git(path, "config", "user.email", "ada@example.com")
    _git(path, "config", "commit.gpgsign", "false")
    _git(path, "config", "core.autocrlf", "false")
    hooks = path / "empty-hooks"
    hooks.mkdir()
    _git(path, "config", "core.hooksPath", str(hooks))
    return path


def test_push_and_pull_require_flags(tmp_path):
    repo = init_repo(tmp_path / "repo")
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


def test_git_config_reset_and_clean_require_the_destructive_flag(tmp_path):
    repo = init_repo(tmp_path / "repo")
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


def test_commit_drops_trailers_and_keeps_repo_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Not Ada")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "not@example.com")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Not Ada")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "not@example.com")
    repo = init_repo(tmp_path / "repo")
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


def test_context_file_is_excluded_and_unstaged(tmp_path):
    repo = init_repo(tmp_path / "repo")
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


def test_turn_flags_follow_the_user_message():
    assert TurnFlags.from_message("please push the branch").push_requested is True
    assert TurnFlags.from_message("don't push").push_requested is False
    assert TurnFlags.from_message("pull the latest and fetch tags").pull_requested is True
    assert TurnFlags.from_message("git reset --hard HEAD").destructive_requested is True
    assert TurnFlags.from_message("commit the fix").push_requested is False
