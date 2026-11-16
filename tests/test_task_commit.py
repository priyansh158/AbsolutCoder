"""A scripted turn commits locally, and push is allowed only when asked."""

import shlex
import subprocess
from pathlib import Path

from langchain_core.messages import AIMessage

from absolutcoder.config import Config
from absolutcoder.repl import Repl
from tests.support import last_human_text, role_models, scripted_model, tool_call, tool_names_since_last_human

_COMMIT_MESSAGE = (
    "feat: add hello\n\n"
    "Co-authored-by: Cursor <cursoragent@cursor.com>\n"
)


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _responder(messages):
    human = last_human_text(messages).lower()
    names = tool_names_since_last_human(messages)
    if "push" in human:
        if "execute" not in names:
            return tool_call("execute", {"command": "git push"}, "call-push")
        return AIMessage(content="push attempted")
    if "write_file" not in names:
        return tool_call(
            "write_file",
            {"file_path": "/hello.txt", "content": "hello\n"},
            "call-write",
        )
    if "execute" not in names:
        command = "git add hello.txt && git commit -m " + shlex.quote(_COMMIT_MESSAGE)
        return tool_call("execute", {"command": command}, "call-commit")
    return AIMessage(content="committed")


def _allowed_push(entries) -> bool:
    return any(entry["allowed"] and "git push" in entry["command"] for entry in entries)


def test_scripted_turn_commits_locally_and_pushes_only_when_asked(tmp_path, monkeypatch, git_repo):
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Not Ada")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "not@example.com")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Not Ada")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "not@example.com")
    repo = git_repo(tmp_path / "repo")
    repl = Repl(
        repo,
        Config(main_model="openai:main-test", subagent_model="openai:worker-test", auto_run=True),
        chat_model=scripted_model(_responder),
        model_factory=role_models(
            {"general-purpose": scripted_model(lambda _m: AIMessage(content="worker"))}
        ),
    )
    repl.run_turn("Add hello.txt containing hello and commit it")
    first_log = list(repl.turn_vet_log)

    assert (repo / "hello.txt").read_text(encoding="utf-8") == "hello\n"
    assert _git(repo, "rev-list", "--count", "HEAD").strip() == "1"
    assert _git(repo, "log", "-1", "--format=%an").strip() == "Ada Fixture"
    assert _git(repo, "log", "-1", "--format=%cn").strip() == "Ada Fixture"
    body = _git(repo, "log", "-1", "--format=%B")
    assert "feat: add hello" in body
    assert "Co-authored-by" not in body
    assert "AbsolutCoder.md" not in _git(repo, "ls-files")
    ignored = subprocess.run(
        ["git", "check-ignore", "AbsolutCoder.md"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    assert ignored.returncode == 0
    assert not _allowed_push(first_log)

    repl.run_turn("please push the branch")
    assert _allowed_push(repl.turn_vet_log)
    assert not _allowed_push(first_log)
