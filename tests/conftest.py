"""Keep the suite off the network and away from the real home directory."""

import subprocess
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_absolutcoder(monkeypatch, tmp_path):
    monkeypatch.setenv("ABSOLUTCODER_HOME", str(tmp_path / "absolutcoder-home"))
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING", "false")
    monkeypatch.setenv("LANGSMITH_OTEL_ENABLED", "false")


@pytest.fixture
def git_repo():
    """Return ``make(path)`` which creates a git repo with a fixed test identity."""

    def run(path: Path, *args: str) -> None:
        subprocess.run(["git", *args], cwd=path, check=True, capture_output=True, text=True)

    def make(path: Path) -> Path:
        path.mkdir(parents=True)
        run(path, "init")
        run(path, "config", "user.name", "Ada Fixture")
        run(path, "config", "user.email", "ada@example.com")
        run(path, "config", "commit.gpgsign", "false")
        run(path, "config", "core.autocrlf", "false")
        hooks = path / "empty-hooks"
        hooks.mkdir()
        run(path, "config", "core.hooksPath", str(hooks))
        return path

    return make
