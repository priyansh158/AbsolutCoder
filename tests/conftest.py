"""Keep the suite off the network and away from the real home directory."""

import pytest


@pytest.fixture(autouse=True)
def isolated_absolutcoder(monkeypatch, tmp_path):
    monkeypatch.setenv("ABSOLUTCODER_HOME", str(tmp_path / "absolutcoder-home"))
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING", "false")
    monkeypatch.setenv("LANGSMITH_OTEL_ENABLED", "false")
