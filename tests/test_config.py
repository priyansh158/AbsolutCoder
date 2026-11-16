"""Config files round-trip, and only OpenAI model ids are accepted."""

import json

import pytest

from absolutcoder.config import Config, load_config, resolve_model, save_config


def test_bare_model_id_becomes_openai():
    assert resolve_model("gpt-6.1-sol") == "openai:gpt-6.1-sol"


def test_openai_prefix_is_preserved():
    assert resolve_model("openai:gpt-6.1-sol") == "openai:gpt-6.1-sol"


def test_other_provider_is_rejected():
    with pytest.raises(ValueError):
        resolve_model("anthropic:claude-sonnet-5")


def test_save_and_load_preserves_models_and_auto_run(tmp_path):
    path = tmp_path / "config.json"
    original = Config(
        main_model="openai:gpt-6.1-sol",
        subagent_model="openai:gpt-6.1-mini",
        auto_run=True,
    )
    save_config(path, original)
    loaded = load_config(path)
    assert loaded.main_model == original.main_model
    assert loaded.subagent_model == original.subagent_model
    assert loaded.auto_run is True
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["auto_run"] is True
