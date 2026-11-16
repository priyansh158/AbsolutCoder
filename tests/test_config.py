"""Config files round-trip, and provider-prefixed model ids are accepted."""

import json

import pytest

from absolutcoder.config import Config, load_config, resolve_model, save_config


def test_bare_model_id_becomes_openai():
    assert resolve_model("gpt-6.1-sol") == "openai:gpt-6.1-sol"


def test_openai_prefix_is_preserved():
    assert resolve_model("openai:gpt-6.1-sol") == "openai:gpt-6.1-sol"


def test_malformed_provider_is_rejected():
    for bad in (":x", "a b:x", "openai:"):
        with pytest.raises(ValueError):
            resolve_model(bad)


def test_other_provider_is_accepted():
    assert resolve_model("anthropic:claude-x") == "anthropic:claude-x"


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


def _write(tmp_path, **fields):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"main_model": "openai:a", "subagent_model": "openai:b", **fields}), encoding="utf-8")
    return path


def test_verify_timeout_defaults_and_loads(tmp_path):
    assert load_config(_write(tmp_path)).verify_timeout == 1800
    assert load_config(_write(tmp_path, verify_timeout=60)).verify_timeout == 60


@pytest.mark.parametrize(
    "key", ["verify_timeout", "max_rounds", "web_max_chars", "max_background_tasks", "code_map_min_files"]
)
@pytest.mark.parametrize("value", [0, -1, "5", 2.5, True, None])
def test_numeric_settings_must_be_positive_integers(tmp_path, key, value):
    with pytest.raises(ValueError, match=key):
        load_config(_write(tmp_path, **{key: value}))


def test_code_map_settings_default_and_round_trip(tmp_path):
    loaded = load_config(_write(tmp_path))
    assert loaded.code_map == "auto"
    assert loaded.code_map_min_files == 20
    path = tmp_path / "saved.json"
    original = Config(
        main_model="openai:a", subagent_model="openai:b", code_map="off", code_map_min_files=7
    )
    save_config(path, original)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["code_map"] == "off"
    assert payload["code_map_min_files"] == 7
    again = load_config(path)
    assert (again.code_map, again.code_map_min_files) == ("off", 7)


def test_bad_code_map_settings_are_config_errors(tmp_path):
    with pytest.raises(ValueError, match="code_map must be 'auto' or 'off'"):
        load_config(_write(tmp_path, code_map="sometimes"))
    with pytest.raises(ValueError, match="code_map_min_files must be a positive integer"):
        load_config(_write(tmp_path, code_map_min_files=0))


@pytest.mark.parametrize("key", ["main_model", "subagent_model"])
def test_model_ids_must_be_strings(tmp_path, key):
    with pytest.raises(ValueError, match=key):
        load_config(_write(tmp_path, **{key: 5}))
