"""Provider-aware model ids, model construction and key checks."""

import json

import pytest

from absolutcoder.config import Config, load_config, save_config
from absolutcoder.models import (
    ProviderSettings,
    build_model,
    effective_api,
    install_hint,
    missing_keys,
    normalize_model_id,
    provider_of,
)

LOCAL = {
    "local": ProviderSettings(
        kind="openai", base_url="http://localhost:11434/v1", api="chat"
    )
}


def test_provider_prefix_is_kept():
    assert normalize_model_id("anthropic:claude-x") == "anthropic:claude-x"
    assert normalize_model_id(" Ollama:llama3:8b ") == "ollama:llama3:8b"
    assert provider_of("gpt-x") == "openai"


def test_default_openai_passes_string_through():
    first = build_model("gpt-x", {})
    second = build_model("anthropic:claude-x", {})
    assert first == "openai:gpt-x" and isinstance(first, str)
    assert second == "anthropic:claude-x" and isinstance(second, str)


def test_compatible_endpoint_uses_chat_completions(monkeypatch):
    from langchain_openai import ChatOpenAI

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    model = build_model("local:llama3", LOCAL)
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "llama3"
    assert model.openai_api_base == "http://localhost:11434/v1"
    assert model.use_responses_api is False


def test_missing_keys_follow_configured_providers():
    config = Config(main_model="anthropic:a", subagent_model="openai:b")
    assert missing_keys(config, {}) == ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]
    assert missing_keys(config, {"ANTHROPIC_API_KEY": "k"}) == ["OPENAI_API_KEY"]
    local = Config(main_model="local:x", subagent_model="local:x", providers=LOCAL)
    assert missing_keys(local, {}) == []
    keyed = {
        "local": ProviderSettings(
            kind="openai", base_url="http://h/v1", api="chat", api_key_env="MY_KEY"
        )
    }
    config = Config(main_model="local:x", subagent_model="local:x", providers=keyed)
    assert missing_keys(config, {}) == ["MY_KEY"]


def test_roster_model_ids_are_key_checked():
    config = Config(main_model="openai:a", subagent_model="openai:b")
    assert missing_keys(config, {"OPENAI_API_KEY": "k"}, extra_model_ids=["anthropic:x"]) == [
        "ANTHROPIC_API_KEY"
    ]


def test_google_accepts_either_key():
    config = Config(main_model="google_genai:g", subagent_model="google_genai:g")
    assert missing_keys(config, {}) == ["GOOGLE_API_KEY or GEMINI_API_KEY"]
    assert missing_keys(config, {"GEMINI_API_KEY": "k"}) == []
    assert missing_keys(config, {"GOOGLE_API_KEY": "k"}) == []


def test_install_hint_names_extra():
    message = (
        "Initializing ChatAnthropic requires the langchain-anthropic package. "
        "Please install it with `pip install langchain-anthropic`"
    )
    assert install_hint(ImportError(message)) == 'pip install "absolutcoder[anthropic]"'
    other = message.replace("langchain-anthropic", "langchain-groq")
    assert install_hint(ImportError(other)) == "pip install langchain-groq"
    assert install_hint(ImportError("boom")) is None


def test_old_config_file_still_loads(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"main_model": "openai:a", "subagent_model": "openai:b", "auto_run": True}),
        encoding="utf-8",
    )
    loaded = load_config(path)
    assert loaded.providers == {}
    assert loaded.agent_models == {}
    assert loaded.shell == "auto"
    assert loaded.max_rounds == 5


def test_new_fields_round_trip(tmp_path):
    path = tmp_path / "config.json"
    original = Config(
        main_model="local:a",
        subagent_model="openai:b",
        providers=LOCAL,
        agent_models={"researcher": "anthropic:c"},
    )
    save_config(path, original)
    assert load_config(path) == original


def test_unset_alias_key_never_falls_back_to_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
    monkeypatch.delenv("MY_KEY", raising=False)
    providers = {
        "local": ProviderSettings(
            kind="openai", base_url="http://localhost:1/v1", api="chat", api_key_env="MY_KEY"
        )
    }
    model = build_model("local:llama3", providers)
    assert model.openai_api_key.get_secret_value() == "not-set"


def test_alias_key_is_read_from_its_variable(monkeypatch):
    monkeypatch.setenv("MY_KEY", "k1")
    providers = {
        "local": ProviderSettings(
            kind="openai", base_url="http://localhost:1/v1", api="chat", api_key_env="MY_KEY"
        )
    }
    model = build_model("local:llama3", providers)
    assert model.openai_api_key.get_secret_value() == "k1"


def test_provider_names_are_lowercased_on_load(tmp_path):
    path = tmp_path / "config.json"
    base = {"main_model": "openai:a", "subagent_model": "openai:b"}
    path.write_text(json.dumps({**base, "providers": {"Local": {"kind": "openai"}}}))
    assert list(load_config(path).providers) == ["local"]
    path.write_text(json.dumps({**base, "providers": {"local": "x"}}))
    with pytest.raises(ValueError, match="providers.local must be an object"):
        load_config(path)


def test_openai_key_never_sent_to_alias_endpoint(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
    providers = {
        "local": ProviderSettings(kind="openai", base_url="http://example.com/v1", api="chat")
    }
    model = build_model("local:llama3", providers)
    assert model.openai_api_key.get_secret_value() == "not-needed"


def test_openai_key_never_sent_to_overridden_openai_base_url(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
    providers = {"openai": ProviderSettings(base_url="http://example.com/v1")}
    model = build_model("gpt-x", providers)
    assert model.openai_api_key.get_secret_value() == "not-needed"


def test_alias_defaults_to_chat_completions(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    providers = {"local": ProviderSettings(kind="openai", base_url="http://localhost:1/v1")}
    model = build_model("local:llama3", providers)
    assert model.use_responses_api is False
    assert effective_api("openai", ProviderSettings()) == "responses"
    assert effective_api("openai", ProviderSettings(base_url="http://x/v1")) == "chat"
    assert effective_api("openai", ProviderSettings(api="chat")) == "chat"
    with pytest.raises(ValueError, match="providers.local.api must be"):
        effective_api("local", ProviderSettings(api="rest"))


def test_invalid_api_value_is_a_config_error(tmp_path):
    path = tmp_path / "config.json"
    base = {"main_model": "openai:a", "subagent_model": "openai:b"}
    path.write_text(json.dumps({**base, "providers": {"local": {"kind": "openai", "api": "rest"}}}))
    with pytest.raises(ValueError, match="providers.local.api must be 'responses' or 'chat'"):
        load_config(path)


def test_unset_api_is_saved_as_null(tmp_path):
    path = tmp_path / "config.json"
    save_config(
        path,
        Config(main_model="local:a", subagent_model="local:a", providers={"local": ProviderSettings(kind="openai")}),
    )
    assert json.loads(path.read_text())["providers"]["local"]["api"] is None


def test_provider_entry_does_not_disable_builtin_key_check():
    config = Config(
        main_model="openai:a",
        subagent_model="openai:a",
        providers={"openai": ProviderSettings(api="chat")},
    )
    assert missing_keys(config, {}) == ["OPENAI_API_KEY"]
    config = Config(
        main_model="anthropic:a",
        subagent_model="anthropic:a",
        providers={"anthropic": ProviderSettings()},
    )
    assert missing_keys(config, {}) == ["ANTHROPIC_API_KEY"]
    config = Config(
        main_model="openai:a",
        subagent_model="openai:a",
        providers={"openai": ProviderSettings(base_url="http://x/v1")},
    )
    assert missing_keys(config, {}) == []


def test_missing_package_name():
    from absolutcoder.models import missing_package

    message = "Initializing ChatGroq requires the langchain-groq package."
    assert missing_package(ImportError(message)) == "langchain-groq"
    assert missing_package(ImportError("boom")) is None


def test_non_object_sections_are_config_errors(tmp_path):
    path = tmp_path / "config.json"
    base = {"main_model": "openai:a", "subagent_model": "openai:b"}
    for key in ("providers", "agent_models"):
        path.write_text(json.dumps({**base, key: []}))
        with pytest.raises(ValueError, match=f"{key} must be an object"):
            load_config(path)
    path.write_text("[]")
    with pytest.raises(ValueError, match="top level"):
        load_config(path)


def test_openai_key_env_without_base_url_is_used_for_requests(monkeypatch):
    from langchain_openai import ChatOpenAI

    monkeypatch.setenv("OPENAI_API_KEY", "default-key")
    monkeypatch.setenv("WORK_OPENAI_KEY", "work-key")
    providers = {"openai": ProviderSettings(api_key_env="WORK_OPENAI_KEY")}
    model = build_model("openai:gpt-x", providers)
    assert isinstance(model, ChatOpenAI)
    assert model.openai_api_key.get_secret_value() == "work-key"
    assert model.use_responses_api is True
    # Without api_key_env the id string still goes to deepagents, which reads OPENAI_API_KEY.
    assert build_model("openai:gpt-x", {"openai": ProviderSettings()}) == "openai:gpt-x"


def test_provider_entry_without_kind_explains_itself():
    from absolutcoder.models import ProviderMissing

    providers = {"mybox": ProviderSettings(base_url="http://h/v1")}
    with pytest.raises(ProviderMissing) as raised:
        build_model("mybox:model-a", providers)
    assert str(raised.value) == (
        'providers.mybox needs "kind": "openai" to be used as an OpenAI-compatible endpoint.'
    )
    # A provider langchain knows keeps its own handling.
    assert build_model("anthropic:claude-x", {"anthropic": ProviderSettings()}) == "anthropic:claude-x"


def test_dependencies_pin_the_libraries_used_directly():
    import tomllib
    from pathlib import Path

    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    dependencies = project["dependencies"]
    for expected in (
        "deepagents==0.7.35",
        "langchain>=1.5,<1.6",
        "langchain-core>=1.8,<1.9",
        "langgraph>=1.2,<1.3",
        "langchain-openai",
        "langgraph-checkpoint-sqlite",
        "rich",
        "httpx",
        "pyyaml",
        "ddgs",
        "prompt_toolkit",
    ):
        assert expected in dependencies, expected
