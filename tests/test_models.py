"""Provider-aware model ids, model construction and key checks."""

import json

from absolutcoder.config import Config, load_config, save_config
from absolutcoder.models import (
    ProviderSettings,
    build_model,
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
