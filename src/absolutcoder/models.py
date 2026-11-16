"""Provider-aware model ids, model construction and API key checks."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain_core.language_models.chat_models import BaseChatModel

if TYPE_CHECKING:
    from absolutcoder.config import Config

# (role, model_id) -> model; role is "main" or the worker name.
ModelFactory = Callable[[str, str], BaseChatModel | str]

_PROVIDER_RE = re.compile(r"^[a-z0-9_]+$")
_PACKAGE_RE = re.compile(r"requires the ([\w-]+) package")
_EXTRAS = {
    "langchain-anthropic": "anthropic",
    "langchain-google-genai": "google",
    "langchain-ollama": "ollama",
}
_GOOGLE_KEYS = "GOOGLE_API_KEY or GEMINI_API_KEY"


@dataclass(frozen=True)
class ProviderSettings:
    """Per-provider overrides from config.json."""

    kind: str | None = None
    base_url: str | None = None
    api: str = "responses"  # "responses" | "chat"
    api_key_env: str | None = None


class ProviderMissing(RuntimeError):
    """Raised when a provider cannot be used (for example a missing package)."""


def normalize_model_id(model_id: str) -> str:
    """Return ``provider:model``; a bare id means OpenAI."""

    text = model_id.strip()
    if not text:
        raise ValueError("model id is empty")
    if ":" not in text:
        return f"openai:{text}"
    provider, _, rest = text.partition(":")
    provider = provider.lower()
    rest = rest.strip()
    if not _PROVIDER_RE.match(provider) or not rest:
        raise ValueError(f"invalid model id {model_id!r}; use provider:model")
    return f"{provider}:{rest}"


def provider_of(model_id: str) -> str:
    return normalize_model_id(model_id).partition(":")[0]


def build_model(
    model_id: str, providers: Mapping[str, ProviderSettings]
) -> BaseChatModel | str:
    """Return a model instance for OpenAI-compatible setups, else the id string."""

    normalized = normalize_model_id(model_id)
    provider, _, rest = normalized.partition(":")
    settings = providers.get(provider)
    if settings is None:
        return normalized
    if settings.kind == "openai":
        pass
    elif provider == "openai" and (settings.base_url or settings.api == "chat"):
        pass
    else:
        return normalized

    from langchain.chat_models import init_chat_model

    extra: dict[str, str] = {}
    if settings.base_url:
        extra["base_url"] = settings.base_url
    if settings.api_key_env:
        value = os.environ.get(settings.api_key_env, "").strip()
        if value:
            extra["api_key"] = value
    elif not os.environ.get("OPENAI_API_KEY", "").strip():
        extra["api_key"] = "not-needed"
    return init_chat_model(
        model=rest,
        model_provider="openai",
        use_responses_api=(settings.api == "responses"),
        **extra,
    )


def _is_set(environ: Mapping[str, str], name: str) -> bool:
    return bool(environ.get(name, "").strip())


def missing_keys(config: Config, environ: Mapping[str, str]) -> list[str]:
    """Names of API key variables the configured providers need but lack."""

    ids = [config.main_model, config.subagent_model, *config.agent_models.values()]
    seen: list[str] = []
    for model_id in ids:
        provider = provider_of(model_id)
        if provider not in seen:
            seen.append(provider)

    missing: list[str] = []
    for provider in seen:
        settings = config.providers.get(provider)
        if settings is not None:
            if settings.api_key_env and not _is_set(environ, settings.api_key_env):
                missing.append(settings.api_key_env)
        elif provider == "openai":
            if not _is_set(environ, "OPENAI_API_KEY"):
                missing.append("OPENAI_API_KEY")
        elif provider == "anthropic":
            if not _is_set(environ, "ANTHROPIC_API_KEY"):
                missing.append("ANTHROPIC_API_KEY")
        elif provider == "google_genai":
            if not (
                _is_set(environ, "GOOGLE_API_KEY") or _is_set(environ, "GEMINI_API_KEY")
            ):
                missing.append(_GOOGLE_KEYS)
    return missing


def install_hint(error: ImportError) -> str | None:
    """Turn langchain's missing-package message into an install command."""

    match = _PACKAGE_RE.search(str(error))
    if match is None:
        return None
    package = match.group(1)
    extra = _EXTRAS.get(package)
    if extra is not None:
        return f'pip install "absolutcoder[{extra}]"'
    return f"pip install {package}"
