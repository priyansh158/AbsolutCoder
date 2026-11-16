"""Provider-aware model ids, model construction and API key checks."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable, Mapping
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
    api: str | None = None  # "responses" | "chat"; None picks a default
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


def effective_api(provider: str, settings: ProviderSettings) -> str:
    """Resolve ``responses`` or ``chat``; compatible servers default to chat."""

    if settings.api is not None:
        if settings.api not in ("responses", "chat"):
            raise ValueError(f"providers.{provider}.api must be 'responses' or 'chat'")
        return settings.api
    if provider != "openai" or settings.base_url:
        return "chat"
    return "responses"


def build_model(
    model_id: str, providers: Mapping[str, ProviderSettings]
) -> BaseChatModel | str:
    """Return a model instance for OpenAI-compatible setups, else the id string."""

    normalized = normalize_model_id(model_id)
    provider, _, rest = normalized.partition(":")
    settings = providers.get(provider)
    if settings is None:
        return normalized
    api = effective_api(provider, settings)
    builds_instance = settings.kind == "openai" or (
        provider == "openai" and (api == "chat" or settings.api_key_env is not None)
    )
    if not builds_instance:
        if settings.kind is None and provider not in _known_providers():
            raise ProviderMissing(
                f'providers.{provider} needs "kind": "openai" to be used as an OpenAI-compatible endpoint.'
            )
        return normalized

    from langchain.chat_models import init_chat_model

    extra: dict[str, str] = {}
    if settings.base_url:
        extra["base_url"] = settings.base_url
    if provider != "openai" or settings.base_url or settings.api_key_env:
        # The key comes from api_key_env when one is named; a custom endpoint
        # never falls back to OPENAI_API_KEY.
        value = os.environ.get(settings.api_key_env, "").strip() if settings.api_key_env else ""
        extra["api_key"] = value or ("not-set" if settings.api_key_env else "not-needed")
    return init_chat_model(
        model=rest,
        model_provider="openai",
        use_responses_api=(api == "responses"),
        **extra,
    )


def _known_providers() -> frozenset[str]:
    """Provider prefixes langchain's init_chat_model understands (a langchain internal)."""

    try:
        from langchain.chat_models import base
    except ImportError:
        return frozenset()
    return frozenset(getattr(base, "_BUILTIN_PROVIDERS", {}) or ())


def _is_set(environ: Mapping[str, str], name: str) -> bool:
    return bool(environ.get(name, "").strip())


def missing_keys(
    config: Config, environ: Mapping[str, str], extra_model_ids: Iterable[str] = ()
) -> list[str]:
    """Names of API key variables the configured providers need but lack.

    ``extra_model_ids`` (for example ids pinned by worker roles) are checked
    like the configured ones.
    """

    ids = [
        config.main_model,
        config.subagent_model,
        *config.agent_models.values(),
        *extra_model_ids,
    ]
    seen: list[str] = []
    for model_id in ids:
        provider = provider_of(model_id)
        if provider not in seen:
            seen.append(provider)

    missing: list[str] = []
    for provider in seen:
        settings = config.providers.get(provider)
        if settings is not None and settings.api_key_env:
            if not _is_set(environ, settings.api_key_env):
                missing.append(settings.api_key_env)
        elif settings is not None and settings.base_url:
            continue
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


def missing_package(error: ImportError) -> str | None:
    """Name of the package langchain says is missing, if the message names one."""

    match = _PACKAGE_RE.search(str(error))
    return None if match is None else match.group(1)


def install_hint(error: ImportError) -> str | None:
    """Turn langchain's missing-package message into an install command."""

    package = missing_package(error)
    if package is None:
        return None
    extra = _EXTRAS.get(package)
    if extra is not None:
        return f'pip install "absolutcoder[{extra}]"'
    return f"pip install {package}"


def provider_error(error: Exception, model_id: str) -> ProviderMissing | None:
    """Map a model-construction failure to ProviderMissing, or None if unrecognised."""

    if isinstance(error, ImportError):
        hint = install_hint(error)
        if hint is None:
            return None
        package = missing_package(error)
        return ProviderMissing(
            f"The {package} package for this model is not installed. Install it with: {hint}"
        )
    if isinstance(error, ValueError) and str(error).startswith("Unable to infer model provider"):
        return ProviderMissing(
            f"Unknown model provider in {model_id!r}. Use provider:model, for example "
            "openai:gpt-x or anthropic:claude-x, or define the provider under "
            '"providers" in config.json.'
        )
    return None


def construct_model(factory: ModelFactory, role: str, model_id: str) -> BaseChatModel | str:
    """Call the model factory; every failure becomes ProviderMissing."""

    try:
        return factory(role, model_id)
    except ProviderMissing:
        raise
    except Exception as error:
        converted = provider_error(error, model_id)
        if converted is None:
            converted = ProviderMissing(f"Could not set up {model_id}: {error}")
        raise converted from error
