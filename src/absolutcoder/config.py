"""Load and save AbsolutCoder settings, and resolve provider model ids."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

from absolutcoder.models import ProviderSettings, effective_api, normalize_model_id


def data_dir() -> Path:
    """Directory for config and session data.

    Tests point this at a temp folder with ABSOLUTCODER_HOME.
    """

    override = os.environ.get("ABSOLUTCODER_HOME", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".absolutcoder"


def config_path() -> Path:
    return data_dir() / "config.json"


@dataclass(frozen=True)
class Config:
    """Local settings. Model ids are resolved before they are stored."""

    main_model: str
    subagent_model: str
    auto_run: bool = False
    providers: dict[str, ProviderSettings] = field(default_factory=dict)
    agent_models: dict[str, str] = field(default_factory=dict)
    shell: str = "auto"
    verify_command: str | None = None
    max_rounds: int = 5
    web_max_chars: int = 20000
    max_background_tasks: int = 4
    verify_timeout: int = 1800  # seconds the check command may run
    code_map: str = "auto"  # "auto" | "off"
    code_map_min_files: int = 20  # fewer code files than this: no code map


_PROVIDER_FIELDS = {f.name for f in fields(ProviderSettings)}
_POSITIVE_INTS = (
    "max_rounds",
    "web_max_chars",
    "max_background_tasks",
    "verify_timeout",
    "code_map_min_files",
)
_CODE_MAP_MODES = ("auto", "off")


def load_config(path: Path | str) -> Config:
    """Read a config file. Model strings are returned as stored."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config top level must be an object")
    for key in ("providers", "agent_models"):
        if payload.get(key) is not None and not isinstance(payload[key], dict):
            raise ValueError(f"{key} must be an object")
    defaults = Config(main_model="", subagent_model="")
    for key in ("main_model", "subagent_model"):
        if key in payload and not isinstance(payload[key], str):
            raise ValueError(f"{key} must be a string")
    counts = {key: payload.get(key, getattr(defaults, key)) for key in _POSITIVE_INTS}
    for key, value in counts.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    code_map = payload.get("code_map", defaults.code_map)
    if code_map not in _CODE_MAP_MODES:
        raise ValueError("code_map must be 'auto' or 'off'")
    providers: dict[str, ProviderSettings] = {}
    for name, value in (payload.get("providers") or {}).items():
        if not isinstance(value, dict):
            raise ValueError(f"providers.{name} must be an object")
        entry = ProviderSettings(**{k: v for k, v in value.items() if k in _PROVIDER_FIELDS})
        effective_api(name.lower(), entry)
        providers[name.lower()] = entry
    return Config(
        main_model=payload["main_model"],
        subagent_model=payload["subagent_model"],
        auto_run=bool(payload.get("auto_run", False)),
        providers=providers,
        agent_models=dict(payload.get("agent_models") or {}),
        shell=payload.get("shell", defaults.shell),
        verify_command=payload.get("verify_command", defaults.verify_command),
        code_map=code_map,
        **counts,
    )


def save_config(path: Path | str, config: Config) -> None:
    """Write every config field."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")


def resolve_model(model_id: str) -> str:
    """Normalize a model id to ``provider:model``; a bare id means OpenAI."""

    return normalize_model_id(model_id)


def ensure_config(path: Path | str | None = None, *, input_fn=input, output_fn=print) -> Config:
    """Load the config, or ask for both model ids on first launch."""

    target = Path(path) if path is not None else config_path()
    if target.exists():
        config = load_config(target)
        return replace(
            config,
            main_model=resolve_model(config.main_model),
            subagent_model=resolve_model(config.subagent_model),
            agent_models={k: resolve_model(v) for k, v in config.agent_models.items()},
        )
    output_fn(
        "First launch: enter the model ids for AbsolutCoder "
        "(a bare id means OpenAI; use provider:model for others)."
    )
    main_model = resolve_model(input_fn("Main model id: "))
    subagent_model = resolve_model(input_fn("Subagent model id: "))
    config = Config(main_model=main_model, subagent_model=subagent_model, auto_run=False)
    save_config(target, config)
    return config
