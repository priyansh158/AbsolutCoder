"""Load and save AbsolutCoder settings, and resolve provider model ids."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

from absolutcoder.models import ProviderSettings, normalize_model_id


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


_PROVIDER_FIELDS = {f.name for f in fields(ProviderSettings)}


def load_config(path: Path | str) -> Config:
    """Read a config file. Model strings are returned as stored."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    defaults = Config(main_model="", subagent_model="")
    providers = {
        name: ProviderSettings(
            **{k: v for k, v in (value or {}).items() if k in _PROVIDER_FIELDS}
        )
        for name, value in (payload.get("providers") or {}).items()
    }
    return Config(
        main_model=payload["main_model"],
        subagent_model=payload["subagent_model"],
        auto_run=bool(payload.get("auto_run", False)),
        providers=providers,
        agent_models=dict(payload.get("agent_models") or {}),
        shell=payload.get("shell", defaults.shell),
        verify_command=payload.get("verify_command", defaults.verify_command),
        max_rounds=payload.get("max_rounds", defaults.max_rounds),
        web_max_chars=payload.get("web_max_chars", defaults.web_max_chars),
        max_background_tasks=payload.get(
            "max_background_tasks", defaults.max_background_tasks
        ),
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
