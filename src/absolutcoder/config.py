"""Load and save AbsolutCoder settings, and resolve OpenAI model ids."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

OPENAI_PROVIDER = "openai"


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


def load_config(path: Path | str) -> Config:
    """Read a config file. Model strings are returned as stored."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return Config(
        main_model=payload["main_model"],
        subagent_model=payload["subagent_model"],
        auto_run=bool(payload.get("auto_run", False)),
    )


def save_config(path: Path | str, config: Config) -> None:
    """Write both model fields and auto_run."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")


def resolve_model(model_id: str) -> str:
    """Normalize a model id to ``openai:<id>``.

    A value with no provider prefix is treated as an OpenAI model. Any other
    provider prefix is rejected. This version does not call other providers.
    """

    text = model_id.strip()
    if not text:
        raise ValueError("model id is empty")
    if ":" in text:
        provider, _, rest = text.partition(":")
        if provider.lower() != OPENAI_PROVIDER or not rest.strip():
            raise ValueError(
                f"Only OpenAI models are supported, got provider {provider!r}"
            )
        return f"{OPENAI_PROVIDER}:{rest.strip()}"
    return f"{OPENAI_PROVIDER}:{text}"


def ensure_config(path: Path | str | None = None, *, input_fn=input, output_fn=print) -> Config:
    """Load the config, or ask for both model ids on first launch."""

    target = Path(path) if path is not None else config_path()
    if target.exists():
        config = load_config(target)
        return Config(
            main_model=resolve_model(config.main_model),
            subagent_model=resolve_model(config.subagent_model),
            auto_run=config.auto_run,
        )
    output_fn("First launch: enter the OpenAI model ids for AbsolutCoder.")
    main_model = resolve_model(input_fn("Main model id: "))
    subagent_model = resolve_model(input_fn("Subagent model id: "))
    config = Config(main_model=main_model, subagent_model=subagent_model, auto_run=False)
    save_config(target, config)
    return config
