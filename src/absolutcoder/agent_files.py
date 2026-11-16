"""Worker roles defined as markdown files with YAML frontmatter."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from absolutcoder.config import data_dir
from absolutcoder.models import normalize_model_id
from absolutcoder.subagents import AgentDef, builtin_agents, merge_agents, validate_groups

_NAME_RE = re.compile(r"^[a-z][a-z0-9-]{1,39}$")
_KNOWN_KEYS = ("name", "description", "tools", "model")
_DEFAULT_TOOLS = ("read", "search")
_MAX_DESCRIPTION = 300


class AgentFileError(ValueError):
    """A worker role file that cannot be used."""


def _split_frontmatter(text: str) -> tuple[str, str]:
    lines = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if lines[0] != "---":
        raise AgentFileError("missing frontmatter: the file must start with a --- line")
    for index in range(1, len(lines)):
        if lines[index] == "---":
            return "\n".join(lines[1:index]), "\n".join(lines[index + 1 :])
    raise AgentFileError("unclosed frontmatter: no closing --- line")


def _load_meta(path: Path) -> tuple[dict, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise AgentFileError(f"cannot read file: {error}") from error
    header, body = _split_frontmatter(text)
    try:
        meta = yaml.safe_load(header)
    except yaml.YAMLError as error:
        raise AgentFileError(f"invalid YAML in frontmatter: {error}") from error
    if not isinstance(meta, dict):
        raise AgentFileError("frontmatter must be a YAML mapping")
    return meta, body


def _parse_tools(raw) -> tuple[str, ...]:
    if raw is None:
        return validate_groups(_DEFAULT_TOOLS)
    if isinstance(raw, str):
        items = [part.strip() for part in raw.split(",") if part.strip()]
    elif isinstance(raw, list) and all(isinstance(item, str) for item in raw):
        items = [item.strip() for item in raw]
    else:
        raise AgentFileError("tools must be a list or a comma-separated string of groups")
    try:
        return validate_groups(items)
    except ValueError as error:
        raise AgentFileError(f"invalid tools: {error}") from error


def _build(path: Path, source: str) -> tuple[AgentDef, dict]:
    meta, body = _load_meta(path)
    name = meta.get("name", path.stem)
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise AgentFileError(f"invalid name {name!r}: use {_NAME_RE.pattern}")
    description = meta.get("description")
    if not isinstance(description, str) or not description.strip():
        raise AgentFileError("description is missing")
    description = description.strip()
    if len(description) > _MAX_DESCRIPTION:
        raise AgentFileError(
            f"description is too long ({len(description)} characters, at most {_MAX_DESCRIPTION})"
        )
    groups = _parse_tools(meta.get("tools"))
    model = None
    if meta.get("model") is not None:
        try:
            model = normalize_model_id(str(meta["model"]))
        except ValueError as error:
            raise AgentFileError(f"malformed model: {error}") from error
    prompt = body.strip()
    if not prompt:
        raise AgentFileError("the system prompt after the frontmatter is empty")
    definition = AgentDef(
        name=name,
        description=description,
        prompt=prompt,
        groups=groups,
        model=model,
        source=source,
    )
    return definition, meta


def parse_agent_file(path: Path, source: str = "repo") -> AgentDef:
    """Parse one role file; every problem raises AgentFileError."""

    return _build(path, source)[0]


def load_agent_dir(directory: Path, source: str) -> tuple[list[AgentDef], list[str]]:
    """Load every ``*.md`` file in ``directory``; bad files become warnings."""

    if not directory.is_dir():
        return [], []
    definitions: list[AgentDef] = []
    warnings: list[str] = []
    seen: set[str] = set()
    for path in sorted(directory.glob("*.md")):
        if not path.is_file():
            continue
        try:
            definition, meta = _build(path, source)
        except AgentFileError as error:
            warnings.append(f"{path.name}: {error}")
            continue
        if definition.name in seen:
            warnings.append(f"{path.name}: duplicate worker name {definition.name}")
            continue
        seen.add(definition.name)
        warnings.extend(
            f"{path.name}: unknown key {key} ignored" for key in meta if key not in _KNOWN_KEYS
        )
        definitions.append(definition)
    return definitions, warnings


def load_roster(repo: Path) -> tuple[list[AgentDef], list[str]]:
    """Built-in roles, then home files, then repo files, merged by name."""

    home, home_warnings = load_agent_dir(data_dir() / "agents", "home")
    mine, repo_warnings = load_agent_dir(Path(repo) / ".absolutcoder" / "agents", "repo")
    roster = merge_agents(builtin_agents(), home, mine)
    return roster, home_warnings + repo_warnings
