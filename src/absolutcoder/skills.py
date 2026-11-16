"""Skill folders: a global one shared by every project, and the project's own.

A skill is a folder holding a ``SKILL.md`` with YAML frontmatter that names it
and describes it. Global skills may also carry an ``always.md`` whose text is
added to every request; project skills never can, so a cloned repository
cannot add text to every request.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

import yaml

from absolutcoder.config import data_dir

GLOBAL_ROUTE = "/.absolutcoder-global/skills/"
PROJECT_SOURCE = "/.absolutcoder/skills/"
ALWAYS_LIMIT = 4000

_NAME_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_NAME_MAX = 64
_SKIPPED_FOLDERS = frozenset({"fixtures", "node_modules", ".hallmark", ".git", "__pycache__"})
_SKIPPED_FILES = ("*.test.mjs", "*.test.js", "*.test.cjs")


@dataclass(frozen=True)
class SkillInfo:
    name: str
    description: str
    scope: str  # "global" | "project"
    folder: Path


def global_skills_dir() -> Path:
    """The skills folder shared by every project."""

    return data_dir() / "skills"


def _project_skills_dir(repo: Path) -> Path:
    return Path(repo) / ".absolutcoder" / "skills"


def _frontmatter(skill_md: Path) -> dict | None:
    """The frontmatter mapping of ``skill_md``, or ``None`` when it is not a valid skill file."""

    try:
        lines = skill_md.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return None
    if not lines or lines[0].rstrip() != "---":
        return None
    closing = next((i for i in range(1, len(lines)) if lines[i].rstrip() == "---"), None)
    if closing is None:
        return None
    try:
        data = yaml.safe_load("\n".join(lines[1:closing]))
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict):
        return None
    for key in ("name", "description"):
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            return None
    return data


def _read_skill(folder: Path, scope: str) -> SkillInfo | None:
    data = _frontmatter(folder / "SKILL.md")
    if data is None:
        return None
    return SkillInfo(name=data["name"], description=data["description"], scope=scope, folder=folder)


def _skills_in(root: Path, scope: str) -> list[SkillInfo]:
    """The valid skills directly inside ``root``, sorted by name."""

    try:
        folders = [p for p in root.iterdir() if p.is_dir()]
    except OSError:
        return []
    found = (_read_skill(folder, scope) for folder in folders)
    return sorted((s for s in found if s is not None), key=lambda s: s.name)


def list_skills(repo: Path) -> list[SkillInfo]:
    """Global and project skills by name; a project skill replaces a global one."""

    merged: dict[str, SkillInfo] = {}
    for skill in _skills_in(global_skills_dir(), "global"):
        merged[skill.name] = skill
    for skill in _skills_in(_project_skills_dir(repo), "project"):
        merged[skill.name] = skill
    return [merged[name] for name in sorted(merged)]


def skill_sources(repo: Path) -> list[tuple[str, str]]:
    """deepagents skill sources, global first so the project wins on a shared name."""

    sources: list[tuple[str, str]] = []
    if global_skills_dir().is_dir():
        sources.append((GLOBAL_ROUTE, "Global"))
    if _project_skills_dir(repo).is_dir():
        sources.append((PROJECT_SOURCE, "Project"))
    return sources


def always_text(repo: Path) -> str:
    """Always-on rule blocks from global skills only, in name order."""

    blocks: list[str] = []
    for skill in _skills_in(global_skills_dir(), "global"):
        try:
            text = (skill.folder / "always.md").read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            continue
        if len(text) > ALWAYS_LIMIT:
            text = text[:ALWAYS_LIMIT] + "\n[truncated]"
        blocks.append(f"## Always-on rules from the {skill.name} skill\n{text}")
    return "\n\n".join(blocks)


def skill_paths_note(repo: Path) -> str:
    """Where the skill folders are on disk, so a skill's scripts run with real paths."""

    parts: list[str] = []
    for path, _label in skill_sources(repo):
        if path == GLOBAL_ROUTE:
            parts.append(f"global skills are in {global_skills_dir().as_posix()}/<name>")
        else:
            parts.append(f"project skills are in {_project_skills_dir(repo).as_posix()}/<name>")
    if not parts:
        return ""
    return (
        "Skill folders on disk: "
        + "; ".join(parts)
        + ". Where a skill's text says <skill-dir>, use that folder, and run a skill's scripts with those real paths."
    )


def _check_name(name: str) -> None:
    if len(name) > _NAME_MAX or not _NAME_PATTERN.match(name):
        raise ValueError(
            f"invalid skill name {name!r}: use lowercase letters, digits and single hyphens, "
            f"at most {_NAME_MAX} characters"
        )


def _contains(outer: Path, inner: Path) -> bool:
    return inner == outer or outer in inner.parents


def _ignore(directory: str, names: list[str]) -> set[str]:
    base = Path(directory)
    skipped = set()
    for name in names:
        path = base / name
        if path.is_symlink():
            skipped.add(name)
        elif path.is_dir():
            if name in _SKIPPED_FOLDERS:
                skipped.add(name)
        elif any(fnmatch(name, pattern) for pattern in _SKIPPED_FILES):
            skipped.add(name)
    return skipped


def install_skill(source: Path) -> SkillInfo:
    """Copy the skill folder ``source`` into the global skills folder, replacing a copy."""

    source = Path(source)
    if not source.is_dir():
        raise ValueError(f"{source} is not a folder")
    data = _frontmatter(source / "SKILL.md")
    if data is None:
        raise ValueError(
            f"{source} has no valid SKILL.md (frontmatter between --- lines with a name and a description)"
        )
    name = data["name"]
    _check_name(name)
    destination = global_skills_dir() / name
    real_source = source.resolve()
    real_destination = destination.resolve()
    if _contains(real_source, real_destination) or _contains(real_destination, real_source):
        raise ValueError(f"cannot install {source} onto itself ({destination})")
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, ignore=_ignore)
    return SkillInfo(name=name, description=data["description"], scope="global", folder=destination)


def remove_skill(name: str) -> bool:
    """Delete the global skill folder ``name``; return whether it existed."""

    _check_name(name)
    target = global_skills_dir() / name
    if not target.is_dir():
        return False
    shutil.rmtree(target)
    return True
