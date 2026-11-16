"""Skill folders: a global one shared by every project, and the project's own.

A skill is a folder holding a ``SKILL.md`` with YAML frontmatter that names it
and describes it; its name is its folder's name. Global skills may also carry
an ``always.md`` whose text is added to every request; project skills never
can, so a cloned repository cannot add text to every request.

Reads stay inside the skills folders: a skill folder, ``SKILL.md`` or
``always.md`` that is a link, or that resolves elsewhere, is ignored.
"""

from __future__ import annotations

import os
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
SKILL_FILE_LIMIT = 1_000_000  # bytes; a larger SKILL.md is not a skill

_NAME_PATTERN = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
_NAME_MAX = 64
_RESERVED_NAMES = frozenset(
    {"con", "prn", "aux", "nul"} | {f"com{i}" for i in range(1, 10)} | {f"lpt{i}" for i in range(1, 10)}
)
_INCOMING = ".skills-incoming"
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


def _is_link(path: Path) -> bool:
    return path.is_symlink() or path.is_junction()


def _plain_child(path: Path, parent: Path) -> bool:
    """``path`` is not a link and resolves to an entry directly inside ``parent``."""

    try:
        return not _is_link(path) and path.resolve().parent == parent.resolve()
    except Exception:
        return False


def _valid_name(name: object) -> bool:
    return (
        isinstance(name, str)
        and len(name) <= _NAME_MAX
        and _NAME_PATTERN.fullmatch(name) is not None
        and name not in _RESERVED_NAMES
    )


def _read_skill_file(folder: Path) -> str | None:
    """The text of ``folder/SKILL.md``, or ``None`` when it is missing, a link, too large or unreadable."""

    path = folder / "SKILL.md"
    try:
        if not _plain_child(path, folder) or not path.is_file():
            return None
        with path.open("rb") as handle:
            raw = handle.read(SKILL_FILE_LIMIT + 1)
        if len(raw) > SKILL_FILE_LIMIT:
            return None
        return raw.decode("utf-8-sig")
    except Exception:
        return None


def _frontmatter(text: str | None) -> dict | None:
    """The frontmatter mapping of a SKILL.md text, or ``None`` when it is not a valid skill file."""

    if text is None:
        return None
    lines = text.splitlines()
    if not lines or lines[0].rstrip() != "---":
        return None
    closing = next((i for i in range(1, len(lines)) if lines[i].rstrip() == "---"), None)
    if closing is None:
        return None
    try:
        data = yaml.safe_load("\n".join(lines[1:closing]))
    except Exception:  # PyYAML also raises ValueError, RecursionError, ... on hostile input
        return None
    if not isinstance(data, dict):
        return None
    for key in ("name", "description"):
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            return None
    return data


def _read_skill(folder: Path, root: Path, scope: str) -> SkillInfo | None:
    if not _plain_child(folder, root) or not _valid_name(folder.name):
        return None
    data = _frontmatter(_read_skill_file(folder))
    if data is None or data["name"] != folder.name:
        return None
    return SkillInfo(name=folder.name, description=data["description"], scope=scope, folder=folder)


def _skills_in(root: Path, scope: str) -> list[SkillInfo]:
    """The valid skills directly inside ``root``, sorted by name."""

    try:
        folders = [p for p in root.iterdir() if p.is_dir()]
    except Exception:
        return []
    found = (_read_skill(folder, root, scope) for folder in folders)
    return sorted((s for s in found if s is not None), key=lambda s: s.name)


def list_skills(repo: Path) -> list[SkillInfo]:
    """Global and project skills by name; a project skill replaces a global one."""

    merged: dict[str, SkillInfo] = {}
    for skill in _skills_in(global_skills_dir(), "global"):
        merged[skill.name] = skill
    for skill in _skills_in(_project_skills_dir(repo), "project"):
        merged[skill.name] = skill
    return [merged[name] for name in sorted(merged)]


def skill_body(info: SkillInfo) -> str:
    """The text of the skill's ``SKILL.md`` after its frontmatter; ``""`` when it cannot be read."""

    text = _read_skill_file(info.folder)
    if text is None or _frontmatter(text) is None:
        return ""
    lines = text.splitlines()
    closing = next(i for i in range(1, len(lines)) if lines[i].rstrip() == "---")
    return "\n".join(lines[closing + 1 :]).strip()


def skill_sources(repo: Path) -> list[tuple[str, str]]:
    """deepagents skill sources, global first so the project wins on a shared name."""

    sources: list[tuple[str, str]] = []
    if global_skills_dir().is_dir():
        sources.append((GLOBAL_ROUTE, "Global"))
    if _project_skills_dir(repo).is_dir():
        sources.append((PROJECT_SOURCE, "Project"))
    return sources


def _read_always(folder: Path) -> str | None:
    """At most ``ALWAYS_LIMIT + 1`` characters of ``folder/always.md`` after leading whitespace."""

    path = folder / "always.md"
    try:
        if not _plain_child(path, folder) or not path.is_file():
            return None
        text = ""
        with path.open(encoding="utf-8-sig") as handle:
            while len(text) <= ALWAYS_LIMIT:
                chunk = handle.read(ALWAYS_LIMIT + 1 - len(text))
                if not chunk:
                    break
                text = text + chunk if text else chunk.lstrip()
        return text
    except Exception:
        return None


def always_text(repo: Path) -> str:
    """Always-on rule blocks from global skills only, in name order."""

    blocks: list[str] = []
    for skill in _skills_in(global_skills_dir(), "global"):
        text = _read_always(skill.folder)
        if text is None:
            continue
        text = text.strip()
        if len(text) > ALWAYS_LIMIT:
            text = text[:ALWAYS_LIMIT] + "\n[truncated]"
        blocks.append(f"## Always-on rules from the {skill.folder.name} skill\n{text}")
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


def _check_name(name: object) -> None:
    if not isinstance(name, str) or len(name) > _NAME_MAX or _NAME_PATTERN.fullmatch(name) is None:
        raise ValueError(
            f"invalid skill name {name!r}: use lowercase letters, digits and single hyphens, "
            f"at most {_NAME_MAX} characters"
        )
    if name in _RESERVED_NAMES:
        raise ValueError("that name is reserved on Windows")


def _contains(outer: Path, inner: Path) -> bool:
    return inner == outer or outer in inner.parents


def _ignore(directory: str, names: list[str]) -> set[str]:
    base = Path(directory)
    skipped = set()
    for name in names:
        path = base / name
        if _is_link(path):
            skipped.add(name)
        elif path.is_dir():
            if name in _SKIPPED_FOLDERS:
                skipped.add(name)
        elif any(fnmatch(name, pattern) for pattern in _SKIPPED_FILES):
            skipped.add(name)
    return skipped


def install_skill(source: Path) -> SkillInfo:
    """Copy the skill folder ``source`` into the global skills folder, replacing a copy.

    The copy is made in a staging folder outside the skills folder and swapped
    in only when complete; a failed install leaves the previous copy in place.
    """

    source = Path(source)
    if not source.is_dir():
        raise ValueError(f"{source} is not a folder")
    data = _frontmatter(_read_skill_file(source))
    if data is None:
        raise ValueError(
            f"{source} has no valid SKILL.md (frontmatter between --- lines with a name and a description)"
        )
    name = data["name"]
    _check_name(name)
    destination = global_skills_dir() / name
    incoming = data_dir() / _INCOMING
    real_source = source.resolve()
    for target in (destination.resolve(), incoming.resolve()):
        if _contains(real_source, target) or _contains(target, real_source):
            raise ValueError(f"cannot install {source} onto itself ({destination})")
    staging = incoming / name
    retired = incoming / f"{name}.old"
    try:
        for leftover in (staging, retired):
            if leftover.exists():
                shutil.rmtree(leftover)
        shutil.copytree(source, staging, ignore=_ignore)
        destination.parent.mkdir(parents=True, exist_ok=True)
        had_old = destination.exists()
        if had_old:
            os.replace(destination, retired)
        try:
            os.replace(staging, destination)
        except BaseException:
            if had_old:
                os.replace(retired, destination)
            raise
    finally:
        shutil.rmtree(incoming, ignore_errors=True)
    return SkillInfo(name=name, description=data["description"], scope="global", folder=destination)


def remove_skill(name: str) -> bool:
    """Delete the global skill folder ``name``; return whether it existed."""

    _check_name(name)
    target = global_skills_dir() / name
    if target.is_symlink():
        target.unlink()
        return True
    if target.is_junction():
        os.rmdir(target)
        return True
    if not target.is_dir():
        return False
    shutil.rmtree(target)
    return True
