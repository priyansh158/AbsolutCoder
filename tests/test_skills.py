"""Skill folders: global and project discovery, always-on rules, install and remove."""

import pytest

from absolutcoder.skills import (
    ALWAYS_LIMIT,
    GLOBAL_ROUTE,
    PROJECT_SOURCE,
    always_text,
    global_skills_dir,
    install_skill,
    list_skills,
    remove_skill,
    skill_paths_note,
    skill_sources,
)


def _skill(folder, name, description="Does things.", body="# Body\n"):
    target = folder / name
    target.mkdir(parents=True, exist_ok=True)
    (target / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n{body}", encoding="utf-8"
    )
    return target


def _project_dir(repo):
    return repo / ".absolutcoder" / "skills"


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "repo"
    path.mkdir()
    return path


def test_list_skills_scopes_and_override(repo):
    _skill(global_skills_dir(), "alpha", "Global alpha.")
    _skill(global_skills_dir(), "beta", "Global beta.")
    _skill(_project_dir(repo), "beta", "Project beta.")
    skills = list_skills(repo)
    assert [(s.name, s.scope, s.description) for s in skills] == [
        ("alpha", "global", "Global alpha."),
        ("beta", "project", "Project beta."),
    ]
    assert skills[1].folder == _project_dir(repo) / "beta"


def test_malformed_skills_are_skipped(repo):
    root = global_skills_dir()
    cases = {
        "nofront": "# no frontmatter\n",
        "unclosed": "---\nname: unclosed\ndescription: x\n",
        "listfront": "---\n- a\n- b\n---\nbody\n",
        "nodesc": "---\nname: nodesc\n---\nbody\n",
        "badyaml": "---\nname: [unclosed\n---\n",
    }
    for name, text in cases.items():
        (root / name).mkdir(parents=True)
        (root / name / "SKILL.md").write_text(text, encoding="utf-8")
    (root / "binary").mkdir()
    (root / "binary" / "SKILL.md").write_bytes(b"---\nname: binary\ndescription: \xff\xfe\n---\n")
    (root / "binary" / "always.md").write_text("binary rules", encoding="utf-8")
    (root / "loose.md").write_text("not a folder", encoding="utf-8")
    _skill(root, "good")
    assert [s.name for s in list_skills(repo)] == ["good"]
    assert "binary rules" not in always_text(repo)


def test_skill_sources_order_and_absence(repo):
    assert skill_sources(repo) == []
    _project_dir(repo).mkdir(parents=True)
    assert skill_sources(repo) == [(PROJECT_SOURCE, "Project")]
    global_skills_dir().mkdir(parents=True)
    assert skill_sources(repo) == [(GLOBAL_ROUTE, "Global"), (PROJECT_SOURCE, "Project")]
    _project_dir(repo).rmdir()
    assert skill_sources(repo) == [(GLOBAL_ROUTE, "Global")]


def test_always_text_is_capped_and_project_always_is_ignored(repo):
    big = _skill(global_skills_dir(), "design")
    text = "x" * 4000 + "y" * 1000
    (big / "always.md").write_text("\n" + text + "\n\n", encoding="utf-8")
    small = _skill(global_skills_dir(), "alpha")
    (small / "always.md").write_text("Use short names.\n", encoding="utf-8")
    _skill(global_skills_dir(), "plain")
    project = _skill(_project_dir(repo), "local")
    (project / "always.md").write_text("PROJECT RULE INJECTED", encoding="utf-8")

    result = always_text(repo)
    assert result == (
        "## Always-on rules from the alpha skill\nUse short names.\n\n"
        "## Always-on rules from the design skill\n" + "x" * ALWAYS_LIMIT + "\n[truncated]"
    )
    assert "PROJECT RULE INJECTED" not in result
    assert ALWAYS_LIMIT == 4000


def test_always_text_is_empty_without_global_rules(repo):
    assert always_text(repo) == ""
    _skill(global_skills_dir(), "plain")
    assert always_text(repo) == ""


def test_always_text_survives_a_project_override(repo):
    shared = _skill(global_skills_dir(), "shared", "Global version.")
    (shared / "always.md").write_text("Global rule.", encoding="utf-8")
    _skill(_project_dir(repo), "shared", "Project version.")
    assert [s.scope for s in list_skills(repo)] == ["project"]
    assert always_text(repo) == "## Always-on rules from the shared skill\nGlobal rule."


def test_skill_dir_note_names_real_folders(repo):
    assert skill_paths_note(repo) == ""
    global_skills_dir().mkdir(parents=True)
    _project_dir(repo).mkdir(parents=True)
    note = skill_paths_note(repo)
    global_dir = global_skills_dir().as_posix()
    project_dir = repo.as_posix() + "/.absolutcoder/skills"
    assert note == (
        f"Skill folders on disk: global skills are in {global_dir}/<name>; "
        f"project skills are in {project_dir}/<name>. Where a skill's text says <skill-dir>, "
        "use that folder, and run a skill's scripts with those real paths."
    )
    assert "\\" not in note
    _project_dir(repo).rmdir()
    assert skill_paths_note(repo) == (
        f"Skill folders on disk: global skills are in {global_dir}/<name>. Where a skill's "
        "text says <skill-dir>, use that folder, and run a skill's scripts with those real paths."
    )


def test_install_skill_copies_and_validates(tmp_path):
    source = _skill(tmp_path / "src", "demo", "A demo.")
    files = {
        "references/a.md": "ref",
        "scripts/run.mjs": "run",
        "scripts/run.test.mjs": "test",
        "scripts/run.test.js": "test",
        "scripts/run.test.cjs": "test",
        "fixtures/x.txt": "fixture",
        "node_modules/y.js": "dep",
        ".hallmark/z": "state",
        ".git/HEAD": "ref",
        "__pycache__/m.pyc": "cache",
    }
    for rel, text in files.items():
        (source / rel).parent.mkdir(parents=True, exist_ok=True)
        (source / rel).write_text(text, encoding="utf-8")

    info = install_skill(source)
    target = global_skills_dir() / "demo"
    assert (info.name, info.description, info.scope, info.folder) == ("demo", "A demo.", "global", target)
    copied = sorted(p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file())
    assert copied == ["SKILL.md", "references/a.md", "scripts/run.mjs"]

    (target / "stale.txt").write_text("old", encoding="utf-8")
    install_skill(source)
    assert not (target / "stale.txt").exists()
    assert (target / "SKILL.md").is_file()

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="SKILL.md"):
        install_skill(empty)
    bad = _skill(tmp_path / "bad", "Bad_Name")
    with pytest.raises(ValueError, match="name"):
        install_skill(bad)
    with pytest.raises(ValueError):
        install_skill(global_skills_dir())
    with pytest.raises(ValueError):
        install_skill(target)
    long_name = _skill(tmp_path / "long", "a" * 65)
    with pytest.raises(ValueError, match="name"):
        install_skill(long_name)


def test_remove_skill(tmp_path):
    install_skill(_skill(tmp_path / "src", "demo"))
    assert remove_skill("demo") is True
    assert not (global_skills_dir() / "demo").exists()
    assert remove_skill("demo") is False
    with pytest.raises(ValueError):
        remove_skill("../x")
