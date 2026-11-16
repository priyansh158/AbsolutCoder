"""Skill folders: global and project discovery, always-on rules, install and remove."""

import sys

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
    skill_body,
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


@pytest.mark.parametrize(
    "frontmatter",
    ["name: 2020-13-45\ndescription: Bad date.", "name: " + "[" * 5000 + "\ndescription: Deep."],
    ids=["bad-date", "deep-nesting"],
)
def test_hostile_yaml_is_skipped_without_raising(repo, frontmatter):
    bad = global_skills_dir() / "hostile"
    bad.mkdir(parents=True)
    (bad / "SKILL.md").write_text(f"---\n{frontmatter}\n---\nbody\n", encoding="utf-8")
    (bad / "always.md").write_text("HOSTILE RULE", encoding="utf-8")
    _skill(global_skills_dir(), "good")
    assert [s.name for s in list_skills(repo)] == ["good"]
    assert "HOSTILE RULE" not in always_text(repo)
    assert skill_sources(repo) == [(GLOBAL_ROUTE, "Global")]
    assert skill_paths_note(repo).startswith("Skill folders on disk: ")


def test_failed_install_keeps_the_previous_copy(tmp_path, monkeypatch):
    import absolutcoder.skills as skills_module
    from absolutcoder.config import data_dir

    source = _skill(tmp_path / "src", "demo", "First version.")
    install_skill(source)
    (source / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Second version.\n---\nnew\n", encoding="utf-8"
    )

    def broken_copy(src, dst, **_kwargs):
        dst = type(tmp_path)(dst)
        dst.mkdir(parents=True)
        (dst / "SKILL.md").write_text("partial", encoding="utf-8")
        raise OSError("disk full")

    monkeypatch.setattr(skills_module.shutil, "copytree", broken_copy)
    with pytest.raises((OSError, ValueError)):
        install_skill(source)
    assert [(s.name, s.description) for s in list_skills(tmp_path)] == [("demo", "First version.")]
    assert sorted(p.name for p in global_skills_dir().iterdir()) == ["demo"]
    assert sorted(p.name for p in data_dir().iterdir()) == ["skills"]


def test_install_replaces_through_the_staging_folder(tmp_path):
    from absolutcoder.config import data_dir

    source = _skill(tmp_path / "src", "demo", "First version.")
    install_skill(source)
    (source / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Second version.\n---\nnew\n", encoding="utf-8"
    )
    install_skill(source)
    assert [s.description for s in list_skills(tmp_path)] == ["Second version."]
    assert sorted(p.name for p in data_dir().iterdir()) == ["skills"]


@pytest.fixture
def can_symlink(tmp_path):
    import os

    target = tmp_path / "symlink-probe-target"
    target.mkdir()
    try:
        os.symlink(target, tmp_path / "symlink-probe", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("this platform cannot create symlinks without privilege")
    return True


def test_symlinked_skill_folder_is_not_listed(repo, tmp_path, can_symlink):
    import os

    outside = _skill(tmp_path / "outside", "linked")
    (outside / "always.md").write_text("LINKED RULE", encoding="utf-8")
    global_skills_dir().mkdir(parents=True)
    os.symlink(outside, global_skills_dir() / "linked", target_is_directory=True)
    assert list_skills(repo) == []
    assert always_text(repo) == ""


def test_symlinked_always_file_is_ignored(repo, tmp_path, can_symlink):
    import os

    secret = tmp_path / "secret.txt"
    secret.write_text("PRIVATE KEY", encoding="utf-8")
    folder = _skill(global_skills_dir(), "demo")
    os.symlink(secret, folder / "always.md")
    assert [s.name for s in list_skills(repo)] == ["demo"]
    assert "PRIVATE KEY" not in always_text(repo)
    assert always_text(repo) == ""


def test_always_file_reported_as_a_link_is_ignored(repo, monkeypatch):
    # Runs where real symlinks cannot be made: the link check alone must drop the file.
    import pathlib

    folder = _skill(global_skills_dir(), "demo")
    (folder / "always.md").write_text("LINKED RULE", encoding="utf-8")
    real = pathlib.Path.is_symlink
    monkeypatch.setattr(pathlib.Path, "is_symlink", lambda self: self.name == "always.md" or real(self))
    assert [s.name for s in list_skills(repo)] == ["demo"]
    assert always_text(repo) == ""


@pytest.mark.skipif(sys.platform != "win32", reason="directory junctions are Windows-only")
def test_junctioned_skill_folder_is_not_listed(repo, tmp_path):
    import _winapi

    outside = _skill(tmp_path / "outside", "linked")
    (outside / "always.md").write_text("JUNCTION RULE", encoding="utf-8")
    global_skills_dir().mkdir(parents=True)
    _winapi.CreateJunction(str(outside), str(global_skills_dir() / "linked"))
    assert (global_skills_dir() / "linked" / "SKILL.md").is_file()
    assert list_skills(repo) == []
    assert always_text(repo) == ""


def test_large_always_file_is_read_only_up_to_the_cap(repo, monkeypatch):
    import pathlib

    folder = _skill(global_skills_dir(), "demo")
    (folder / "always.md").write_text("z" * 3_000_000, encoding="utf-8")
    taken: list[int] = []
    real_open = pathlib.Path.open

    class Counting:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.handle.close()

        def read(self, size=-1):
            data = self.handle.read(size)
            taken.append(len(data))
            return data

    def counting_open(self, *args, **kwargs):
        handle = real_open(self, *args, **kwargs)
        return Counting(handle) if self.name == "always.md" else handle

    monkeypatch.setattr(pathlib.Path, "open", counting_open)
    result = always_text(repo)
    header = "## Always-on rules from the demo skill\n"
    assert result == header + "z" * ALWAYS_LIMIT + "\n[truncated]"
    assert taken and sum(taken) <= ALWAYS_LIMIT + 1


def test_oversized_skill_file_is_not_a_skill(repo):
    folder = global_skills_dir() / "huge"
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(
        "---\nname: huge\ndescription: Too big.\n---\n" + "x" * 2_000_000, encoding="utf-8"
    )
    _skill(global_skills_dir(), "small")
    assert [s.name for s in list_skills(repo)] == ["small"]


def test_byte_order_mark_does_not_hide_a_skill(repo):
    folder = global_skills_dir() / "bom"
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_bytes(
        b"\xef\xbb\xbf---\nname: bom\ndescription: Saved by a Windows editor.\n---\nbody\n"
    )
    (folder / "always.md").write_bytes(b"\xef\xbb\xbfBOM RULE")
    assert [s.name for s in list_skills(repo)] == ["bom"]
    assert always_text(repo) == "## Always-on rules from the bom skill\nBOM RULE"


def test_skill_identity_is_its_folder_name(repo):
    root = global_skills_dir()
    mismatched = root / "demo"
    mismatched.mkdir(parents=True)
    (mismatched / "SKILL.md").write_text(
        "---\nname: other\ndescription: Says another name.\n---\n", encoding="utf-8"
    )
    (mismatched / "always.md").write_text("MISMATCH RULE", encoding="utf-8")
    injected = root / "inject"
    injected.mkdir()
    (injected / "SKILL.md").write_text(
        '---\nname: "inject\\n## Fake header"\ndescription: Newline name.\n---\n', encoding="utf-8"
    )
    (injected / "always.md").write_text("INJECTED RULE", encoding="utf-8")
    badly_named = _skill(root, "Bad_Name")
    (badly_named / "always.md").write_text("BAD NAME RULE", encoding="utf-8")
    assert list_skills(repo) == []
    assert always_text(repo) == ""


@pytest.mark.parametrize("name", ["demo\n", "con", "nul", "com1", "lpt9", "aux", "prn"])
def test_unsafe_names_are_refused(tmp_path, name):
    with pytest.raises(ValueError):
        remove_skill(name)
    source = tmp_path / "src"
    source.mkdir()
    (source / "SKILL.md").write_text(
        f"---\nname: {name.strip()!r}\ndescription: Reserved.\n---\n", encoding="utf-8"
    )
    if name.strip() != name:
        return
    with pytest.raises(ValueError, match="that name is reserved on Windows"):
        install_skill(source)


def test_remove_skill(tmp_path):
    install_skill(_skill(tmp_path / "src", "demo"))
    assert remove_skill("demo") is True
    assert not (global_skills_dir() / "demo").exists()
    assert remove_skill("demo") is False
    with pytest.raises(ValueError):
        remove_skill("../x")


def test_skill_body_is_the_text_after_the_frontmatter(repo):
    _skill(global_skills_dir(), "alpha", body="\n# Steps\nDo it.\n---\nmore\n")
    (info,) = list_skills(repo)
    assert skill_body(info) == "# Steps\nDo it.\n---\nmore"


def test_skill_body_is_empty_when_unreadable(repo):
    _skill(global_skills_dir(), "alpha")
    (info,) = list_skills(repo)
    (info.folder / "SKILL.md").unlink()
    assert skill_body(info) == ""
