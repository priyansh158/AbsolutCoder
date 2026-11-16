"""Worker roles loaded from markdown files."""

import pytest

from absolutcoder.agent_files import AgentFileError, load_agent_dir, load_roster, parse_agent_file
from absolutcoder.config import data_dir
from absolutcoder.subagents import AgentDef


def _write(directory, name, text):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(text, encoding="utf-8", newline="")
    return path


def _file(description="Reviews code.", extra="", prompt="You review."):
    return f"---\ndescription: {description}\n{extra}---\n{prompt}\n"


def test_parse_valid_file(tmp_path):
    path = _write(
        tmp_path,
        "reviewer.md",
        "---\ndescription: Reviews diffs.\ntools: [search, read]\nmodel: Anthropic:claude-x\n---\n\n  Be strict.  \n\n",
    )
    assert parse_agent_file(path) == AgentDef(
        name="reviewer",
        description="Reviews diffs.",
        prompt="Be strict.",
        groups=("read", "search"),
        model="anthropic:claude-x",
        source="repo",
    )


def test_name_defaults_to_file_stem_and_tools_default(tmp_path):
    definition = parse_agent_file(_write(tmp_path, "auditor.md", _file()), "home")
    assert definition.name == "auditor"
    assert definition.groups == ("read", "search")
    assert definition.model is None
    assert definition.source == "home"


def test_tools_accepts_comma_string(tmp_path):
    path = _write(tmp_path, "editor.md", _file(extra="tools: read, search, edit\n"))
    assert parse_agent_file(path).groups == ("read", "search", "edit")


def test_web_with_shell_is_rejected(tmp_path):
    path = _write(tmp_path, "bad.md", _file(extra="tools: [web, shell]\n"))
    with pytest.raises(AgentFileError):
        parse_agent_file(path)


@pytest.mark.parametrize(
    "text",
    [
        "just text\n",
        "---\ndescription: x\nprompt without close\n",
        "---\n- a\n- b\n---\nprompt\n",
        "---\nname: okay\n---\nprompt\n",
        f"---\ndescription: {'d' * 301}\n---\nprompt\n",
        "---\nname: Bad_Name\ndescription: x\n---\nprompt\n",
        "---\ndescription: x\ntools: [fly]\n---\nprompt\n",
        '---\ndescription: x\nmodel: "a b:x"\n---\nprompt\n',
        "---\ndescription: x\n---\n  \n",
    ],
)
def test_invalid_files_raise(tmp_path, text):
    with pytest.raises(AgentFileError):
        parse_agent_file(_write(tmp_path, "case.md", text))


def test_bad_file_is_skipped_with_warning(tmp_path):
    _write(tmp_path, "good.md", _file())
    _write(tmp_path, "broken.md", "no frontmatter")
    definitions, warnings = load_agent_dir(tmp_path, "repo")
    assert [d.name for d in definitions] == ["good"]
    assert len(warnings) == 1 and warnings[0].startswith("broken.md: ")


def test_unknown_key_warns_but_loads(tmp_path):
    _write(tmp_path, "odd.md", _file(extra="color: red\n"))
    definitions, warnings = load_agent_dir(tmp_path, "repo")
    assert [d.name for d in definitions] == ["odd"]
    assert warnings == ["odd.md: unknown key color ignored"]


def test_duplicate_name_in_directory_is_skipped(tmp_path):
    _write(tmp_path, "a.md", _file(extra="name: twin\n", prompt="first"))
    _write(tmp_path, "b.md", _file(extra="name: twin\n", prompt="second"))
    definitions, warnings = load_agent_dir(tmp_path, "repo")
    assert [d.prompt for d in definitions] == ["first"]
    assert warnings == ["b.md: duplicate worker name twin"]


def test_missing_directory_is_empty(tmp_path):
    assert load_agent_dir(tmp_path / "nope", "repo") == ([], [])


def test_repo_overrides_home(tmp_path):
    repo = tmp_path / "repo"
    _write(data_dir() / "agents", "explore.md", _file(prompt="home prompt"))
    _write(data_dir() / "agents", "helper.md", _file(prompt="helper prompt"))
    _write(repo / ".absolutcoder" / "agents", "explore.md", _file(prompt="repo prompt"))
    roster, warnings = load_roster(repo)
    assert warnings == []
    names = [d.name for d in roster]
    assert names == ["explore", "plan", "general-purpose", "web-researcher", "helper"]
    explore = roster[0]
    assert explore.source == "repo" and explore.prompt == "repo prompt"
    assert roster[-1].source == "home"


def test_bom_and_crlf_are_tolerated(tmp_path):
    path = tmp_path / "crlf.md"
    path.write_bytes(
        "﻿---\r\ndescription: Windows file.\r\ntools: [read]\r\n---\r\nLine one.\r\nLine two.\r\n".encode(
            "utf-8"
        )
    )
    definition = parse_agent_file(path)
    assert definition.description == "Windows file."
    assert definition.groups == ("read",)
    assert definition.prompt == "Line one.\nLine two."
