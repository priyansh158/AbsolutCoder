"""Shell guard, commit scrubbing, and the untracked memory file."""

import os
import shlex
import subprocess
from pathlib import Path

import pytest

from absolutcoder.git_policy import (
    PolicyDenied,
    TurnFlags,
    execute_needs_approval,
    protect_context_file,
    render_command,
    scrub_commit_message,
    sets_commit_identity,
    unwrap_shell,
    vet,
)
from absolutcoder.shell import GuardedShellBackend


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def test_push_and_pull_require_flags(tmp_path, git_repo):
    repo = git_repo(tmp_path / "repo")
    backend = GuardedShellBackend(root_dir=repo)
    denied = TurnFlags()
    for command in (
        "git push",
        "git push --force",
        "git push -f",
        "git pull",
        "git fetch origin",
        "git status && git push",
        "git status && git pull",
    ):
        with pytest.raises(PolicyDenied):
            vet(command, denied)
        with pytest.raises(PolicyDenied):
            backend.execute(command)
    vet("git push", TurnFlags(push_requested=True))
    vet("git push --force", TurnFlags(push_requested=True))
    vet("git pull", TurnFlags(pull_requested=True))
    vet("git fetch", TurnFlags(pull_requested=True))
    assert not (repo / "should-not-exist").exists()


def test_git_config_reset_and_clean_require_the_destructive_flag(tmp_path, git_repo):
    repo = git_repo(tmp_path / "repo")
    backend = GuardedShellBackend(root_dir=repo)
    for command in ("git config user.name Mallory", "git reset --hard", "git clean -fd"):
        with pytest.raises(PolicyDenied):
            vet(command, TurnFlags())
        with pytest.raises(PolicyDenied):
            backend.execute(command)
    assert _git(repo, "config", "user.name").strip() == "Ada Fixture"
    allowed = TurnFlags(destructive_requested=True)
    vet("git config user.name Mallory", allowed)
    vet("git reset --hard", allowed)
    vet("git clean -fd", allowed)


def test_commit_drops_trailers_and_keeps_repo_identity(tmp_path, monkeypatch, git_repo):
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Not Ada")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "not@example.com")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Not Ada")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "not@example.com")
    repo = git_repo(tmp_path / "repo")
    backend = GuardedShellBackend(root_dir=repo)
    (repo / "file.txt").write_text("x\n", encoding="utf-8")
    message = (
        "feat: add file\n\n"
        "Co-authored-by: Cursor <cursoragent@cursor.com>\n"
        "Assisted-by: Claude\n"
        "Generated-by: AbsolutCoder\n"
    )
    backend.execute("git add file.txt && git commit -m " + shlex.quote(message))
    body = _git(repo, "log", "-1", "--format=%B")
    assert "feat: add file" in body
    assert "Co-authored-by" not in body
    assert "Assisted-by" not in body
    assert "Generated-by" not in body
    assert "AbsolutCoder" not in body
    assert _git(repo, "log", "-1", "--format=%an").strip() == "Ada Fixture"
    assert _git(repo, "log", "-1", "--format=%cn").strip() == "Ada Fixture"
    assert _git(repo, "config", "user.name").strip() == "Ada Fixture"
    assert os.environ["GIT_AUTHOR_NAME"] == "Not Ada"


def test_scrub_commit_message_removes_attribution():
    scrubbed = scrub_commit_message(
        "feat: work\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n"
        "Assisted-by: someone\nGenerated-by: AbsolutCoder\n"
    )
    assert scrubbed == "feat: work"


def test_graphify_out_is_git_excluded(tmp_path, git_repo):
    repo = git_repo(tmp_path / "repo")
    protect_context_file(repo)
    protect_context_file(repo)
    lines = (repo / ".git" / "info" / "exclude").read_text(encoding="utf-8").splitlines()
    assert lines.count("graphify-out/") == 1
    assert lines.count("AbsolutCoder.md") == 1
    (repo / "graphify-out").mkdir()
    (repo / "graphify-out" / "graph.json").write_text("{}", encoding="utf-8")
    checked = subprocess.run(
        ["git", "check-ignore", "graphify-out/graph.json"], cwd=repo, capture_output=True, text=True
    )
    assert checked.returncode == 0, checked.stderr


def _ignored(folder: Path, path: str) -> bool:
    checked = subprocess.run(["git", "check-ignore", path], cwd=folder, capture_output=True, text=True)
    return checked.returncode == 0


def test_exclude_works_in_a_linked_worktree(tmp_path, git_repo):
    repo = git_repo(tmp_path / "repo")
    (repo / "readme.txt").write_text("x\n", encoding="utf-8")
    _git(repo, "add", "readme.txt")
    _git(repo, "commit", "-m", "first")
    worktree = tmp_path / "linked"
    _git(repo, "worktree", "add", "--detach", str(worktree))
    assert (worktree / ".git").is_file()
    protect_context_file(worktree)
    protect_context_file(worktree)
    assert _ignored(worktree, "AbsolutCoder.md")
    assert _ignored(worktree, "graphify-out/graph.json")
    common = Path(_git(worktree, "rev-parse", "--git-common-dir").strip())
    if not common.is_absolute():
        common = worktree / common
    lines = (common / "info" / "exclude").read_text(encoding="utf-8").splitlines()
    assert lines.count("AbsolutCoder.md") == 1
    assert lines.count("graphify-out/") == 1


def test_exclude_works_in_a_subfolder_of_a_repository(tmp_path, git_repo):
    repo = git_repo(tmp_path / "repo")
    exclude = repo / ".git" / "info" / "exclude"
    exclude.parent.mkdir(parents=True, exist_ok=True)
    exclude.write_text("*.log", encoding="utf-8")  # no trailing newline
    sub = repo / "pkg" / "inner"
    sub.mkdir(parents=True)
    protect_context_file(sub)
    assert _ignored(sub, "AbsolutCoder.md")
    assert _ignored(sub, "graphify-out/graph.json")
    assert exclude.read_text(encoding="utf-8").splitlines() == ["*.log", "AbsolutCoder.md", "graphify-out/"]


def test_exclude_works_under_a_non_ascii_folder(tmp_path, git_repo):
    before = set(os.listdir(tmp_path))
    parent = tmp_path / "prøjekt-日本"
    repo = git_repo(parent / "repo")
    (repo / "readme.txt").write_text("x\n", encoding="utf-8")
    _git(repo, "add", "readme.txt")
    _git(repo, "commit", "-m", "first")
    worktree = parent / "linked"
    _git(repo, "worktree", "add", "--detach", str(worktree))
    for folder in (repo, worktree):
        protect_context_file(folder)
        assert _ignored(folder, "AbsolutCoder.md"), folder
        assert _ignored(folder, "graphify-out/graph.json"), folder
    assert set(os.listdir(tmp_path)) == before | {"prøjekt-日本"}
    assert set(os.listdir(parent)) == {"repo", "linked"}
    lines = (repo / ".git" / "info" / "exclude").read_text(encoding="utf-8").splitlines()
    assert lines.count("AbsolutCoder.md") == 1
    assert lines.count("graphify-out/") == 1


def test_bare_repository_exclude_is_untouched(tmp_path):
    bare = tmp_path / "bare.git"
    subprocess.run(["git", "init", "--bare", str(bare)], check=True, capture_output=True)
    exclude = bare / "info" / "exclude"
    before = exclude.read_bytes() if exclude.exists() else None
    protect_context_file(bare)
    assert (exclude.read_bytes() if exclude.exists() else None) == before


def test_read_only_exclude_file_does_not_raise(tmp_path, git_repo):
    import stat

    repo = git_repo(tmp_path / "repo")
    exclude = repo / ".git" / "info" / "exclude"
    exclude.parent.mkdir(parents=True, exist_ok=True)
    exclude.write_text("*.log\n", encoding="utf-8")
    os.chmod(exclude, stat.S_IREAD)
    try:
        protect_context_file(repo)
        assert exclude.read_text(encoding="utf-8") == "*.log\n"
    finally:
        os.chmod(exclude, stat.S_IREAD | stat.S_IWRITE)


def test_missing_reported_git_directory_creates_nothing(tmp_path, monkeypatch):
    import absolutcoder.git_policy as git_policy

    folder = tmp_path / "plain"
    missing = tmp_path / "missing-git-dir"
    real_run = subprocess.run

    def fake_run(args, *rest, **kwargs):
        if list(args[:2]) == ["git", "rev-parse"]:
            stdout = f"true\n{missing.as_posix()}/info/exclude\n".encode()
            return subprocess.CompletedProcess(args, 0, stdout, b"")
        return real_run(args, *rest, **kwargs)

    monkeypatch.setattr(git_policy.subprocess, "run", fake_run)
    protect_context_file(folder)
    assert not missing.exists()
    assert sorted(p.name for p in tmp_path.iterdir() if p.name != "absolutcoder-home") == ["plain"]


def test_non_repository_gets_no_git_files(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    folder = tmp_path / "plain"
    protect_context_file(folder)
    protect_context_file(folder)
    assert sorted(p.name for p in folder.iterdir()) == ["AbsolutCoder.md"]


def test_context_file_is_excluded_and_unstaged(tmp_path, git_repo):
    repo = git_repo(tmp_path / "repo")
    protect_context_file(repo)
    (repo / "tracked.txt").write_text("y\n", encoding="utf-8")
    subprocess.run(["git", "add", "-f", "AbsolutCoder.md"], cwd=repo, check=True, capture_output=True)
    backend = GuardedShellBackend(root_dir=repo)
    backend.execute("git add -A")
    staged = _git(repo, "diff", "--cached", "--name-only").splitlines()
    assert "tracked.txt" in staged
    assert "AbsolutCoder.md" not in staged
    exclude = (repo / ".git" / "info" / "exclude").read_text(encoding="utf-8")
    assert "AbsolutCoder.md" in exclude
    with pytest.raises(PolicyDenied):
        backend.execute("git add AbsolutCoder.md")
    with pytest.raises(PolicyDenied):
        backend.execute("git add -f AbsolutCoder.md")


def test_exported_identity_is_denied():
    for command in (
        "export GIT_AUTHOR_NAME=x",
        "GIT_AUTHOR_NAME=x",
        "set GIT_AUTHOR_NAME=x",
        "setx GIT_AUTHOR_NAME x",
        '$env:GIT_AUTHOR_NAME = "x"',
        "declare -x GIT_COMMITTER_EMAIL=x",
        "env GIT_AUTHOR_NAME=x git commit -m y",
        "env GIT_AUTHOR_NAME=x make",
        "export GIT_COMMITTER_NAME=x; git commit -m y",
    ):
        assert sets_commit_identity(command.split(";")[0]), command
        with pytest.raises(PolicyDenied, match="commit identity variables cannot be set"):
            vet(command, TurnFlags())
    assert not sets_commit_identity("GIT_AUTHOR_NAME=x git commit -m y")
    vet("GIT_AUTHOR_NAME=x git commit -m y", TurnFlags())
    vet("echo $GIT_AUTHOR_NAME && git log --author=Ada", TurnFlags())


def test_render_command_quotes_for_posix_shells():
    tokens = ["git", "commit", "-m", "two words", "it's", "$HOME and $(id)", 'say "hi"', ""]
    for kind in ("bash", "sh"):
        assert shlex.split(render_command(tokens, kind)) == tokens


def test_turn_flags_follow_the_user_message():
    assert TurnFlags.from_message("please push the branch").push_requested is True
    assert TurnFlags.from_message("don't push").push_requested is False
    assert TurnFlags.from_message("pull the latest and fetch tags").pull_requested is True
    assert TurnFlags.from_message("git reset --hard HEAD").destructive_requested is True
    assert TurnFlags.from_message("commit the fix").push_requested is False


_ALL_FLAGS = TurnFlags(push_requested=True, pull_requested=True, destructive_requested=True)


@pytest.mark.parametrize(
    "command",
    [
        'sh -c "git push"',
        "bash -lc 'git push origin main'",
        "cmd /c git push",
        'cmd.exe /d /s /c "git push"',
        'powershell -Command "git push"',
        'eval "git push"',
        "env X=1 git push",
        "nohup git push",
        "bash -c \"sh -c 'git push'\"",
        '/bin/sh -c "git push"',
        'C:\\Windows\\System32\\cmd.exe /c "git push"',
        'if sh -c "git push"; then echo ok; fi',
        "sudo -u root sh -c 'git push'",
        "echo $(sh -c 'git push')",
    ],
)
def test_wrapped_git_push_is_denied(command):
    with pytest.raises(PolicyDenied):
        vet(command, TurnFlags())
    vet(command, TurnFlags(push_requested=True))


def test_wrapped_destructive_git_is_denied():
    with pytest.raises(PolicyDenied):
        vet('sh -c "git reset --hard"', TurnFlags())


@pytest.mark.parametrize(
    "command",
    [
        'sh -c "git commit -m x"',
        "bash -lc 'git add -A && git commit -m x'",
        'eval "git commit -m x"',
        "env X=1 git add f.txt",
        "echo $(git commit -m x)",
    ],
)
def test_wrapped_commit_and_add_are_refused(command):
    for flags in (TurnFlags(), _ALL_FLAGS):
        with pytest.raises(PolicyDenied, match="run git commit and git add directly"):
            vet(command, flags)
    vet("git commit -m x", TurnFlags())
    vet("git add f.txt", TurnFlags())


def _nested_bash(depth: int, inner: str) -> str:
    command = inner
    for _ in range(depth):
        command = "bash -c " + shlex.quote(command)
    return command


@pytest.mark.parametrize(
    "command",
    ["env env env env env sh -c 'git push'", _nested_bash(6, "git push")],
)
def test_too_many_nested_wrappers_are_refused(command):
    for flags in (TurnFlags(), _ALL_FLAGS):
        with pytest.raises(PolicyDenied, match="too many nested wrappers"):
            vet(command, flags)
    assert execute_needs_approval(command) is True


def test_nested_wrappers_within_the_limit_are_vetted_normally():
    vet("env env env env echo hi", TurnFlags())
    vet(_nested_bash(5, "echo hi"), TurnFlags())
    with pytest.raises(PolicyDenied, match="git push is not allowed"):
        vet(_nested_bash(5, "git push"), TurnFlags())


def test_wrapper_always_needs_approval():
    assert execute_needs_approval('sh -c "git commit -m x"') is True
    assert execute_needs_approval("git commit -m x") is False


@pytest.mark.parametrize("command", ["git status", 'python -c "print(1)"', "ls -la"])
def test_unwrap_shell_returns_none_for_plain_commands(command):
    assert unwrap_shell(shlex.split(command)) is None


def test_unwrap_shell_returns_the_inner_command():
    assert unwrap_shell(["bash", "-ec", "git status"]) == "git status"
    assert unwrap_shell(["cmd", "/C", "git", "push"]) == "git push"
    assert unwrap_shell(["pwsh.exe", "-NoProfile", "-c", "git", "push"]) == "git push"
    assert unwrap_shell(["eval", "git", "push"]) == "git push"
    assert shlex.split(unwrap_shell(["env", "-u", "HOME", "X=1", "sh", "-c", "git push"])) == [
        "sh",
        "-c",
        "git push",
    ]
    assert unwrap_shell(["X=1", "git", "push"]) == "git push"
    assert unwrap_shell(["sudo", "-u", "root", "git", "push"]) == "git push"


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("push this logic into a helper", False),
        ("push the button handler down into the base class", False),
        ("push the value onto the stack", False),
        ("please push", True),
        ("push it", True),
        ("commit and push the changes", True),
        ("git push", True),
        ("run git push into prod", True),
        ("then push to origin", True),
        ("don't push", False),
        ("commit the fix", False),
        ("fix the bug, then push.", True),
        ("please push\nthen run the tests", True),
        ("commit this\npush", True),
    ],
)
def test_push_as_ordinary_verb_does_not_unlock(message, expected):
    assert TurnFlags.from_message(message).push_requested is expected


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("pull this out into a function", False),
        ("open a pull request", False),
        ("pull the latest changes", True),
        ("fetch origin", True),
        ("git pull", True),
    ],
)
def test_pull_as_ordinary_verb_does_not_unlock(message, expected):
    assert TurnFlags.from_message(message).pull_requested is expected


# --- A1: here-documents ------------------------------------------------------


def test_heredoc_body_is_not_vetted_as_commands():
    command = "cat > notes.txt <<'EOF'\ngit push origin main\ngit reset --hard\nEOF\ngit add notes.txt"
    vet(command, TurnFlags())
    with pytest.raises(PolicyDenied, match="git push"):
        vet(command + "\ngit push", TurnFlags())


def test_expanding_heredoc_body_substitutions_are_vetted():
    with pytest.raises(PolicyDenied, match="git push"):
        vet("cat <<EOF\n$(git push)\nEOF", TurnFlags())
    vet("cat <<'EOF'\n$(git push)\nEOF", TurnFlags())


def test_span_aware_splitter_keeps_two_heredocs_with_their_segments():
    from absolutcoder.git_policy import command_segments, split_command_chain

    command = (
        "cat > a.txt <<'EOF' && cat > b.txt <<-END; echo done\n"
        "a && b; c | d\n"
        "EOF\n"
        "\tb body\n"
        "\tEND\n"
        "git add a.txt b.txt"
    )
    segments = command_segments(command)
    assert [(s.operator, s.text) for s in segments] == [
        ("", "cat > a.txt <<'EOF'"),
        ("&&", "cat > b.txt <<-END"),
        (";", "echo done"),
        ("\n", "git add a.txt b.txt"),
    ]
    for segment in segments:
        assert command[segment.start : segment.end] == segment.text
    first, second = segments[0].heredocs, segments[1].heredocs
    assert [(doc.delimiter, doc.expands) for doc in first] == [("EOF", False)]
    assert [(doc.delimiter, doc.expands) for doc in second] == [("END", True)]
    assert command[first[0].start : first[0].end] == "a && b; c | d\nEOF"
    assert command[second[0].start : second[0].end] == "\tb body\n\tEND"
    assert segments[2].heredocs == () and segments[3].heredocs == ()
    assert split_command_chain(command) == [(s.operator, s.text) for s in segments]
    assert split_command_chain("cat <<<'a; b' && echo x") == [("", "cat <<<'a; b'"), ("&&", "echo x")]


# --- A2: shell punctuation ---------------------------------------------------


@pytest.mark.parametrize(
    ("command", "flags"),
    [
        ("(cd sub && git push)", TurnFlags(push_requested=True)),
        ("(git push)", TurnFlags(push_requested=True)),
        ("git push&", TurnFlags(push_requested=True)),
        ("git push>log.txt", TurnFlags(push_requested=True)),
        ("git push 2>&1", TurnFlags(push_requested=True)),
        ("(cd sub && git reset --hard)", TurnFlags(destructive_requested=True)),
        ("git branch tmp & git push", TurnFlags(push_requested=True)),
        ("{ git push; }", TurnFlags(push_requested=True)),
        ('echo "$(git push)"', TurnFlags(push_requested=True)),
        ("git status; git log; git push", TurnFlags(push_requested=True)),
    ],
)
def test_shell_punctuation_does_not_hide_git(command, flags):
    with pytest.raises(PolicyDenied):
        vet(command, TurnFlags())
    vet(command, flags)


@pytest.mark.parametrize(
    "command",
    [
        'git branch "$(echo x)"',
        "git branch `echo x`",
        "git branch x & echo hi",
        "git merge x > file",
        "(git commit -m x)",
        "git commit -m x | cat",
        "git commit -F - <<'EOF'\nx\nEOF",
    ],
)
def test_local_git_with_shell_features_needs_approval(command):
    assert execute_needs_approval(command) is True


@pytest.mark.parametrize(
    "command",
    ['git commit -m "x"', "git branch feature", "git switch main", "git merge feature",
     'git commit -m "a > b & (c)"'],
)
def test_plain_local_git_needs_no_approval(command):
    assert execute_needs_approval(command) is False


def test_add_then_commit_still_needs_approval():
    # Pinned as found before the punctuation fix: add is not a no-prompt subcommand.
    assert execute_needs_approval('git add a && git commit -m "x"') is True


# --- B: commit rewriting -----------------------------------------------------


def _rewrite(segment: str, tmp_path: Path, kind: str = "bash") -> tuple[str, str | None]:
    from absolutcoder.git_policy import prepare_commit_segment

    rewritten, path = prepare_commit_segment(segment, tmp_path, kind)
    text = path.read_text(encoding="utf-8") if path is not None else None
    if path is not None:
        path.unlink()
    return rewritten, text


def test_dynamic_message_is_left_to_the_shell_and_filtered(tmp_path):
    for segment in (
        'git commit -m "$(cat msg.txt)"',
        "git commit -m \"subject $VAR\"",
        "git commit -m `cat msg.txt`",
    ):
        rewritten, written = _rewrite(segment, tmp_path)
        assert written is None, segment
        assert rewritten.startswith("printf '%s\\n\\n' "), rewritten
        message_word = segment.split(" -m ", 1)[1]
        assert message_word in rewritten
        assert "| grep -Eiv " in rewritten and rewritten.endswith(" -F -")
        assert "env -u GIT_AUTHOR_NAME" in rewritten
    # Single quotes keep $ literal: the message is still written ahead.
    rewritten, written = _rewrite("git commit -m '$(not run) $HOME'", tmp_path)
    assert written == "$(not run) $HOME\n"
    # cmd keeps writing the message ahead.
    _rewritten, written = _rewrite('git commit -m "$(cat msg.txt)"', tmp_path, "cmd")
    assert written == "$(cat msg.txt)\n"


def test_untracked_mode_is_not_read_as_a_message(tmp_path):
    rewritten, written = _rewrite('git commit -unormal -m "x"', tmp_path)
    assert written == "x\n"
    assert "-unormal" in shlex.split(rewritten)
    # -um is unusual: it may pass through, but the message must never be corrupted.
    rewritten, written = _rewrite('git commit -um "x"', tmp_path)
    assert written in {None, "x\n"}
    if written is None:
        assert "-um" in shlex.split(rewritten)


@pytest.mark.parametrize(
    ("segment", "message", "kept"),
    [
        ('git commit -m"fix algorithm"', "fix algorithm", []),
        ("git commit -mSubject", "Subject", []),
        ('git commit -am "x"', "x", ["-a"]),
        ('git commit -amfix', "fix", ["-a"]),
        ('git commit -m"feat: y" -m"Co-authored-by: A <a@b.c>"', "feat: y", []),
    ],
)
def test_glued_message_options(tmp_path, segment, message, kept):
    rewritten, written = _rewrite(segment, tmp_path)
    assert written == message + "\n"
    words = shlex.split(rewritten)
    for flag in kept:
        assert flag in words
    assert not any(word.startswith("-m") for word in words)


def test_attribution_trailers_are_dropped_and_others_kept(tmp_path):
    rewritten, _ = _rewrite(
        'git commit -s -m x --trailer "Co-authored-by: A <a@b.c>" --trailer="Assisted-by: tool" '
        '--trailer "Reviewed-by: B <b@c.d>" --trailer=Generated-by:AbsolutCoder --trailer "Note: absolutcoder"',
        tmp_path,
    )
    words = shlex.split(rewritten)
    assert "Reviewed-by: B <b@c.d>" in words
    assert "-s" in words
    assert not any("Co-authored-by" in w or "Assisted-by" in w or "Generated-by" in w for w in words)
    assert not any("absolutcoder" in w.lower() for w in words)


def test_merge_message_is_scrubbed_and_revert_left_alone(tmp_path):
    rewritten, written = _rewrite('git merge --no-ff -m "merge\n\nGenerated-by: tool" side', tmp_path)
    assert written == "merge\n"
    words = shlex.split(rewritten)
    assert "-F" in words and "side" in words and "--no-ff" in words
    assert "env -u GIT_AUTHOR_NAME" in rewritten
    for segment in ("git revert -m 1 HEAD", "git cherry-pick -m 1 abc"):
        assert _rewrite(segment, tmp_path) == (segment, None)
