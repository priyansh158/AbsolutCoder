"""Git and shell policy for one user turn.

Push, pull, and destructive git operations run only when that turn's message
asked for them. Commit messages lose attribution trailers before git stores them.
AbsolutCoder.md stays out of the index.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

CONTEXT_FILENAME = "AbsolutCoder.md"
_NO_PROMPT_GIT = frozenset({"commit", "branch", "switch", "merge"})
_TRAILER = re.compile(r"^\s*(co-authored-by|assisted-by|generated-by)\s*:", re.IGNORECASE)
_ATTRIBUTION = re.compile(r"absolut\s*coder|absolutcoder", re.IGNORECASE)
_NEGATION = re.compile(r"\b(don't|do not|never|without|no)\s*$")
_IDENTITY_PREFIXES = ("GIT_AUTHOR_", "GIT_COMMITTER_")
_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_ASSIGNING_COMMANDS = frozenset(
    {"export", "declare", "typeset", "readonly", "local", "set", "setx", "setx.exe", "env"}
)
# The lines scrub_commit_message drops, for grep -Eiv at run time.
_SCRUB_PATTERN = (
    "^[[:space:]]*(co-authored-by|assisted-by|generated-by)[[:space:]]*:|absolut[[:space:]]*coder"
)
# A redirection token such as 2>&1, >/dev/null or <<EOF stays unquoted for a POSIX shell.
_REDIRECTION = re.compile(r"^(?:\d*|&)(?:>>|>|<<<|<<-|<<|<)&?[\w./:\\~+-]*$")

DEFAULT_MEMORY = """# AbsolutCoder

Project memory for this repository. Update this file as you learn how the
project is built, tested, and laid out.

This file stays untracked. Do not commit or push it.
"""


class PolicyDenied(ValueError):
    """The shell guard refused to run a command."""


@dataclass(frozen=True)
class TurnFlags:
    """What the current user message allowed the shell to do."""

    push_requested: bool = False
    pull_requested: bool = False
    destructive_requested: bool = False

    @classmethod
    def from_message(cls, message: str) -> TurnFlags:
        text = message.lower()
        return cls(
            push_requested=_asked(text, "push"),
            pull_requested=_asked(text, "pull") or _asked(text, "fetch"),
            destructive_requested=_asked_destructive(text),
        )


def vet(command: str, flags: TurnFlags) -> None:
    """Raise PolicyDenied when a command is outside this turn's git policy.

    Compound commands are refused as a whole. ``git status && git push`` does
    not run when the user did not ask to push.
    """

    for segment in iter_vet_segments(command):
        if sets_commit_identity(segment):
            raise PolicyDenied("commit identity variables cannot be set")
        _vet_segment(segment, flags)


def sets_commit_identity(segment: str) -> bool:
    """True when ``segment`` assigns, exports or sets a GIT_AUTHOR_* / GIT_COMMITTER_* variable.

    A prefix assignment on a ``git commit`` line is allowed: it is stripped
    before the commit runs. Anywhere else the value could outlive the segment
    (one process runs the whole command) or reach a git that does not strip it.
    """

    tokens = _tokens(segment)
    if any(_names_powershell_identity(token) for token in tokens):
        return True
    index = 0
    while index < len(tokens) and _ASSIGNMENT.match(tokens[index]):
        index += 1
    leading, rest = tokens[:index], tokens[index:]
    if any(_is_identity_assignment(token) for token in leading):
        if not rest or _command_name(rest[0]) not in {"git", "git.exe"}:
            return True
        if _split_git_args(rest[1:])[0] != "commit":
            return True
    if rest and _command_name(rest[0]) in _ASSIGNING_COMMANDS:
        for token in rest[1:]:
            if token.split("=", 1)[0].upper().startswith(_IDENTITY_PREFIXES):
                return True
    return False


def scrub_commit_message(message: str) -> str:
    """Drop attribution trailers and AbsolutCoder credits from a commit message."""

    kept: list[str] = []
    for line in message.splitlines():
        if _TRAILER.match(line) or _ATTRIBUTION.search(line):
            continue
        kept.append(line)
    text = "\n".join(kept).strip()
    return text or "Update"


def protect_context_file(repo: Path | str) -> None:
    """Create AbsolutCoder.md when missing and keep it out of git."""

    root = Path(repo).resolve()
    root.mkdir(parents=True, exist_ok=True)
    memory = root / CONTEXT_FILENAME
    if not memory.exists():
        memory.write_text(DEFAULT_MEMORY, encoding="utf-8")
    git_dir = root / ".git"
    if not git_dir.is_dir():
        return
    info = git_dir / "info"
    info.mkdir(parents=True, exist_ok=True)
    exclude = info / "exclude"
    existing = exclude.read_text(encoding="utf-8") if exclude.exists() else ""
    lines = [line.strip() for line in existing.splitlines()]
    if CONTEXT_FILENAME not in lines:
        suffix = "" if existing.endswith("\n") or not existing else "\n"
        exclude.write_text(existing + suffix + CONTEXT_FILENAME + "\n", encoding="utf-8")
    unstage_context_file(root)


def execute_needs_approval(command: str) -> bool:
    """Ordinary shell commands ask first. Local git writes do not ask again."""

    segments = [segment for segment in iter_vet_segments(command) if segment.strip()]
    if not segments:
        return True
    for segment in segments:
        if git_subcommand(segment) not in _NO_PROMPT_GIT:
            return True
    return False


def split_command_chain(command: str) -> list[tuple[str, str]]:
    """Split a shell command on &&, ||, ;, |, and newlines outside quotes."""

    parts: list[tuple[str, str]] = []
    buf: list[str] = []
    operator = ""
    quote = ""
    index = 0
    while index < len(command):
        char = command[index]
        if quote:
            buf.append(char)
            if char == quote and (index == 0 or command[index - 1] != "\\"):
                quote = ""
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            buf.append(char)
            index += 1
            continue
        if command.startswith("&&", index) or command.startswith("||", index):
            parts.append((operator, "".join(buf)))
            operator = command[index : index + 2]
            buf = []
            index += 2
            continue
        if char in {";", "\n", "|"}:
            parts.append((operator, "".join(buf)))
            operator = char
            buf = []
            index += 1
            continue
        buf.append(char)
        index += 1
    parts.append((operator, "".join(buf)))
    return [(op, segment.strip()) for op, segment in parts if segment.strip()]


def iter_vet_segments(command: str) -> list[str]:
    segments: list[str] = []
    for _operator, segment in split_command_chain(command):
        segments.append(segment)
        segments.extend(_substitutions(segment))
    return segments


def git_subcommand(segment: str) -> str | None:
    """Return the git subcommand for a segment, if it is a git invocation."""

    invocation = _git_invocation(_tokens(segment))
    if invocation is None:
        return None
    _subcommand, _rest = _split_git_args(invocation[1:])
    return _subcommand


def prepare_commit_segment(segment: str, repo: Path, kind: str = "cmd") -> tuple[str, Path | None]:
    """Rewrite a git commit so the stored message has no attribution trailer.

    Identity assignments and ``--author`` are removed. The message is passed
    with ``-F`` so the host shell does not have to quote newlines. ``kind`` is
    the shell that runs the result: ``cmd``, ``bash`` or ``sh``.

    For a POSIX shell, a message read with ``-F <file>`` or ``-F -`` (a pipe or
    here-document) is filtered through ``grep`` at run time, because its text
    is not known before the script runs.
    """

    if git_subcommand(segment) != "commit":
        return segment, None
    posix = kind != "cmd"
    tokens = [token for token in _tokens(segment) if not _is_identity_assignment(token)]
    messages: list[str] = []
    kept: list[str] = []
    message_source: str | None = None
    stdin_redirects: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if posix and token in {"-F", "--file"} and index + 1 < len(tokens):
            message_source = tokens[index + 1]
            index += 2
            continue
        if posix and token.startswith("--file="):
            message_source = token.split("=", 1)[1]
            index += 1
            continue
        if posix and token.startswith("-F") and len(token) > 2:
            message_source = token[2:]
            index += 1
            continue
        if posix and token.startswith("<") and _REDIRECTION.match(token):
            stdin_redirects.append(token)
            if token in {"<", "<<", "<<-", "<<<"} and index + 1 < len(tokens):
                stdin_redirects.append(tokens[index + 1])
                index += 1
            index += 1
            continue
        if token == "--author":
            index += 2
            continue
        if token.startswith("--author="):
            index += 1
            continue
        if token in {"-m", "--message"}:
            index += 1
            if index < len(tokens):
                messages.append(tokens[index])
            index += 1
            continue
        if token.startswith("--message="):
            messages.append(token.split("=", 1)[1])
            index += 1
            continue
        if token.startswith("-") and not token.startswith("--") and token.endswith("m") and len(token) > 2:
            prefix = token[1:-1]
            index += 1
            if index < len(tokens):
                messages.append(tokens[index])
            if prefix:
                kept.append("-" + prefix)
            index += 1
            continue
        kept.append(token)
        index += 1
    if messages:
        path = _write_message(repo, scrub_commit_message("\n\n".join(messages)))
        kept.extend(["-F", str(path)])
        kept.extend(stdin_redirects)
        return render_command(kept, kind), path
    if message_source is None:
        return render_command(kept + stdin_redirects, kind), None
    source = stdin_redirects if message_source == "-" else ["<", message_source]
    scrubber = render_command(["grep", "-Eiv", _SCRUB_PATTERN, *source], kind)
    return scrubber + " | " + render_command([*kept, "-F", "-"], kind), None


def render_command(tokens: list[str], kind: str) -> str:
    """Join ``tokens`` into one command line for the shell ``kind``.

    ``cmd`` keeps the double-quote rules cmd.exe understands. ``bash`` and
    ``sh`` quote every word with ``shlex.quote``, keep redirections such as
    ``2>&1`` or ``<<EOF`` working (a here-document delimiter is quoted, so its
    body is not expanded), and write existing absolute Windows paths with
    forward slashes.
    """

    if kind == "cmd":
        return _join_cmd(tokens)
    rendered: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in {"<<", "<<-"} and index + 1 < len(tokens):
            rendered.append(_heredoc(token, tokens[index + 1]))
            index += 2
            continue
        rendered.append(_posix_word(token))
        index += 1
    return " ".join(rendered)


def _heredoc(operator: str, delimiter: str) -> str:
    if "'" in delimiter:
        return operator + shlex.quote(delimiter)
    return operator + "'" + delimiter + "'"


def _posix_word(token: str) -> str:
    if _REDIRECTION.match(token):
        if not token.startswith("<<<"):
            for operator in ("<<-", "<<"):
                if token.startswith(operator) and len(token) > len(operator):
                    return _heredoc(operator, token[len(operator):])
        return token
    assignment = _ASSIGNMENT.match(token)
    if assignment:
        return assignment.group(0) + shlex.quote(token[assignment.end():])
    if "\\" in token and _is_existing_absolute(token):
        token = token.replace("\\", "/")
    return shlex.quote(token)


def _is_existing_absolute(token: str) -> bool:
    try:
        path = Path(token)
        return path.is_absolute() and path.exists()
    except (OSError, ValueError):
        return False


def _asked(text: str, verb: str) -> bool:
    for match in re.finditer(rf"\b{verb}\b", text):
        window = text[max(0, match.start() - 24) : match.start()]
        if _NEGATION.search(window):
            continue
        return True
    return False


def _asked_destructive(text: str) -> bool:
    if re.search(r"\bgit\s+config\b", text):
        return True
    if re.search(r"\breset\b", text) and re.search(r"\bhard\b", text):
        return True
    if re.search(r"\bclean\b", text) and re.search(r"(?:^|\s)-[a-z]*f|\bforce\b", text):
        return True
    return False


def _vet_segment(segment: str, flags: TurnFlags) -> None:
    invocation = _git_invocation(_tokens(segment))
    if invocation is None:
        return
    if _git_sets_config(invocation[1:]):
        if not flags.destructive_requested:
            raise PolicyDenied("git config is not allowed unless this message asks for it")
    subcommand, rest = _split_git_args(invocation[1:])
    if subcommand is None:
        return
    if subcommand == "push" and not flags.push_requested:
        raise PolicyDenied("git push is not allowed unless this message asks to push")
    if subcommand in {"pull", "fetch"} and not flags.pull_requested:
        raise PolicyDenied("git pull and git fetch are not allowed unless this message asks to pull")
    if subcommand == "config" and not flags.destructive_requested:
        raise PolicyDenied("git config is not allowed unless this message asks for it")
    if subcommand == "reset" and _reset_is_hard(rest) and not flags.destructive_requested:
        raise PolicyDenied("git reset --hard is not allowed unless this message asks for it")
    if subcommand == "clean" and _clean_is_forced(rest) and not flags.destructive_requested:
        raise PolicyDenied("git clean -fd is not allowed unless this message asks for it")
    if subcommand == "add" and _adds_context_file(rest):
        raise PolicyDenied("AbsolutCoder.md is excluded from git and cannot be added")


def _git_sets_config(args: list[str]) -> bool:
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--":
            return False
        if arg in {"-c", "--config-env"}:
            return True
        if arg.startswith("-c") and not arg.startswith("--"):
            return True
        if arg.startswith("--config-env"):
            return True
        if arg in {"-C", "--git-dir", "--work-tree", "--namespace", "--exec-path", "--super-prefix"}:
            index += 2
            continue
        if arg.startswith("--") and "=" not in arg and arg not in {"--no-pager", "--paginate", "--bare", "--literal-pathspecs"}:
            if arg in {"--git-dir", "--work-tree", "--namespace"}:
                index += 2
                continue
        if not arg.startswith("-"):
            return False
        index += 1
    return False


def _split_git_args(args: list[str]) -> tuple[str | None, list[str]]:
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--":
            return None, args[index + 1 :]
        if arg in {"-C", "--git-dir", "--work-tree", "--namespace", "--exec-path", "--super-prefix", "-c", "--config-env"}:
            index += 2
            continue
        if arg.startswith("--") and "=" in arg:
            index += 1
            continue
        if arg.startswith("-"):
            index += 1
            continue
        return arg.lower(), args[index + 1 :]
    return None, []


def _git_invocation(tokens: list[str]) -> list[str] | None:
    for index, token in enumerate(tokens):
        if _is_identity_assignment(token):
            continue
        base = token.replace("\\", "/").rsplit("/", 1)[-1].lower()
        if base in {"git", "git.exe"}:
            return tokens[index:]
    return None


def _tokens(segment: str) -> list[str]:
    try:
        return shlex.split(segment, posix=True)
    except ValueError:
        return segment.split()


def _is_identity_assignment(token: str) -> bool:
    return "=" in token and token.upper().startswith(_IDENTITY_PREFIXES)


def _names_powershell_identity(token: str) -> bool:
    lowered = token.lower()
    for prefix in ("$env:", "${env:"):
        if lowered.startswith(prefix):
            return token[len(prefix) :].upper().startswith(_IDENTITY_PREFIXES)
    return False


def _command_name(token: str) -> str:
    return token.replace("\\", "/").rsplit("/", 1)[-1].lower()


def _reset_is_hard(args: list[str]) -> bool:
    return any(arg == "--hard" or arg.startswith("--hard=") for arg in args)


def _clean_is_forced(args: list[str]) -> bool:
    for arg in args:
        if arg in {"--force", "-f", "-fd", "-df", "-ff", "-xf", "-fx"}:
            return True
        if arg.startswith("-") and not arg.startswith("--") and "f" in arg[1:]:
            return True
        if arg == "--force":
            return True
    return False


def _adds_context_file(args: list[str]) -> bool:
    for arg in args:
        if arg == "--":
            continue
        if arg.startswith("-"):
            continue
        name = arg.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
        if name == CONTEXT_FILENAME:
            return True
    return False


def _substitutions(segment: str) -> list[str]:
    found: list[str] = []
    quote = ""
    index = 0
    while index < len(segment):
        char = segment[index]
        if quote:
            if char == quote:
                quote = ""
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            index += 1
            continue
        if segment.startswith("$(", index):
            end = _matching_paren(segment, index + 1)
            if end is None:
                break
            inner = segment[index + 2 : end]
            found.append(inner)
            found.extend(_substitutions(inner))
            index = end + 1
            continue
        if char == "`":
            end = segment.find("`", index + 1)
            if end == -1:
                break
            inner = segment[index + 1 : end]
            found.append(inner)
            found.extend(_substitutions(inner))
            index = end + 1
            continue
        index += 1
    return found


def _matching_paren(text: str, open_index: int) -> int | None:
    depth = 0
    for index in range(open_index, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return index
    return None


def _join_cmd(tokens: list[str]) -> str:
    rendered: list[str] = []
    for token in tokens:
        if any(char in token for char in ' \t"'):
            rendered.append('"' + token.replace('"', "") + '"')
        else:
            rendered.append(token)
    return " ".join(rendered)


def _write_message(repo: Path, message: str) -> Path:
    directory = repo / ".git"
    if not directory.is_dir():
        directory = Path(os.environ.get("TEMP") or os.environ.get("TMP") or ".")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"absolut-commit-{uuid.uuid4().hex}.txt"
    text = message if message.endswith("\n") else message + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def unstage_context_file(repo: Path, env: dict[str, str] | None = None) -> None:
    """Remove AbsolutCoder.md from the index without deleting the working file.

    ``git restore --staged`` needs a HEAD. Before the first commit, fall back
    to dropping the index entry directly.
    """

    kwargs: dict = {"cwd": repo, "check": False, "capture_output": True, "text": True}
    if env is not None:
        kwargs["env"] = env
    restored = subprocess.run(["git", "restore", "--staged", "--", CONTEXT_FILENAME], **kwargs)
    if restored.returncode != 0:
        subprocess.run(
            ["git", "rm", "--cached", "--ignore-unmatch", "--", CONTEXT_FILENAME],
            **kwargs,
        )
