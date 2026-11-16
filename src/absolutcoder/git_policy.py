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
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)|[;\n]")
# "push" ends its clause, or is followed by a word that only finishes the request.
_PUSH_ENDING = re.compile(
    r"\s*(?:$|[.,;:!?)])|\s+(?:now|too|as well|afterwards|after|when done|once done|when finished)\b"
)
_PUSH_OBJECTS = frozenset(
    {"it", "them", "everything", "branch", "branches", "commit", "commits", "change", "changes",
     "work", "code", "tag", "tags", "remote", "origin", "upstream", "repo", "repository", "fork",
     "github", "gitlab", "bitbucket", "main", "master", "pr"}
)
_IDENTITY_PREFIXES = ("GIT_AUTHOR_", "GIT_COMMITTER_")
_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_ASSIGNING_COMMANDS = frozenset(
    {"export", "declare", "typeset", "readonly", "local", "set", "setx", "setx.exe", "env"}
)
_ENV_WITHOUT_IDENTITY = (
    "env",
    *(
        part
        for name in (
            "GIT_AUTHOR_NAME",
            "GIT_AUTHOR_EMAIL",
            "GIT_AUTHOR_DATE",
            "GIT_COMMITTER_NAME",
            "GIT_COMMITTER_EMAIL",
            "GIT_COMMITTER_DATE",
            "GIT_CONFIG_COUNT",
            "GIT_CONFIG_PARAMETERS",
        )
        for part in ("-u", name)
    ),
)
# The lines scrub_commit_message drops, for grep -Eiv at run time.
_SCRUB_PATTERN = (
    "^[[:space:]]*(co-authored-by|assisted-by|generated-by)[[:space:]]*:|absolut[[:space:]]*coder"
)
_POSIX_SHELLS = frozenset(
    name + suffix for name in ("sh", "bash", "zsh", "dash", "ksh") for suffix in ("", ".exe")
)
_POWERSHELLS = frozenset({"powershell", "powershell.exe", "pwsh", "pwsh.exe"})
# Commands that run the rest of their arguments, with the options that take a value.
_RUNNERS = {
    "nohup": frozenset(),
    "time": frozenset({"-f", "-o", "--format", "--output"}),
    "command": frozenset(),
    "exec": frozenset({"-a"}),
    "sudo": frozenset({"-u", "-g", "-C", "-D", "-h", "-p", "-r", "-t", "-U", "-T", "--user", "--group"}),
}
_SHELL_KEYWORDS = frozenset({"do", "then", "else", "elif", "if", "while", "until", "!", "{", "("})
_MAX_UNWRAP_DEPTH = 5
_WRAPPED_INDEX_WRITE = "run git commit and git add directly, not inside sh -c, eval or another wrapper"
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
    not run when the user did not ask to push. Commands inside ``sh -c``,
    ``eval`` and similar wrappers are vetted too, and a ``git commit`` or
    ``git add`` inside a wrapper or a ``$(...)`` substitution is refused: the
    shell backend rewrites only the ones it can see at the top level.
    """

    for segment, wrapped in _walk_segments(command):
        if sets_commit_identity(segment):
            raise PolicyDenied("commit identity variables cannot be set")
        if wrapped and git_subcommand(segment) in {"commit", "add"}:
            raise PolicyDenied(_WRAPPED_INDEX_WRITE)
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
    """Ordinary shell commands ask first. Local git writes do not ask again.

    A wrapper such as ``sh -c`` or ``env`` always asks.
    """

    segments = [segment for segment in iter_vet_segments(command) if segment.strip()]
    if not segments:
        return True
    for segment in segments:
        if unwrap_shell(_command_tokens(segment)) is not None:
            return True
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
    """Every segment ``vet`` checks: the chain, substitutions, and wrapped commands."""

    return [segment for segment, _wrapped in _walk_segments(command)]


def _walk_segments(command: str, depth: int = 0, wrapped: bool = False) -> list[tuple[str, bool]]:
    """``(segment, wrapped)`` pairs; ``wrapped`` is True inside a wrapper or a substitution.

    A bare ``NAME=value`` prefix is unwrapped for vetting but does not count as
    a wrapper: the shell backend still sees and rewrites the git command behind it.
    """

    found: list[tuple[str, bool]] = []
    for _operator, segment in split_command_chain(command):
        found.append((segment, wrapped))
        for inner in _substitutions(segment):
            # Substitution bodies only get shorter, so they need no depth limit.
            found.append((inner, True))
            found.extend(_walk_segments(inner, depth, True))
        tokens = _command_tokens(segment)
        inner = unwrap_shell(tokens)
        if inner is None or depth >= _MAX_UNWRAP_DEPTH:
            continue
        inner_wrapped = wrapped or not _is_assignment_prefix(tokens)
        found.append((inner, inner_wrapped))
        found.extend(_walk_segments(inner, depth + 1, inner_wrapped))
    return found


def _is_assignment_prefix(tokens: list[str]) -> bool:
    for token in tokens:
        if token not in _SHELL_KEYWORDS:
            return bool(_ASSIGNMENT.match(token))
    return False


def _command_tokens(segment: str) -> list[str]:
    """Tokens of ``segment``, keeping backslashes in an unquoted first word (a Windows path)."""

    words = segment.split(None, 1)
    if words and "\\" in words[0] and not any(q in words[0] for q in "'\""):
        rest = _tokens(words[1]) if len(words) > 1 else []
        return [words[0], *rest]
    return _tokens(segment)


def unwrap_shell(tokens: list[str]) -> str | None:
    """Return the command a wrapper such as ``sh -c``, ``eval`` or ``env`` runs, else None.

    Leading shell keywords (``if``, ``then``, ``do``, ...) are skipped first.
    Executable names match case-insensitively on their basename.
    """

    index = 0
    while index < len(tokens) and tokens[index] in _SHELL_KEYWORDS:
        index += 1
    tokens = tokens[index:]
    if not tokens:
        return None
    if _ASSIGNMENT.match(tokens[0]):
        rest = tokens[:]
        while rest and _ASSIGNMENT.match(rest[0]):
            rest.pop(0)
        return shlex.join(rest) if rest else None
    name = _command_name(tokens[0])
    args = tokens[1:]
    if name in _POSIX_SHELLS:
        return _unwrap_posix_shell(args)
    if name in {"cmd", "cmd.exe"}:
        return _unwrap_cmd(args)
    if name in _POWERSHELLS:
        return _unwrap_powershell(args)
    if name == "eval":
        return " ".join(args) or None
    if name == "env":
        return _unwrap_env(args)
    if name in _RUNNERS:
        rest = _skip_options(args, _RUNNERS[name])
        return shlex.join(rest) if rest else None
    return None


def _unwrap_posix_shell(args: list[str]) -> str | None:
    seen_c = False
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--":
            index += 1
            break
        if arg in {"-o", "+o", "-O", "+O", "--rcfile", "--init-file"}:
            index += 2
            continue
        if arg.startswith(("-", "+")) and len(arg) > 1:
            if not arg.startswith("--") and "c" in arg[1:]:
                seen_c = True
            index += 1
            continue
        break
    if not seen_c or index >= len(args):
        return None
    return args[index]


def _unwrap_cmd(args: list[str]) -> str | None:
    for index, arg in enumerate(args):
        if not arg.startswith("/"):
            return None
        switch = arg[:2].lower()
        if switch in {"/c", "/k"}:
            rest = ([arg[2:]] if len(arg) > 2 else []) + args[index + 1 :]
            return " ".join(rest) or None
    return None


def _unwrap_powershell(args: list[str]) -> str | None:
    for index, arg in enumerate(args):
        option = arg.lower().lstrip("-/")
        if arg[:1] in {"-", "/"} and (option == "c" or (len(option) >= 3 and "command".startswith(option))):
            return " ".join(args[index + 1 :]) or None
    return None


def _unwrap_env(args: list[str]) -> str | None:
    rest: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in {"-u", "--unset", "-C", "--chdir"}:
            index += 2
            continue
        if arg in {"-S", "--split-string"} and index + 1 < len(args):
            rest = _tokens(args[index + 1])
            index += 2
            break
        if arg.startswith("--split-string="):
            rest = _tokens(arg.split("=", 1)[1])
            index += 1
            break
        if arg.startswith("-") or _ASSIGNMENT.match(arg):
            index += 1
            continue
        break
    rest = rest + args[index:]
    return shlex.join(rest) if rest else None


def _skip_options(args: list[str], takes_value: frozenset[str]) -> list[str]:
    index = 0
    while index < len(args) and args[index].startswith("-"):
        if args[index] == "--":
            return args[index + 1 :]
        index += 2 if args[index] in takes_value else 1
    return args[index:]


def git_subcommand(segment: str) -> str | None:
    """Return the git subcommand for a segment, if it is a git invocation."""

    invocation = _git_invocation(_tokens(segment))
    if invocation is None:
        return None
    _subcommand, _rest = _split_git_args(invocation[1:])
    return _subcommand


_COMMAND_PREFIXES = frozenset(
    {"do", "then", "else", "elif", "if", "while", "until", "!", "{", "(", "time", "command",
     "exec", "nohup", "env", "sudo", "xargs", "timeout"}
)


def runs_git(segment: str) -> bool:
    """True when git is the command this segment runs, not just a word in it.

    Leading assignments, shell keywords and simple wrappers such as ``env``
    or ``command`` (with their options) may come first: ``echo git`` is false,
    ``FOO=1 env git status`` is true.
    """

    for token in _tokens(segment):
        name = _command_name(token)
        if name in {"git", "git.exe"}:
            return True
        if _ASSIGNMENT.match(token) or name in _COMMAND_PREFIXES or token.startswith("-") or token.isdigit():
            continue
        return False
    return False


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
    if posix:
        kept = _without_identity_env(kept)
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


def _without_identity_env(tokens: list[str]) -> list[str]:
    """Put ``env -u ...`` directly before the git executable.

    One process runs the whole script, so an earlier part of it may have set
    identity or config variables by any means; ``env`` drops them for the
    commit alone, and because it execs the real program a shell function or
    alias named ``git`` cannot stand in for the commit.
    """

    for index, token in enumerate(tokens):
        if _command_name(token) in {"git", "git.exe"}:
            return [*tokens[:index], *_ENV_WITHOUT_IDENTITY, *tokens[index:]]
    return tokens


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
    """True when ``text`` (lowercased) asks for ``verb`` as a git operation.

    ``push`` and ``pull`` are also ordinary English verbs, so an occurrence
    counts only in a git-like context (see ``_push_counts``, ``_pull_counts``).
    """

    for match in re.finditer(rf"\b{verb}\b", text):
        window = text[max(0, match.start() - 24) : match.start()]
        if _NEGATION.search(window):
            continue
        if re.search(r"\bgit\s+$", text[: match.start()]):
            return True
        after = text[match.end() :]
        sentence = _SENTENCE_END.split(after, maxsplit=1)[0]
        following = re.findall(r"[\w']+", sentence)[:6]
        counts = _push_counts(after, following) if verb == "push" else _pull_counts(verb, following)
        if counts:
            return True
    return False


def _push_counts(after: str, following: list[str]) -> bool:
    if {"into", "onto"} & set(following):
        return False
    if _PUSH_ENDING.match(after):
        return True
    return bool(_PUSH_OBJECTS & set(following))


def _pull_counts(verb: str, following: list[str]) -> bool:
    if verb == "pull" and following[:1] in (["request"], ["requests"]):
        return False
    return not ({"out", "into", "apart"} & set(following))


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
    """The ``$(...)`` and backtick bodies outside quotes, without the ones nested in them."""

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
            index = end + 1
            continue
        if char == "`":
            end = segment.find("`", index + 1)
            if end == -1:
                break
            inner = segment[index + 1 : end]
            found.append(inner)
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
