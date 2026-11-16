"""Shell backend that applies the git policy before anything runs.

Every command runs as one process in a known shell (``find_shell``), with an
environment copied for that call. Git commands on one repository take turns.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
from contextlib import nullcontext
from pathlib import Path

from deepagents.backends.filesystem import FilesystemBackend, _raise_if_symlink_loop
from deepagents.backends.local_shell import LocalShellBackend
from deepagents.backends.protocol import ExecuteResponse

from absolutcoder.git_policy import (
    CONTEXT_FILENAME,
    PolicyDenied,
    TurnFlags,
    git_subcommand,
    prepare_commit_segment,
    protect_context_file,
    render_command,
    runs_git,
    split_command_chain,
    unstage_context_file,
    vet,
)

_IDENTITY_PREFIXES = ("GIT_AUTHOR_", "GIT_COMMITTER_")
_EXTENDED_PREFIX = "\\\\?\\"
_EXTENDED_UNC_PREFIX = "\\\\?\\UNC\\"
_GIT_MARKERS = ("cmd/git.exe", "bin/git.exe", "mingw64/bin/git.exe")
_MENTIONS_GIT = re.compile(r"(?<![\w.-])git(?:\.exe)?(?![\w-])", re.IGNORECASE)
_LEADING_KEYWORDS = re.compile(r"^((?:(?:do|then|else|elif|if|while|until|!|\{)\s+)*)(.*)$", re.DOTALL)
_HEREDOC = re.compile(r"<<(?!<)")
_JOINERS = {"&&": " && ", "||": " || ", ";": "; ", "|": " | ", "\n": "\n"}

_REPO_LOCKS: dict[str, threading.RLock] = {}
_REPO_LOCKS_GUARD = threading.Lock()


class ShellNotFound(ValueError):
    """The configured shell setting names no usable shell."""


def find_shell(preference: str = "auto") -> tuple[str, list[str]]:
    """Return ``(kind, argv_prefix)`` for ``preference``; the script is appended to the prefix.

    ``kind`` is ``bash``, ``sh`` or ``cmd``. On Windows ``auto`` and ``bash``
    mean Git for Windows' own bash, never WSL's ``System32\\bash.exe``.
    """

    choice = (preference or "auto").strip()
    lowered = choice.lower()
    windows = os.name == "nt"
    if lowered == "auto":
        bash = _git_shell("bash") if windows else shutil.which("bash")
        if bash:
            return "bash", [bash, "-c"]
        return ("cmd", _cmd_argv()) if windows else ("sh", ["/bin/sh", "-c"])
    if lowered == "cmd":
        if not windows:
            raise ShellNotFound("cmd is only available on Windows")
        return "cmd", _cmd_argv()
    if lowered in {"bash", "sh"}:
        if windows:
            found = _git_shell(lowered)
        else:
            found = shutil.which(lowered) or ("/bin/sh" if lowered == "sh" and Path("/bin/sh").is_file() else None)
        if not found:
            where = " (looked for Git for Windows' own copy)" if windows else " on PATH"
            raise ShellNotFound(f"{lowered} was not found{where}")
        return lowered, [found, "-c"]
    path = Path(choice).expanduser()
    if not path.is_file():
        raise ShellNotFound(f"{choice!r} is not bash, sh, cmd or an existing shell executable")
    kind = "bash" if path.stem.lower() == "bash" else "sh"
    return kind, [str(path), "-c"]


def _cmd_argv() -> list[str]:
    return [os.environ.get("ComSpec") or "cmd.exe", "/d", "/s", "/c"]


def _git_shell(name: str) -> str | None:
    """Find ``bin\\<name>.exe`` or ``usr\\bin\\<name>.exe`` in the Git for Windows install."""

    git = shutil.which("git")
    if not git:
        return None
    for parent in Path(git).resolve().parents:
        if not any((parent / marker).is_file() for marker in _GIT_MARKERS):
            continue
        for candidate in (parent / "bin" / f"{name}.exe", parent / "usr" / "bin" / f"{name}.exe"):
            if candidate.is_file() and "system32" not in (p.lower() for p in candidate.parts):
                return str(candidate)
    return None


def repo_lock(repo: Path | str) -> threading.RLock:
    """One lock per resolved repository path, shared by every backend in the process."""

    key = os.path.normcase(str(Path(repo).resolve()))
    with _REPO_LOCKS_GUARD:
        return _REPO_LOCKS.setdefault(key, threading.RLock())


def strip_extended_prefix(path: str) -> str:
    """Drop a Windows extended-length prefix from ``path``.

    ``\\\\?\\C:\\repo\\file`` becomes ``C:\\repo\\file``.
    ``\\\\?\\UNC\\server\\share\\file`` becomes ``\\\\server\\share\\file``.
    Slashes may point either way. Any other path is returned unchanged.
    """

    if not path:
        return path
    normalized = path.replace("/", "\\")
    if normalized.startswith(_EXTENDED_UNC_PREFIX):
        return "\\\\" + normalized[len(_EXTENDED_UNC_PREFIX):]
    if normalized.startswith(_EXTENDED_PREFIX):
        return normalized[len(_EXTENDED_PREFIX):]
    return path


def plain_path(path: Path) -> Path:
    """Return ``path`` without a Windows extended-length prefix."""

    text = str(path)
    stripped = strip_extended_prefix(text)
    if stripped == text:
        return path
    return Path(stripped)


def _anchor_differs(path: Path, root: Path) -> bool:
    path_anchor = path.anchor
    root_anchor = root.anchor
    if not path_anchor or not root_anchor:
        return False
    return path_anchor.casefold() != root_anchor.casefold()


class _ExtendedPathMixin:
    """Normalize ``\\\\?\\`` paths before deepagents' root check.

    ``Path.resolve`` keeps the prefix when the caller already supplied it, and
    ``relative_to`` then treats ``\\\\?\\C:\\repo\\file`` as outside ``C:\\repo``.
    """

    cwd: Path
    virtual_mode: bool

    def _resolve_path(self, key: str) -> Path:
        key = strip_extended_prefix(key)
        if not self.virtual_mode:
            return self._resolve_unrestricted(key)
        virtual = key if key.startswith("/") else "/" + key
        if ".." in virtual or virtual.startswith("~"):
            raise ValueError("Path traversal not allowed")
        candidate = plain_path(self.cwd / virtual.lstrip("/"))
        root = plain_path(self.cwd)
        if candidate.is_absolute() and _anchor_differs(candidate, root):
            raise ValueError(f"Path:{candidate} outside root directory: {self.cwd}")
        full = plain_path(candidate.resolve())
        try:
            full.relative_to(root)
        except ValueError:
            raise ValueError(f"Path:{full} outside root directory: {self.cwd}") from None
        _raise_if_symlink_loop(full)
        return full

    def _resolve_unrestricted(self, key: str) -> Path:
        path = Path(key)
        if path.is_absolute():
            plain = plain_path(path)
            _raise_if_symlink_loop(plain)
            return plain
        resolved = plain_path((self.cwd / path).resolve())
        _raise_if_symlink_loop(resolved)
        return resolved

    def _to_virtual_path(self, path: Path) -> str:
        root = plain_path(self.cwd)
        plain_path(path.resolve()).relative_to(root)
        return "/" + plain_path(path).relative_to(root).as_posix()


class RepoFilesystemBackend(_ExtendedPathMixin, FilesystemBackend):
    """Filesystem route whose root check understands Windows extended paths."""

    def __init__(
        self,
        root_dir: str | Path | None = None,
        virtual_mode: bool = True,
        max_file_size_mb: int = 10,
    ) -> None:
        super().__init__(
            root_dir=root_dir,
            virtual_mode=virtual_mode,
            max_file_size_mb=max_file_size_mb,
        )
        self.cwd = plain_path(self.cwd)


class GuardedShellBackend(_ExtendedPathMixin, LocalShellBackend):
    """Local shell whose git commands are vetted for the current user turn.

    ``virtual_mode`` keeps filesystem tools inside the repo. It does not
    contain the shell, so ``vet`` runs before every ``execute``. Windows
    extended paths are normalized before the root check.

    One backend may be shared by workers running in parallel threads, so
    ``execute`` never mutates ``self._env`` and ``vet_log`` appends are locked.
    """

    def __init__(
        self,
        root_dir,
        *,
        flags: TurnFlags | None = None,
        virtual_mode: bool = True,
        inherit_env: bool = True,
        shell: str = "auto",
        agent_label: str = "main",
        **kwargs,
    ) -> None:
        self.shell_kind, self._argv_prefix = find_shell(shell)
        super().__init__(
            root_dir=root_dir,
            virtual_mode=virtual_mode,
            inherit_env=inherit_env,
            **kwargs,
        )
        self.cwd = plain_path(self.cwd)
        self.flags = flags or TurnFlags()
        self.agent_label = agent_label
        self.vet_log: list[dict] = []
        self._log_lock = threading.Lock()
        self._env.setdefault("GIT_TERMINAL_PROMPT", "0")
        protect_context_file(self.cwd)
        self.git_lock = repo_lock(self._repository_top())

    def _repository_top(self) -> Path:
        """The work tree that holds ``self.cwd``, or ``self.cwd`` when it is not a repository."""

        try:
            found = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=self.cwd,
                env=self._env,
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                check=False,
            )
        except OSError:
            return self.cwd
        top = found.stdout.strip()
        return Path(top) if found.returncode == 0 and top else self.cwd

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        if not command or not isinstance(command, str):
            return super().execute(command, timeout=timeout)
        flags = self.flags  # read once: another thread may start a new turn
        try:
            vet(command, flags)
        except PolicyDenied as exc:
            self._log({"command": command, "allowed": False, "reason": str(exc)})
            raise
        self._log({"command": command, "allowed": True})
        effective = timeout if timeout is not None else self._default_timeout
        if effective <= 0:
            raise ValueError(f"timeout must be positive, got {effective}")
        env = dict(self._env)
        if _MENTIONS_GIT.search(command):
            for key in [key for key in env if key.upper().startswith(_IDENTITY_PREFIXES)]:
                del env[key]
        chain = split_command_chain(command)
        subcommands = [git_subcommand(segment) for _op, segment in chain]
        writes_index = any(subcommand in {"commit", "add"} for subcommand in subcommands)
        uses_repo = any(runs_git(segment) for _op, segment in chain)
        with self.git_lock if uses_repo else nullcontext():
            if not writes_index:
                return self._run(command, env, timeout)
            if self.shell_kind == "cmd":
                if any(operator == "|" for operator, _segment in chain):
                    return self._execute_pipeline(chain, env, timeout)
                return self._execute_chain(chain, env, timeout)
            script, message_files = self._posix_script(chain)
            try:
                return self._run(script, env, timeout)
            finally:
                _delete(message_files)

    def _log(self, entry: dict) -> None:
        with self._log_lock:
            self.vet_log.append(entry)

    # --- POSIX shells: one script --------------------------------------------

    def _posix_script(self, chain: list[tuple[str, str]]) -> tuple[str, list[Path]]:
        """Rebuild the chain with AbsolutCoder.md unstaged around each add and commit."""

        unstage = self._unstage_script()
        pieces: list[str] = []
        message_files: list[Path] = []
        for index, (operator, segment) in enumerate(chain):
            if index:
                pieces.append(_JOINERS[operator])
            subcommand = git_subcommand(segment)
            if subcommand not in {"commit", "add"}:
                pieces.append(segment)
                continue
            lead, body = _LEADING_KEYWORDS.match(segment).groups()
            body = _foreground(body)
            if subcommand == "commit":
                rewritten, path = prepare_commit_segment(body, self.cwd, self.shell_kind)
                if path is not None:
                    message_files.append(path)
                # A here-document body starts on the next line, so the brace stays on this one.
                close = "; }" if _HEREDOC.search(rewritten) else "\n}"
                pieces.append(f"{lead}{{ {unstage}; {rewritten}{close}")
            else:
                pieces.append(f"{lead}{{ {body}\n__ac_rc=$?; {unstage}; (exit $__ac_rc)\n}}")
        return "".join(pieces), message_files

    def _unstage_script(self) -> str:
        """The shell form of ``unstage_context_file``: silent, never fails.

        ``env`` runs the real git even if the script defined a ``git`` function.
        """

        repo = str(self.cwd)
        restore = ["env", "git", "-C", repo, "restore", "--staged", "--", CONTEXT_FILENAME]
        remove = ["env", "git", "-C", repo, "rm", "--cached", "--ignore-unmatch", "-q", "--", CONTEXT_FILENAME]
        restore_text = render_command(restore, self.shell_kind)
        remove_text = render_command(remove, self.shell_kind)
        return f"{{ {restore_text} || {remove_text}; }} >/dev/null 2>&1 || :"

    # --- cmd: commit and add run step by step --------------------------------

    def _execute_chain(self, chain: list[tuple[str, str]], env: dict[str, str], timeout: int | None) -> ExecuteResponse:
        outputs: list[str] = []
        exit_code = 0
        truncated = False
        message_files: list[Path] = []
        try:
            for index, (operator, segment) in enumerate(chain):
                if index and operator == "&&" and exit_code != 0:
                    break
                if index and operator == "||" and exit_code == 0:
                    break
                result, created = self._execute_one(segment, env, timeout)
                if created is not None:
                    message_files.append(created)
                outputs.append(result.output)
                exit_code = 0 if result.exit_code is None else result.exit_code
                truncated = truncated or result.truncated
        finally:
            _delete(message_files)
        return ExecuteResponse(output="\n".join(outputs), exit_code=exit_code, truncated=truncated)

    def _execute_pipeline(self, chain: list[tuple[str, str]], env: dict[str, str], timeout: int | None) -> ExecuteResponse:
        pieces: list[str] = []
        message_files: list[Path] = []
        if any(git_subcommand(segment) == "commit" for _operator, segment in chain):
            unstage_context_file(self.cwd, env)
        try:
            for index, (operator, segment) in enumerate(chain):
                if index:
                    pieces.append(f" {operator} ")
                if git_subcommand(segment) == "commit":
                    rewritten, path = prepare_commit_segment(segment, self.cwd, "cmd")
                    pieces.append(rewritten)
                    if path is not None:
                        message_files.append(path)
                else:
                    pieces.append(segment)
            return self._run("".join(pieces), env, timeout)
        finally:
            _delete(message_files)

    def _execute_one(self, segment: str, env: dict[str, str], timeout: int | None) -> tuple[ExecuteResponse, Path | None]:
        command = segment
        created: Path | None = None
        subcommand = git_subcommand(segment)
        if subcommand == "commit":
            unstage_context_file(self.cwd, env)
            command, created = prepare_commit_segment(segment, self.cwd, "cmd")
        result = self._run(command, env, timeout)
        if subcommand == "add":
            unstage_context_file(self.cwd, env)
        return result, created

    # --- one process ---------------------------------------------------------

    def _run(self, script: str, env: dict[str, str], timeout: int | None) -> ExecuteResponse:
        """Run ``script`` once and answer exactly as ``LocalShellBackend.execute`` does."""

        effective = timeout if timeout is not None else self._default_timeout
        if self.shell_kind == "cmd" and os.name == "nt":
            # cmd /s /c strips the outer quotes and reads the rest verbatim;
            # list2cmdline's backslash escapes would reach cmd unchanged.
            args: str | list[str] = subprocess.list2cmdline(self._argv_prefix) + ' "' + script + '"'
        else:
            args = [*self._argv_prefix, script]
        try:
            result = subprocess.run(  # noqa: S603 - the shell is the tool being offered
                args,
                check=False,
                capture_output=True,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=effective,
                env=env,
                cwd=str(self.cwd),
                start_new_session=(sys.platform != "win32"),
            )
        except subprocess.TimeoutExpired:
            if timeout is not None:
                msg = f"Error: Command timed out after {effective} seconds (custom timeout). The command may be stuck or require more time."
            else:
                msg = f"Error: Command timed out after {effective} seconds. For long-running commands, re-run using the timeout parameter."
            return ExecuteResponse(output=msg, exit_code=124, truncated=False)
        except Exception as error:  # noqa: BLE001 - same contract as LocalShellBackend
            return ExecuteResponse(
                output=f"Error executing command ({type(error).__name__}): {error}",
                exit_code=1,
                truncated=False,
            )
        return _response(result, self._max_output_bytes)


def _foreground(segment: str) -> str:
    """Drop a trailing shell comment and a trailing ``&`` from an add or commit segment.

    The rebuilt group must stay valid shell, and a backgrounded commit would
    race with the deletion of its message file.
    """

    quote = ""
    cut = len(segment)
    index = 0
    while index < len(segment):
        char = segment[index]
        if quote:
            if char == "\\" and quote == '"':
                index += 2
                continue
            if char == quote:
                quote = ""
        elif char == "\\":
            index += 2
            continue
        elif char in {"'", '"'}:
            quote = char
        elif char == "#" and (index == 0 or segment[index - 1].isspace()):
            cut = index
            break
        index += 1
    text = segment[:cut].rstrip()
    if text.endswith("&") and not text.endswith(("&&", ">&", "<&")):
        text = text[:-1].rstrip()
    return text


def _response(result: subprocess.CompletedProcess, max_output_bytes: int) -> ExecuteResponse:
    """Assemble output the way ``LocalShellBackend.execute`` does."""

    parts: list[str] = []
    if result.stdout:
        parts.append(result.stdout)
    if result.stderr:
        parts.extend(f"[stderr] {line}" for line in result.stderr.strip().split("\n"))
    output = "\n".join(parts) if parts else "<no output>"
    truncated = False
    encoded = output.encode("utf-8")
    if len(encoded) > max_output_bytes:
        output = encoded[:max_output_bytes].decode("utf-8", errors="ignore")
        output += f"\n\n... Output truncated at {max_output_bytes} bytes."
        truncated = True
    if result.returncode != 0:
        output = f"{output.rstrip()}\n\nExit code: {result.returncode}"
    return ExecuteResponse(output=output, exit_code=result.returncode, truncated=truncated)


def _delete(paths: list[Path]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)
