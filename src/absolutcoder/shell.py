"""Shell backend that applies the git policy before anything runs."""

from __future__ import annotations

from pathlib import Path

from deepagents.backends.filesystem import FilesystemBackend, _raise_if_symlink_loop
from deepagents.backends.local_shell import LocalShellBackend
from deepagents.backends.protocol import ExecuteResponse

from absolutcoder.git_policy import (
    PolicyDenied,
    TurnFlags,
    git_subcommand,
    prepare_commit_segment,
    protect_context_file,
    split_command_chain,
    unstage_context_file,
    vet,
)

_IDENTITY_PREFIXES = ("GIT_AUTHOR_", "GIT_COMMITTER_")
_EXTENDED_PREFIX = "\\\\?\\"
_EXTENDED_UNC_PREFIX = "\\\\?\\UNC\\"


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
    """

    def __init__(
        self,
        root_dir,
        *,
        flags: TurnFlags | None = None,
        virtual_mode: bool = True,
        inherit_env: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            root_dir=root_dir,
            virtual_mode=virtual_mode,
            inherit_env=inherit_env,
            **kwargs,
        )
        self.cwd = plain_path(self.cwd)
        self.flags = flags or TurnFlags()
        self.vet_log: list[dict] = []
        self._env.setdefault("GIT_TERMINAL_PROMPT", "0")
        protect_context_file(self.cwd)

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        if not command or not isinstance(command, str):
            return super().execute(command, timeout=timeout)
        try:
            vet(command, self.flags)
        except PolicyDenied as exc:
            self.vet_log.append({"command": command, "allowed": False, "reason": str(exc)})
            raise
        self.vet_log.append({"command": command, "allowed": True})
        chain = split_command_chain(command)
        if any(operator == "|" for operator, _segment in chain):
            return self._execute_pipeline(chain, timeout=timeout)
        return self._execute_chain(chain, timeout=timeout)

    def _execute_chain(self, chain: list[tuple[str, str]], *, timeout: int | None) -> ExecuteResponse:
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
                result, created = self._execute_one(segment, timeout=timeout)
                if created is not None:
                    message_files.append(created)
                outputs.append(result.output)
                exit_code = 0 if result.exit_code is None else result.exit_code
                truncated = truncated or result.truncated
        finally:
            _delete(message_files)
        return ExecuteResponse(output="\n".join(outputs), exit_code=exit_code, truncated=truncated)

    def _execute_pipeline(self, chain: list[tuple[str, str]], *, timeout: int | None) -> ExecuteResponse:
        pieces: list[str] = []
        message_files: list[Path] = []
        if any(git_subcommand(segment) == "commit" for _operator, segment in chain):
            self._unstage()
        saved = self._pop_identity() if any(git_subcommand(segment) == "commit" for _op, segment in chain) else {}
        try:
            for index, (operator, segment) in enumerate(chain):
                if index:
                    pieces.append(f" {operator} ")
                if git_subcommand(segment) == "commit":
                    rewritten, path = prepare_commit_segment(segment, self.cwd)
                    pieces.append(rewritten)
                    if path is not None:
                        message_files.append(path)
                else:
                    pieces.append(segment)
            return super().execute("".join(pieces), timeout=timeout)
        finally:
            self._env.update(saved)
            _delete(message_files)

    def _execute_one(self, segment: str, *, timeout: int | None) -> tuple[ExecuteResponse, Path | None]:
        command = segment
        created: Path | None = None
        subcommand = git_subcommand(segment)
        saved: dict[str, str] = {}
        if subcommand == "commit":
            self._unstage()
            command, created = prepare_commit_segment(segment, self.cwd)
            saved = self._pop_identity()
        try:
            result = super().execute(command, timeout=timeout)
        finally:
            self._env.update(saved)
        if subcommand == "add":
            self._unstage()
        return result, created

    def _pop_identity(self) -> dict[str, str]:
        saved: dict[str, str] = {}
        for key in list(self._env):
            if key.startswith(_IDENTITY_PREFIXES):
                saved[key] = self._env.pop(key)
        return saved

    def _unstage(self) -> None:
        unstage_context_file(self.cwd, self._env)


def _delete(paths: list[Path]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)
