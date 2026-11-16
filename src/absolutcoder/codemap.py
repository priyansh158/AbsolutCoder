"""A local code map: Graphify parses the repository into a graph the model can ask first.

Graphify runs as ``python -m graphify`` in a subprocess, always with an
argument list and never through a shell, and with an environment that cannot
reach a model (code-only parsing). Builds and refreshes run on daemon threads;
callbacks are wrapped, state sits under one lock, and no callback runs while
the lock is held.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path

from langchain_core.tools import BaseTool, tool

CODE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".py", ".pyi", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".go", ".rs",
        ".java", ".kt", ".kts", ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh",
        ".cs", ".rb", ".php", ".swift", ".scala", ".lua", ".sh", ".bash", ".ps1",
        ".m", ".mm", ".dart", ".ex", ".exs", ".erl", ".hs", ".ml", ".clj", ".r",
        ".jl", ".zig", ".vue", ".svelte", ".sql",
    }
)  # fmt: skip

MAP_TOOL_NAMES = ("code_map_query", "code_map_path", "code_map_explain")

CODE_MAP_RULE = (
    "A code map of this repository is available. Before your first search or source-file read "
    "on a question about existing code, call code_map_query with the question; use code_map_path "
    "to see how two things connect and code_map_explain for one symbol. Then read only the files "
    "it points to. Put this rule in the brief of any worker that explores code."
)

_SKIP_DIRS = frozenset(
    {"node_modules", "venv", "env", "__pycache__", "dist", "build", "target", "graphify-out"}
)
_MAX_ENTRIES = 20_000
_MODEL_ENDPOINTS = frozenset(
    {"OPENAI_BASE_URL", "ANTHROPIC_BASE_URL", "OLLAMA_BASE_URL", "AZURE_OPENAI_ENDPOINT"}
)
_BUILD_TIMEOUT = 600
_REFRESH_TIMEOUT = 300
_ANSWER_TIMEOUT = 60
_MIN_BUDGET, _MAX_BUDGET, _DEFAULT_BUDGET = 200, 8000, 2000

_NOT_READY = "The code map is {status}. Search and read files directly for now."
_NO_QUESTION = "Error: give a question in plain words."
_NO_NAME = "Error: give a symbol or file name in plain words."
_AMBIGUOUS = "That name is ambiguous. Candidates:"
_NO_MATCH = "No match in the code map. Search with grep or glob instead."
_TIMED_OUT = "Code map timed out. Search and read files directly."


class CodeMap:
    """One repository's Graphify graph: its status, background builds, and answers."""

    def __init__(
        self,
        repo: Path | str,
        *,
        min_files: int = 20,
        runner: Callable[..., subprocess.CompletedProcess] | None = None,
        python: str | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        self.repo = Path(repo).resolve()
        self.min_files = min_files
        self.graph_path = self.repo / "graphify-out" / "graph.json"
        self._runner = runner if runner is not None else subprocess.run
        self._injected = runner is not None
        self._python = python or sys.executable
        self._on_status = on_status
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)
        self._building = False
        self._refreshing = False
        self._refresh_again = False
        self._threads = 0  # build/refresh threads still running, callbacks included
        self._failure: str | None = None
        self._enough_files = False  # cached once true; a repository rarely shrinks
        self._closed = False

    # --- status ---------------------------------------------------------------

    def status(self) -> str:
        with self._lock:
            return self._status_locked()

    def usable(self) -> bool:
        return self.status() in ("ready", "building")

    def _status_locked(self) -> str:
        if not self._injected and find_spec("graphify") is None:
            return "unavailable"
        if self._building:
            return "building"
        if self._failure is not None:
            return f"failed: {self._failure}"
        if self.graph_path.exists():
            return "ready"
        if not self._has_enough_code():
            return "too small"
        return "missing"

    def _has_enough_code(self) -> bool:
        """Walk the repository until ``min_files`` code files are found (bounded)."""

        if self._enough_files:
            return True
        found = visited = 0
        pending = [self.repo]
        while pending:
            try:
                with os.scandir(pending.pop()) as entries:
                    for entry in entries:
                        visited += 1
                        if visited > _MAX_ENTRIES:
                            return False
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                if not entry.name.startswith(".") and entry.name not in _SKIP_DIRS:
                                    pending.append(Path(entry.path))
                                continue
                        except OSError:
                            continue
                        if os.path.splitext(entry.name)[1].lower() in CODE_EXTENSIONS:
                            found += 1
                            if found >= self.min_files:
                                self._enough_files = True
                                return True
            except OSError:
                continue
        return False

    # --- background work --------------------------------------------------------

    def ensure(self, *, force: bool = False) -> None:
        """Start one background build when the map is missing (or, forced, small or failed)."""

        with self._lock:
            if self._closed or self._building:
                return
            status = self._status_locked()
            wanted = status == "missing" or (
                force and (status == "too small" or status.startswith("failed: "))
            )
            if not wanted:
                return
            self._building = True
            self._failure = None
            self._threads += 1
        self._start(self._build, "absolutcoder-codemap-build")

    def refresh(self) -> None:
        """Update a ready map in the background; a request during a refresh runs once more after it."""

        with self._lock:
            if self._closed or self._building:
                return
            if self._refreshing:
                self._refresh_again = True
                return
            if self._status_locked() != "ready":
                return
            self._refreshing = True
            self._threads += 1
        self._start(self._refresh, "absolutcoder-codemap-refresh")

    def wait(self, timeout: float | None = None) -> str:
        """Block until no build or refresh is running; return the status then."""

        with self._idle:
            self._idle.wait_for(lambda: self._threads == 0, timeout=timeout)
        return self.status()

    def close(self) -> None:
        """No new builds or refreshes start; running ones finish on their daemon threads."""

        with self._lock:
            self._closed = True
            self._refresh_again = False

    def _start(self, target: Callable[[], None], name: str) -> None:
        try:
            threading.Thread(target=target, name=name, daemon=True).start()
        except Exception as error:  # noqa: BLE001 - reported as a status, never raised
            with self._lock:
                if self._building:
                    self._building = False
                    self._failure = _reason(str(error) or type(error).__name__)
                self._refreshing = False
                self._threads -= 1
                self._idle.notify_all()

    def _build(self) -> None:
        try:
            self._notify("building")
            try:
                result = self._graphify(["extract", ".", "--code-only"], timeout=_BUILD_TIMEOUT)
                if result.returncode == 0 and self.graph_path.exists():
                    failure = None
                else:
                    failure = _reason(result.stderr) or (
                        f"exit code {result.returncode}"
                        if result.returncode
                        else "no graph was written"
                    )
            except Exception as error:  # noqa: BLE001 - a build thread must never raise
                failure = _reason(str(error)) or type(error).__name__
            with self._lock:
                self._building = False
                self._failure = failure
            self._notify("ready" if failure is None else f"failed: {failure}")
        finally:
            self._thread_done()

    def _refresh(self) -> None:
        try:
            while True:
                try:
                    self._graphify(["update", "."], timeout=_REFRESH_TIMEOUT)
                except Exception:  # noqa: BLE001, S110 - the old graph still answers
                    pass
                with self._lock:
                    if self._refresh_again and not self._closed:
                        self._refresh_again = False
                        continue
                    self._refreshing = False
                    self._refresh_again = False
                    return
        finally:
            self._thread_done()

    def _thread_done(self) -> None:
        with self._lock:
            self._threads -= 1
            self._idle.notify_all()

    def _notify(self, status: str) -> None:
        if self._on_status is None:
            return
        try:
            self._on_status(status)
        except Exception:  # noqa: BLE001, S110 - a display callback must not break a build
            pass

    # --- answers ------------------------------------------------------------------

    def query(self, question: str, budget: int = _DEFAULT_BUDGET) -> str:
        status = self.status()
        if status != "ready":
            return _NOT_READY.format(status=status)
        text = _clean(question)
        if not text:
            return _NO_QUESTION
        size = _clamp_budget(budget)
        args = ["query", text, "--budget", str(size), "--graph", str(self.graph_path)]
        return self._answer(args, limit=size * 4 + 2000)

    def path(self, source: str, target: str) -> str:
        status = self.status()
        if status != "ready":
            return _NOT_READY.format(status=status)
        start, end = _clean(source), _clean(target)
        if not start or not end:
            return _NO_NAME
        return self._answer(["path", start, end, "--graph", str(self.graph_path)])

    def explain(self, name: str) -> str:
        status = self.status()
        if status != "ready":
            return _NOT_READY.format(status=status)
        text = _clean(name)
        if not text:
            return _NO_NAME
        return self._answer(["explain", text, "--graph", str(self.graph_path)])

    def _answer(self, args: list[str], *, limit: int = _DEFAULT_BUDGET * 4 + 2000) -> str:
        try:
            result = self._graphify(args, timeout=_ANSWER_TIMEOUT)
        except subprocess.TimeoutExpired:
            return _TIMED_OUT
        except Exception as error:  # noqa: BLE001 - tools answer with text, never raise
            return f"Code map error: {error}"
        stdout = result.stdout or ""
        if result.returncode == 0:
            return _cut(stdout, limit)
        if result.returncode == 3:
            return _cut(_AMBIGUOUS + "\n" + stdout, limit)
        if result.returncode == 4:
            return _NO_MATCH
        return f"Code map error: {(result.stderr or '').strip()[:300]}"

    def _graphify(self, args: list[str], *, timeout: int) -> subprocess.CompletedProcess:
        """The one place Graphify runs: an argument list, no shell, no model access."""

        return self._runner(
            [self._python, "-m", "graphify", *args],
            cwd=str(self.repo),
            env=_offline_env(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            shell=False,
            stdin=subprocess.DEVNULL,
        )


def _offline_env() -> dict[str, str]:
    """A copy of the environment with model keys and endpoints removed, Graphify kept offline."""

    env = {
        name: value
        for name, value in os.environ.items()
        if not (
            name.upper().endswith("_API_KEY")
            or name.upper() in _MODEL_ENDPOINTS
            or name.upper().startswith("AWS_")
        )
    }
    env["GRAPHIFY_BACKEND"] = "none"
    env["GRAPHIFY_QUERY_LOG_DISABLE"] = "1"
    return env


def _clean(text) -> str:
    """Strip whitespace and leading dashes, so model text is never read as an option."""

    return re.sub(r"^[\s-]+", "", str(text or "")).strip()


def _clamp_budget(budget) -> int:
    try:
        size = int(budget)
    except (TypeError, ValueError):
        size = _DEFAULT_BUDGET
    return max(_MIN_BUDGET, min(_MAX_BUDGET, size))


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n[truncated]"


def _reason(text: str | None) -> str:
    return (text or "").strip()[:200]


def make_map_tools(code_map: CodeMap) -> list[BaseTool]:
    """The model's code map tools, in ``MAP_TOOL_NAMES`` order."""

    @tool(
        description=(
            "Ask the code map where something is or how it works. Returns matching symbols with "
            "file and line. Use before grep or reading files."
        )
    )
    def code_map_query(question: str, budget: int = _DEFAULT_BUDGET) -> str:
        return code_map.query(question, budget)

    @tool(description="Show how two symbols or files are connected.")
    def code_map_path(source: str, target: str) -> str:
        return code_map.path(source, target)

    @tool(description="Show one symbol: where it is defined, what it calls and what calls it.")
    def code_map_explain(name: str) -> str:
        return code_map.explain(name)

    return [code_map_query, code_map_path, code_map_explain]
