"""Background tasks: worker roles that run on a thread pool while the user keeps talking.

Each task has its own graph, its own checkpoint thread and its own guarded
shell backend, so it keeps the permissions of the turn that started it. The
graph is built on the pool thread, so starting a task never waits for it. A task
that needs a shell approval parks (``needs_approval``) instead of holding a
pool thread, and continues on the pool once the main loop resumes it.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass, field, replace
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool, tool
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from absolutcoder.git_policy import TurnFlags
from absolutcoder.render import message_text
from absolutcoder.shell import GuardedShellBackend
from absolutcoder.subagents import AgentDef
from absolutcoder.webtools import UNTRUSTED_CLOSE, UNTRUSTED_NOTE, UNTRUSTED_OPEN

_UNFINISHED = frozenset({"queued", "running", "needs_approval"})
_NOTIFY = frozenset({"done", "failed", "cancelled"})
_NOTICE_LIMIT = 4000


class TaskLimitReached(RuntimeError):
    """``start`` refused: too many tasks are already queued, running or waiting."""

    def __init__(self, unfinished: int, limit: int) -> None:
        super().__init__(f"{unfinished} background tasks are already running or waiting (limit {limit}).")
        self.unfinished = unfinished
        self.limit = limit


@dataclass
class TaskRecord:
    id: str
    agent: str
    prompt: str
    status: str
    result: str | None
    error: str | None
    thread_id: str


@dataclass
class _Task:
    record: TaskRecord
    role: AgentDef
    cancel: threading.Event = field(default_factory=threading.Event)
    backend: GuardedShellBackend | None = None
    graph: CompiledStateGraph | None = None
    future: Future | None = None
    interrupts: list = field(default_factory=list)
    quiet: bool = False  # cancelled by shutdown: no notice


class TaskManager:
    """Start, watch, park, resume and cancel background worker runs."""

    def __init__(
        self,
        build: Callable[[AgentDef, GuardedShellBackend], CompiledStateGraph],
        *,
        roles: Callable[[], list[AgentDef]],
        session_id: str,
        repo: Path,
        shell: str = "auto",
        max_workers: int = 4,
        on_event: Callable[[TaskRecord], None] | None = None,
    ) -> None:
        self._build = build
        self._roles = roles
        self.session_id = session_id
        self.repo = Path(repo)
        self.shell = shell
        self.max_workers = max_workers
        self.limit = 2 * max_workers  # unfinished tasks allowed at once
        self._on_event = on_event
        self._before_settle: Callable[[], None] | None = None  # test seam: after the stream, before settling
        self._lock = threading.Lock()
        self._changed = threading.Condition(self._lock)
        self._tasks: dict[str, _Task] = {}
        self._notices: list[str] = []
        self._count = 0
        self._closed = False
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="absolutcoder-task")

    # --- public interface ----------------------------------------------------

    def start(self, agent: str, prompt: str, *, flags: TurnFlags) -> TaskRecord:
        roles = {role.name: role for role in self._roles()}
        if agent not in roles:
            raise ValueError(f"Unknown worker {agent!r}. Available: {', '.join(roles)}")
        payload = {"messages": [{"role": "user", "content": prompt}]}
        with self._lock:
            if self._closed:
                raise RuntimeError("task manager is shut down")
            unfinished = sum(1 for task in self._tasks.values() if task.record.status in _UNFINISHED)
            if unfinished >= self.limit:
                raise TaskLimitReached(unfinished, self.limit)
            self._count += 1
            number = self._count
            record = TaskRecord(
                id=f"t{number}",
                agent=agent,
                prompt=prompt,
                status="queued",
                result=None,
                error=None,
                thread_id=f"{self.session_id}:task:{number}",
            )
            task = _Task(record=record, role=roles[agent])
            self._tasks[record.id] = task
            snapshot = replace(record)
        # Announced before the pool can report any later status.
        self._emit(snapshot)
        try:
            # Outside the lock: the constructor runs git subprocesses.
            backend = GuardedShellBackend(
                root_dir=self.repo, flags=flags, shell=self.shell, agent_label=f"task {record.id}"
            )
        except Exception as error:  # noqa: BLE001 - reported on the record, never raised
            self._finish(task, "failed", error=str(error))
            return self.get(record.id)
        with self._lock:
            task.backend = backend
            # cancel() or shutdown() may have settled it meanwhile; then it never starts.
            if task.record.status == "queued" and not task.cancel.is_set() and not self._closed:
                task.future = self._pool.submit(self._run, task, payload)
        return self.get(record.id)

    def get(self, task_id: str) -> TaskRecord:
        with self._lock:
            return replace(self._task(task_id).record)

    def list(self) -> list[TaskRecord]:
        with self._lock:
            return [replace(task.record) for task in self._tasks.values()]

    def backend_for(self, task_id: str) -> GuardedShellBackend:
        """The task's shell backend; KeyError for an unknown task or one whose backend could not be made."""

        with self._lock:
            backend = self._task(task_id).backend
        if backend is None:
            raise KeyError(task_id)
        return backend

    def cancel(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.record.status not in _UNFINISHED:
                return False
            task.cancel.set()
            immediate = _stop_waiting(task)
        if immediate:
            self._finish(task, "cancelled")
        return True

    def pending_approvals(self) -> list[tuple[TaskRecord, list]]:
        with self._lock:
            return [
                (replace(task.record), list(task.interrupts))
                for task in self._tasks.values()
                if task.record.status == "needs_approval"
            ]

    def resume(self, task_id: str, decisions: dict) -> None:
        with self._lock:
            task = self._task(task_id)
            if task.record.status != "needs_approval":
                raise ValueError(f"{task_id} is not waiting for approval")
            if self._closed:
                raise RuntimeError("task manager is shut down")
            task.interrupts = []
            task.record.status = "running"
            snapshot = replace(task.record)
            task.future = self._pool.submit(self._run, task, Command(resume=decisions))
            self._changed.notify_all()
        self._emit(snapshot)

    def drain_notices(self) -> list[str]:
        with self._lock:
            notices, self._notices = self._notices, []
        return notices

    def requeue_notices(self, notices: list[str]) -> None:
        """Put drained notices back at the front of the queue (a turn failed before delivering them)."""

        with self._lock:
            self._notices = list(notices) + self._notices

    def wait(self, task_id: str, timeout: float | None = None) -> TaskRecord:
        """Block until the task is not queued or running; return it as it is then."""

        with self._changed:
            task = self._task(task_id)
            self._changed.wait_for(
                lambda: task.record.status not in {"queued", "running"}, timeout=timeout
            )
            return replace(task.record)

    def shutdown(self, *, wait: bool = False) -> int:
        """Cancel every unfinished task and refuse new ones; return how many were cancelled."""

        immediate: list[_Task] = []
        with self._lock:
            self._closed = True
            cancelled = 0
            for task in self._tasks.values():
                if task.record.status not in _UNFINISHED:
                    continue
                cancelled += 1
                task.quiet = True
                task.cancel.set()
                if _stop_waiting(task):
                    immediate.append(task)
        for task in immediate:
            self._finish(task, "cancelled")
        self._pool.shutdown(wait=wait, cancel_futures=True)
        return cancelled

    # --- the run, on a pool thread -------------------------------------------

    def _run(self, task: _Task, payload) -> None:
        if task.cancel.is_set():
            self._finish(task, "cancelled")  # no-op when cancel() already finished it
            return
        with self._lock:
            if task.record.status not in {"queued", "running"}:
                return
            changed = task.record.status != "running"
            task.record.status = "running"
            snapshot = replace(task.record)
            self._changed.notify_all()
        if changed:
            self._emit(snapshot)
        config = {"configurable": {"thread_id": task.record.thread_id}}
        try:
            graph = self._graph_for(task)
            if graph is None:
                self._finish(task, "cancelled")
                return
            with closing(graph.stream(payload, config, stream_mode="updates")) as parts:
                for _part in parts:
                    if task.cancel.is_set():
                        break
            if task.cancel.is_set():
                self._finish(task, "cancelled")
                return
            if self._before_settle is not None:
                self._before_settle()
            state = graph.get_state(config)
            interrupts = _pending_interrupts(state)
            if interrupts:
                self._finish(task, "needs_approval", interrupts=interrupts)
                return
            self._finish(task, "done", result=_final_text(state.values.get("messages", [])))
        except Exception as error:  # noqa: BLE001 - a pool thread must never raise
            self._finish(task, "failed", error=str(error))

    def _graph_for(self, task: _Task) -> CompiledStateGraph | None:
        """Build the task's graph on first run (on this pool thread); None when cancelled meanwhile."""

        with self._lock:
            graph, backend = task.graph, task.backend
        if graph is None:
            graph = self._build(task.role, backend)
            with self._lock:
                task.graph = graph
        return None if task.cancel.is_set() else graph

    def _finish(
        self,
        task: _Task,
        status: str,
        *,
        result: str | None = None,
        error: str | None = None,
        interrupts: list | None = None,
    ) -> None:
        """Settle an unfinished task as ``status``: parked, or finished with a notice.

        Once ``cancel`` has returned True for a task its flag is set, and the
        flag is read here under the lock, so such a task always ends ``cancelled``.
        """

        with self._lock:
            if task.record.status not in _UNFINISHED:
                return
            if task.cancel.is_set():
                status, result, error, interrupts = "cancelled", None, None, None
            task.record.status = status
            task.record.result = result
            task.record.error = error
            task.interrupts = list(interrupts or [])
            snapshot = replace(task.record)
            if status in _NOTIFY and not task.quiet:
                self._notices.append(notice(snapshot))
            self._changed.notify_all()
        self._emit(snapshot)

    def _emit(self, record: TaskRecord) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(record)
        except Exception:  # noqa: BLE001, S110 - a display callback must not break a task
            pass

    def _task(self, task_id: str) -> _Task:
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(task_id)
        return task


def _stop_waiting(task: _Task) -> bool:
    """Drop a task that is not on a pool thread; True when it can be marked cancelled now.

    A queued or parked task is not running anything. A resumed task whose pool
    slot has not started yet is unsubmitted here, so it never starts.
    """

    unstarted = task.future is not None and task.future.cancel()
    return unstarted or task.record.status in {"queued", "needs_approval"}


def notice(record: TaskRecord) -> str:
    """The message the main model reads when a background task finishes."""

    return f"Background task {record.id} ({record.agent}) {record.status}.\n{_body(record)}"


def _body(record: TaskRecord) -> str:
    if record.status == "failed":
        text = f"Error: {record.error}"
    elif record.status == "cancelled":
        text = "Cancelled."
    else:
        text = record.result or ""
    if len(text) <= _NOTICE_LIMIT:
        return text
    kept = text[:_NOTICE_LIMIT] + "\n[truncated]"
    if kept.count(UNTRUSTED_OPEN) > kept.count(UNTRUSTED_CLOSE):
        # The cut fell inside a web-content block: close it again, so later
        # text never reads as part of the untrusted region.
        kept += "\n" + UNTRUSTED_CLOSE + "\n" + UNTRUSTED_NOTE
    return kept


def _pending_interrupts(state) -> list:
    found = []
    for item in getattr(state, "tasks", ()) or ():
        found.extend(getattr(item, "interrupts", ()) or ())
    return found


def _final_text(messages) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            return message_text(message.content)
    return ""


def make_task_tools(manager: TaskManager, current_flags: Callable[[], TurnFlags]) -> list[BaseTool]:
    """The main model's tools for background tasks: start, check, list and cancel."""

    @tool
    def start_task(agent: str, prompt: str) -> str:
        """Start a worker role in the background and return at once.

        Use it for work that can run while you keep talking with the user.
        ``agent`` is a worker name from the roster; ``prompt`` is the full task.
        """

        try:
            record = manager.start(agent, prompt, flags=current_flags())
        except Exception as error:  # noqa: BLE001 - tools answer with text, never raise
            return f"Error: {error}"
        return (
            f"Started background task {record.id} ({record.agent}). Its result will arrive in a "
            "later turn; use check_task to look earlier."
        )

    @tool
    def check_task(task_id: str) -> str:
        """Show a background task's status, and its result once it has finished."""

        try:
            record = manager.get(task_id)
        except KeyError:
            return f"Error: no background task {task_id}."
        line = f"{record.id} ({record.agent}): {record.status}"
        if record.status == "needs_approval":
            return line + "\nwaiting for the user's approval"
        if record.status in _NOTIFY:
            return line + "\n" + _body(record)
        return line

    @tool
    def list_tasks() -> str:
        """List every background task in this session with its status."""

        records = manager.list()
        if not records:
            return "No background tasks."
        return "\n".join(
            f"{r.id}  {r.agent}  {r.status}  {' '.join(r.prompt.split())[:60]}" for r in records
        )

    @tool
    def cancel_task(task_id: str) -> str:
        """Cancel a queued, running or waiting background task."""

        if manager.cancel(task_id):
            return f"Cancelled {task_id}."
        return f"Error: {task_id} is not running."

    return [start_task, check_task, list_tasks, cancel_task]
