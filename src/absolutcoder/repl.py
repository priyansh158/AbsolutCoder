"""Terminal loop for AbsolutCoder."""

from __future__ import annotations

import inspect
import os
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path

from langgraph.errors import GraphBubbleUp
from langgraph.types import Command
from rich.console import Console

from absolutcoder.agent import build_agent, build_role_graph
from absolutcoder.agent_files import load_roster
from absolutcoder.attachments import build_user_content
from absolutcoder.config import Config, config_path, resolve_model, save_config
from absolutcoder.git_policy import PolicyDenied, TurnFlags, protect_context_file
from absolutcoder.loops import VerifyOutcome, parse_interval, run_interval_loop, run_until_verified
from absolutcoder.models import ModelFactory, ProviderMissing, missing_keys
from absolutcoder.prompting import background_output, make_input_fn
from absolutcoder.render import StreamRenderer, output_lock
from absolutcoder.sessions import (
    Session,
    checkpoint_saver,
    create_session,
    list_sessions,
    load_session,
    update_session,
)
from absolutcoder.shell import GuardedShellBackend, ShellNotFound
from absolutcoder.skills import global_skills_dir, install_skill, list_skills, remove_skill, skill_body
from absolutcoder.tasks import TaskManager, TaskRecord, make_task_tools
from absolutcoder.webtools import make_web_tools

_COMMANDS = "Commands: /model, /subagent, /auto, /resume, /attach, /agents, /tasks, /skills, /verify, /loop, /exit, /quit"
_SKILLS_ADD_USAGE = "Usage: /skills add <folder>"
_SKILLS_REMOVE_USAGE = "Usage: /skills remove <name>"
_SKILLS_USAGE = "Usage: /skills [add <folder> | remove <name>]"
_BUILTIN_COMMANDS = frozenset(
    {"/model", "/subagent", "/auto", "/resume", "/attach", "/agents", "/tasks", "/skills",
     "/verify", "/loop", "/exit", "/quit"}
)
_LOOP_USAGE = "Usage: /loop <N>(s|m|h) <prompt>"
_TASKS_USAGE = "Usage: /tasks [show <id> | cancel <id>]"
_TASKS_OFF = "Background tasks are off in this mode."
_UPDATES_HEADER = "[Background task updates — reported by the program, not typed by the user]"
_REJECTED = {"type": "reject", "message": "User rejected the command."}
_NO_TERMINAL = "No terminal to ask on: the command was rejected. Use --auto to allow commands without asking."
_FAILURE_TAIL_LINES = 15
_WAIT_STEP = 0.2  # seconds per join while leaving, so Ctrl+C is seen promptly


class Repl:
    """Stream the main agent, print each delegated task, and gate the shell."""

    def __init__(
        self,
        repo: Path | str,
        config: Config,
        *,
        chat_model=None,
        model_factory: ModelFactory | None = None,
        input_fn=None,
        console: Console | None = None,
        sleep_fn=time.sleep,
        background: bool = True,
        overrides: Mapping[str, object] | None = None,
    ) -> None:
        self.repo = Path(repo).resolve()
        # Settings for this run only (--auto, --until, --max-rounds, --until-timeout).
        self.overrides: dict[str, object] = dict(overrides or {})
        self.saved_config = replace(
            config,
            main_model=resolve_model(config.main_model),
            subagent_model=resolve_model(config.subagent_model),
            agent_models={k: resolve_model(v) for k, v in config.agent_models.items()},
        )
        self.chat_model = chat_model
        self.model_factory = model_factory
        self.input_fn = input_fn or input
        self.console = console or Console()
        # Where "Error:" lines go; a one-shot run points it at stderr.
        self.error_console = self.console
        self.sleep_fn = sleep_fn
        self.pending_attachments: list[Path] = []
        self.last_turn_interrupted = False
        self.last_turn_failed = False
        self._no_terminal_noted = False
        self._task_stopped = False
        protect_context_file(self.repo)
        self.backend = GuardedShellBackend(
            root_dir=self.repo, virtual_mode=True, inherit_env=True, shell=self.config.shell
        )
        self.checkpointer = checkpoint_saver()
        self.turn_vet_log: list[dict] = []
        self._renderer: StreamRenderer | None = None
        self.agents, warnings = load_roster(self.repo)
        # Built once and reused by every rebuild, so no rebuild leaves a client behind.
        self.web_tools = make_web_tools(max_chars=self.config.web_max_chars)
        self._output = output_lock(self.console)  # shared with every turn's renderer
        self._closed: bool | None = None  # close() result once it has run
        # The manager and its tools exist before the first build so every build,
        # and every rebuild, gets the same tool objects; it learns the session id
        # once the session row exists, before any task can start.
        self.tasks: TaskManager | None = None
        self._task_tools: list = []
        if background:
            self.tasks = TaskManager(
                self._build_task_graph,
                roles=lambda: self.agents,
                session_id="",
                repo=self.repo,
                shell=self.config.shell,
                max_workers=max(1, self.config.max_background_tasks),
                on_event=self._task_event,
            )
            self._task_tools = make_task_tools(self.tasks, lambda: self.backend.flags)
        self._failures: list[str] = []
        self._quiet = True  # the caller prints a failed start itself
        agent = self._try_build(self.config)
        self._quiet = False
        if agent is None:
            if self.tasks is not None:
                self.tasks.shutdown()
            raise ProviderMissing("\n".join(self._failures))
        self.session = create_session(
            self.repo,
            self.saved_config.main_model,
            self.saved_config.subagent_model,
            auto_run=self.saved_config.auto_run,
        )
        if self.tasks is not None:
            self.tasks.session_id = self.session.id
        self.thread_id = self.session.id
        self.agent = agent
        self._report_roster(warnings)

    # --- settings: saved vs effective ------------------------------------------
    #
    # saved_config is what config.json and the session rows hold. The process
    # runs with ``config``: saved_config with the run-only overrides on top.

    @property
    def config(self) -> Config:
        return replace(self.saved_config, **self.overrides) if self.overrides else self.saved_config

    @config.setter
    def config(self, value: Config) -> None:
        """Store ``value`` as the saved settings; overridden fields keep their saved values."""

        kept = {name: getattr(self.saved_config, name) for name in self.overrides}
        self.saved_config = replace(value, **kept) if kept else value

    def _build_task_graph(self, role, backend):
        """Build a background task's graph from the settings current when it starts."""

        return build_role_graph(
            self.repo,
            self.config,
            self.checkpointer,
            role,
            model_factory=self.model_factory,
            backend=backend,
            web_tools=self.web_tools,
        )

    def _report_roster(self, warnings: list[str]) -> None:
        for warning in warnings:
            self.console.print(f"agents: {warning}", markup=False)
        counts = {source: 0 for source in ("builtin", "home", "repo")}
        for definition in self.agents:
            counts[definition.source] = counts.get(definition.source, 0) + 1
        if counts["home"] or counts["repo"]:
            self.console.print(
                f"agents: {counts['builtin']} built-in, {counts['home']} home, {counts['repo']} repo",
                markup=False,
            )

    def loop(self) -> None:
        self.console.print(f"AbsolutCoder. {_COMMANDS}")
        if self.config.verify_command:
            self._say(f"verify: {self.config.verify_command} (runs after every request; /verify off to stop)")
        while True:
            self.service_tasks()
            try:
                line = self._ask("absolut> ")
            except (EOFError, KeyboardInterrupt):
                self.console.print()
                self.close()
                return
            if line.strip() in {"/exit", "/quit"}:
                self.close()
                return
            self.handle_line(line)

    def handle_line(self, line: str) -> None:
        stripped = line.strip()
        if not stripped:
            return
        if stripped.startswith("/"):
            self._slash(stripped)
            return
        self.run_task(stripped)

    def run_turn(
        self,
        text: str,
        paths: list[Path] | None = None,
        *,
        flags: TurnFlags | None = None,
        skill_block: str | None = None,
    ):
        """Run one user turn on the current thread and return the graph state.

        A tool or stream failure is printed and swallowed. The thread, agent,
        and checkpointer stay in place so the caller can prompt again.
        ``flags`` default to what ``text`` asks for; a follow-up turn that
        carries command output passes the user's own flags instead.

        Finished background tasks are reported in an extra text block ahead of
        the user's text; the flags never come from it. If the turn fails before
        that block reached the conversation, the notices go back on the queue.
        ``skill_block`` is a skill's instructions, sent as a text block between
        the notices and the request; flags never come from it.
        """

        self.last_turn_interrupted = False
        self.last_turn_failed = False
        self._no_terminal_noted = False
        attachments = list(self.pending_attachments if paths is None else paths)
        self.pending_attachments.clear()
        self.backend.flags = TurnFlags.from_message(text) if flags is None else flags
        start = len(self.backend.vet_log)
        config = {"configurable": {"thread_id": self.thread_id}}
        self._renderer = None
        notices = self.tasks.drain_notices() if self.tasks is not None else []
        updates = _updates_block(notices)
        try:
            content = build_user_content(text, attachments)
            if skill_block is not None:
                content.insert(0, {"type": "text", "text": skill_block})
            if updates is not None:
                content.insert(0, updates)
            # skills_metadata is checkpointed; None makes the skills middleware
            # reload the list from disk, so installs and removals show at once.
            payload: dict = {"messages": [{"role": "user", "content": content}], "skills_metadata": None}
            return self._drive(payload, config)
        except GraphBubbleUp:
            self._requeue_undelivered(notices, updates, config)
            raise
        except KeyboardInterrupt:
            self._requeue_undelivered(notices, updates, config)
            self.last_turn_interrupted = True
            if self._renderer is None:
                self.console.print("interrupted", markup=False, highlight=False)
            else:
                self._renderer.interrupted()
            return None
        except Exception as exc:
            self._requeue_undelivered(notices, updates, config)
            self.last_turn_failed = True
            if self._renderer is not None:
                self._renderer.end_line()
            self.error_console.print(f"Error: {exc}", markup=False, highlight=False)
            return None
        finally:
            self.turn_vet_log = self.backend.vet_log[start:]

    def _requeue_undelivered(self, notices: list[str], updates: dict | None, config) -> None:
        """Put the notices back unless the failed turn's user message was already saved."""

        if not notices:
            return
        try:
            messages = self.agent.get_state(config).values.get("messages", [])
            human = next((m for m in reversed(messages) if getattr(m, "type", None) == "human"), None)
            content = getattr(human, "content", None)
            first = content[0] if isinstance(content, list) and content else None
            delivered = isinstance(first, dict) and first.get("text") == updates["text"]
        except Exception:  # noqa: BLE001 - cannot tell: keep them rather than lose them
            delivered = False
        if not delivered:
            self.tasks.requeue_notices(notices)

    def run_task(
        self, text: str, paths: list[Path] | None = None, *, skill_block: str | None = None
    ) -> VerifyOutcome | None:
        """Run one user request; with a check command, repeat until the check passes.

        Every turn of the request, including the follow-ups that carry check
        output, runs with the flags of the user's own message. ``skill_block``
        goes with the first turn only.
        """

        self._task_stopped = False
        flags = TurnFlags.from_message(text)
        command = self.config.verify_command
        if not command:
            self.run_turn(text, paths, flags=flags, skill_block=skill_block)
            self._service_after_turn()
            return None
        rounds = 0

        def turn(message: str):
            nonlocal rounds
            rounds += 1
            result = self.run_turn(
                message,
                paths if rounds == 1 else [],
                flags=flags,
                skill_block=skill_block if rounds == 1 else None,
            )
            if self.last_turn_interrupted:
                raise KeyboardInterrupt
            self._service_after_turn()
            return result

        limit = max(1, self.config.max_rounds)

        def check():
            self._say(f"verify: {command}")
            self.backend.flags = flags
            result = self.backend.execute(command, timeout=self.config.verify_timeout)
            if result.exit_code != 0:
                self._say(f"verify: failed (exit {result.exit_code}), round {rounds} of {limit}")
            return result

        try:
            outcome = run_until_verified(
                turn, check, text, command=command, max_rounds=self.config.max_rounds
            )
        except PolicyDenied as error:
            self._say(f"verify: command refused: {error}")
            return VerifyOutcome(False, rounds, str(error))
        except KeyboardInterrupt:
            self._task_stopped = True
            self._say("verify: stopped")
            return VerifyOutcome(False, rounds, "stopped")
        except GraphBubbleUp:
            raise
        except Exception as error:  # noqa: BLE001 - a broken check must not end /loop or --until
            self._say(f"verify: could not run the check: {error}")
            return VerifyOutcome(False, rounds, str(error))
        if outcome.passed:
            self._say(f"verify: passed after {outcome.rounds} round(s)")
        else:
            self._say(f"verify: still failing after {outcome.rounds} round(s)")
            for line in outcome.last_output.splitlines()[-_FAILURE_TAIL_LINES:]:
                self._say(line)
        return outcome

    # --- background tasks -----------------------------------------------------

    def _service_after_turn(self) -> None:
        if not self.last_turn_interrupted:
            self.service_tasks()

    def service_tasks(self) -> None:
        """Ask the approvals background tasks are waiting for; only at a prompt boundary.

        Ctrl+C or end of input at one of these prompts rejects that task's
        request and stops asking the others until the next boundary.
        """

        if self.tasks is None:
            return
        for record, interrupts in self.tasks.pending_approvals():
            if not all(getattr(item, "id", None) for item in interrupts):
                # Command(resume={id: ...}) cannot address it, so it could never continue.
                self.tasks.cancel(record.id)
                self._say(f"task {record.id}: Cannot resume: an approval request has no id.")
                continue
            stop = False
            try:
                decisions = self._decisions(
                    interrupts,
                    question=lambda command, r=record: f"task {r.id} [{r.agent}] wants to run: {command}",
                )
            except (KeyboardInterrupt, EOFError) as error:
                self.console.print()
                if isinstance(error, EOFError):
                    self._note_no_terminal()
                decisions = _rejections(interrupts)
                stop = True
            try:
                self.tasks.resume(record.id, decisions)
            except (KeyError, ValueError, RuntimeError):
                pass  # cancelled or shut down while the user was answering
            if stop:
                return

    def _task_event(self, record: TaskRecord) -> None:
        """Print one whole status line for a background task; runs on worker threads."""

        if record.status == "queued":
            return  # the start_task call line already shows it
        status = "needs approval (press Enter to answer)" if record.status == "needs_approval" else record.status
        with self._output:
            renderer = self._renderer
            if renderer is not None:
                renderer.end_line()
            self.console.print(f"task {record.id} [{record.agent}] {status}", markup=False, highlight=False)

    def close(self) -> bool:
        """Stop the background tasks; False when the user left while some were still running.

        Safe to call again: later calls print nothing and return the first result.
        """

        if self._closed is not None:
            return self._closed
        self._closed = True
        if self.tasks is None:
            return True
        cancelled = self.tasks.shutdown()
        if cancelled:
            self._say(f"cancelled {cancelled} running task(s)")
        self._closed = self._wait_for_tasks()
        return self._closed

    def _wait_for_tasks(self) -> bool:
        busy = self._busy_task_ids()
        if not busy:
            return True
        self._say(f"waiting for {len(busy)} background task(s) to stop (Ctrl+C to leave now)")
        try:
            while busy:
                self.tasks.wait(busy[0], timeout=_WAIT_STEP)
                busy = self._busy_task_ids()
        except KeyboardInterrupt:
            self._say(f"left {len(busy)} background task(s) still stopping")
            return False
        return True

    def _busy_task_ids(self) -> list[str]:
        return [r.id for r in self.tasks.list() if r.status in {"queued", "running"}]

    def _tasks_command(self, argument: str) -> None:
        if self.tasks is None:
            self._say(_TASKS_OFF)
            return
        action, _, task_id = argument.partition(" ")
        task_id = task_id.strip()
        if not action:
            records = self.tasks.list()
            if not records:
                self._say("No background tasks.")
            for record in records:
                self._say(_task_line(record))
            return
        if action not in {"show", "cancel"} or not task_id:
            self._say(_TASKS_USAGE)
            return
        if action == "cancel":
            self._say(f"Cancelled {task_id}." if self.tasks.cancel(task_id) else f"{task_id} is not running.")
            return
        try:
            record = self.tasks.get(task_id)
        except KeyError:
            self._say(f"Unknown task {task_id}")
            return
        self._say(_task_line(record))
        if record.status == "failed":
            self._say(f"Error: {record.error}")
        elif record.result:
            self._say(record.result)

    def _say(self, text: str) -> None:
        self.console.print(text, markup=False, highlight=False)

    def _verify(self, argument: str) -> None:
        if not argument:
            self._say(f"verify: {self.config.verify_command or 'off'}")
            return
        command = None if argument.lower() == "off" else argument
        self.config = replace(self.config, verify_command=command)
        self._save()
        self._say(f"verify: {command or 'off'}")

    def _loop(self, argument: str) -> None:
        interval_text, _, prompt = argument.partition(" ")
        prompt = prompt.strip()
        try:
            interval = parse_interval(interval_text)
        except ValueError:
            interval = None
        if interval is None or not prompt:
            self._say(_LOOP_USAGE)
            return

        def turn(message: str):
            result = self.run_task(message)
            if self.last_turn_interrupted or self._task_stopped:
                raise KeyboardInterrupt
            return result

        count = run_interval_loop(
            turn,
            prompt,
            interval,
            sleep=self.sleep_fn,
            on_iteration=lambda n: self._say(f"loop: iteration {n}"),
        )
        self._say(f"loop: stopped after {count} iteration(s)")

    def _slash(self, line: str) -> None:
        command, _, argument = line.partition(" ")
        argument = argument.strip()
        if command == "/model":
            if not argument:
                self.console.print("Usage: /model <model-id>")
                return
            model_id = self._resolve_or_report(argument)
            if model_id is not None:
                self._retarget(main_model=model_id)
            return
        if command == "/subagent":
            if not argument:
                self.console.print("Usage: /subagent <model-id>")
                return
            model_id = self._resolve_or_report(argument)
            if model_id is not None:
                self._retarget(subagent_model=model_id)
            return
        if command == "/auto":
            enabled = self._auto_value(argument)
            if enabled is None:
                return
            if self._retarget(auto_run=enabled):
                self.console.print(f"auto_run {'on' if enabled else 'off'}")
            return
        if command == "/resume":
            self._resume(argument)
            return
        if command == "/attach":
            if not argument:
                self.console.print("Usage: /attach <path>")
                return
            self.pending_attachments.append(Path(argument).expanduser())
            self.console.print(f"attached {argument}")
            return
        if command == "/verify":
            self._verify(argument)
            return
        if command == "/loop":
            self._loop(argument)
            return
        if command == "/tasks":
            self._tasks_command(argument)
            return
        if command == "/agents":
            for definition in self.agents:
                self.console.print(
                    f"{definition.name}  [{definition.source}]  {definition.description}",
                    markup=False,
                )
            return
        if command == "/skills":
            self._skills_command(argument)
            return
        name = command[1:]
        skill = next((s for s in list_skills(self.repo) if s.name == name), None)
        if command not in _BUILTIN_COMMANDS and skill is not None:
            if not argument:
                self._say(f"Usage: /{name} <request>")
                return
            body = skill_body(skill)
            if not body:
                self._say(f"Could not load the {name} skill.")
                return
            block = f"[Instructions of the {name} skill, loaded because the user typed /{name}]\n{body}"
            self.run_task(argument, skill_block=block)
            return
        self.console.print(_COMMANDS)

    def _skills_command(self, argument: str) -> None:
        action, _, rest = argument.partition(" ")
        rest = rest.strip()
        if action == "add":
            if len(rest) >= 2 and rest[0] == rest[-1] and rest[0] in "'\"":
                rest = rest[1:-1].strip()
            if not rest:
                self._say(_SKILLS_ADD_USAGE)
                return
            try:
                info = install_skill(Path(rest).expanduser())
            except ValueError as error:
                self._say(f"Could not add skill: {error}")
                return
            self._say(f"Installed {info.name} into {global_skills_dir()} (global: loads in every project).")
            self._rebuild_after_skill_change()
            return
        if action == "remove":
            if not rest:
                self._say(_SKILLS_REMOVE_USAGE)
                return
            try:
                removed = remove_skill(rest)
            except ValueError as error:
                self._say(f"Could not remove skill: {error}")
                return
            if not removed:
                self._say(f"No global skill named {rest}.")
                return
            self._say(f"Removed {rest} (it no longer loads in any project).")
            self._rebuild_after_skill_change()
            return
        if argument:
            self._say(_SKILLS_USAGE)
            return
        skills = list_skills(self.repo)
        if not skills:
            self._say("No skills. Add one with /skills add <folder>.")
        for skill in skills:
            description = " ".join(skill.description.split())[:100]
            note = (
                f" (type its name in a request; /{skill.name} is a built-in command)"
                if f"/{skill.name}" in _BUILTIN_COMMANDS
                else ""
            )
            self._say(f"{skill.name}  [{skill.scope}]  {description}{note}")

    def _rebuild_after_skill_change(self) -> None:
        """Rebuild so the new skill folders are routed; keep the old graph if that fails."""

        agent = self._try_build(self.config)
        if agent is None:
            self._say("The change is saved but not live in this session yet.")
            return
        self.agent = agent

    def _resolve_or_report(self, argument: str) -> str | None:
        try:
            return resolve_model(argument)
        except ValueError as error:
            self._report(f"Invalid model id: {error}")
            return None

    def _report(self, message: str) -> None:
        self._failures.append(message)
        if not self._quiet:
            # soft_wrap keeps the install command on one line so it can be copied.
            self.console.print(message, markup=False, highlight=False, soft_wrap=True)

    def _try_build(self, config: Config):
        """Check keys and build the agent; print and return None on any failure."""

        self._failures = []
        if self.model_factory is None:
            try:
                pinned = [d.model for d in self.agents if d.model]
                missing = missing_keys(config, os.environ, extra_model_ids=pinned)
            except ValueError as error:
                self._report(f"Invalid model id: {error}")
                return None
            if missing:
                for entry in missing:
                    self._report(f"{entry} is not set. Set it and try again.")
                return None
        try:
            return self._build(config)
        except ProviderMissing as error:
            self._report(str(error))
            return None

    def _retarget(self, *, main_model: str | None = None, subagent_model: str | None = None, auto_run: bool | None = None) -> bool:
        previous = self.saved_config
        saved = replace(
            previous,
            main_model=previous.main_model if main_model is None else main_model,
            subagent_model=previous.subagent_model if subagent_model is None else subagent_model,
            auto_run=previous.auto_run if auto_run is None else auto_run,
        )
        overrides = dict(self.overrides)
        if auto_run is not None:
            overrides.pop("auto_run", None)  # /auto is a real, saved choice
        agent = self._try_build(replace(saved, **overrides))
        if agent is None:
            return False
        self.overrides = overrides
        self.saved_config = saved
        self.agent = agent
        self._save()
        self.session = update_session(
            self.session.id,
            main_model=saved.main_model,
            subagent_model=saved.subagent_model,
            auto_run=saved.auto_run,
        )
        self.thread_id = self.session.id
        return True

    def _resume(self, argument: str) -> None:
        sessions = list_sessions(self.repo)
        if not sessions:
            self.console.print("No sessions for this repo.")
            return
        for index, session in enumerate(sessions, start=1):
            self.console.print(
                f"{index}. {session.id}  {session.main_model}  {session.subagent_model}"
            )
        choice = argument or self._ask("Resume session number: ")
        try:
            selected = sessions[int(choice) - 1]
        except (ValueError, IndexError):
            self.console.print("Unknown session.")
            return
        self._adopt(selected)

    def _adopt(self, session: Session) -> None:
        loaded = load_session(session.id)
        saved = replace(
            self.saved_config,
            main_model=loaded.main_model,
            subagent_model=loaded.subagent_model,
            auto_run=loaded.auto_run,
        )
        agent = self._try_build(replace(saved, **self.overrides))
        if agent is None:
            return
        self.session = loaded
        self.thread_id = loaded.id
        self.saved_config = saved
        self.agent = agent
        self.console.print(f"Resumed {loaded.id}")

    def _build(self, config: Config | None = None):
        return build_agent(
            self.repo,
            config or self.config,
            self.checkpointer,
            chat_model=self.chat_model,
            model_factory=self.model_factory,
            backend=self.backend,
            extra_tools=self._task_tools,
            agents=self.agents,
            web_tools=self.web_tools,
            interactive=self.tasks is not None,
        )

    def _drive(self, payload, config):
        renderer = self._renderer = StreamRenderer(self.console)
        while True:
            parts = self.agent.stream(
                payload,
                config,
                stream_mode=["messages", "updates"],
                subgraphs=True,
                version="v2",
            )
            for part in parts:
                renderer.feed(part)
            snapshot = self.agent.get_state(config)
            interrupts = _pending_interrupts(snapshot)
            if not interrupts:
                renderer.finish(snapshot)
                return snapshot
            if not all(getattr(item, "id", None) for item in interrupts):
                # Command(resume={id: ...}) cannot address it; re-streaming would loop forever.
                raise RuntimeError("Cannot resume: an approval request has no id.")
            renderer.end_line()
            payload = Command(resume=_resume_map(interrupts, self._decisions(interrupts)))

    def _decisions(self, interrupts, *, question: Callable[[str], str] | None = None) -> dict[str, dict]:
        """Ask y/n for each requested command; ``question(command)`` is the line shown first."""

        answered: dict[str, dict] = {}
        for item in interrupts:
            interrupt_id = getattr(item, "id", None)
            if not interrupt_id:
                continue  # callers refuse to resume when any interrupt lacks an id
            decisions = []
            for command in _requested_commands(item):
                if question is None:
                    self.console.print(f"Run command? {command}")
                else:
                    self._say(question(command))
                try:
                    answer = self._ask("[y/n] ").strip().lower()
                except EOFError:
                    if question is not None:
                        raise  # a task approval: service_tasks rejects and stops asking
                    self._note_no_terminal()
                    answer = ""
                if answer in {"y", "yes"}:
                    decisions.append({"type": "approve"})
                else:
                    decisions.append(dict(_REJECTED))
            answered[interrupt_id] = {"decisions": decisions}
        return answered

    def _note_no_terminal(self) -> None:
        """Say once per turn that a request was rejected because nobody can answer."""

        if not self._no_terminal_noted:
            self._no_terminal_noted = True
            self.error_console.print(_NO_TERMINAL, markup=False, highlight=False)

    def _save(self) -> None:
        """Write the saved settings; run-only overrides are never written."""

        save_config(config_path(), self.saved_config)

    def _auto_value(self, argument: str) -> bool | None:
        lowered = argument.lower()
        if lowered in {"", "toggle"}:
            return not self.config.auto_run
        if lowered in {"on", "true", "1"}:
            return True
        if lowered in {"off", "false", "0"}:
            return False
        self.console.print("Usage: /auto [on|off]")
        return None

    def _ask(self, prompt: str) -> str:
        try:
            return self.input_fn(prompt)
        except TypeError:
            return self.input_fn()


def run_cli(args, config: Config | None = None, *, repl_factory=Repl) -> int:
    """Run one task, or open the terminal loop, from the current repo.

    A one-shot task returns 2 when its check command still fails, 1 when its
    turn failed, 130 when it was stopped with Ctrl+C, else 0; its error lines
    go to stderr. ``--until``, ``--max-rounds`` and ``--auto`` apply to that
    run only and are not saved.
    """

    from absolutcoder.config import ensure_config

    repo = Path.cwd()
    if config is None:
        config = ensure_config()
    overrides: dict[str, object] = {}
    if getattr(args, "auto", False):
        overrides["auto_run"] = True
    until_timeout = getattr(args, "until_timeout", None)
    if until_timeout is not None:
        overrides["verify_timeout"] = until_timeout
    task = getattr(args, "task", None)
    if task:
        until = getattr(args, "until", None)
        max_rounds = getattr(args, "max_rounds", None)
        if until is not None:
            overrides["verify_command"] = until
        if max_rounds is not None:
            overrides["max_rounds"] = max_rounds
    try:
        # A one-shot run ends with its turn, so it gets no background tasks.
        repl = _make_repl(repl_factory, repo, config, overrides, background=not task)
    except ProviderMissing as error:
        print(error, file=sys.stderr)
        return 1
    except ShellNotFound as error:
        print(f"Config error: shell: {error}", file=sys.stderr)
        return 1
    if task and hasattr(repl, "error_console"):
        repl.error_console = Console(stderr=True)
    attachments = [Path(item) for item in getattr(args, "attach", [])]
    clean = True
    try:
        if task:
            try:
                outcome = repl.run_task(task, paths=attachments)
            except KeyboardInterrupt:
                outcome, code = None, 130
            else:
                code = _one_shot_code(repl, outcome)
        else:
            repl.pending_attachments.extend(attachments)
            _interactive_loop(repl)
            code = 0
    finally:
        close = getattr(repl, "close", None)
        if close is not None:
            clean = close() is not False
    if not clean:
        # The user chose to leave while a task thread is still inside a model or
        # shell call. Pool threads are joined at interpreter exit, which would
        # hold the process open, so end it here.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    return code


def _one_shot_code(repl, outcome: VerifyOutcome | None) -> int:
    """130 after Ctrl+C; with a check, 0 or 2 by its result; else 1 when the turn failed."""

    if getattr(repl, "last_turn_interrupted", False) or getattr(repl, "_task_stopped", False):
        return 130
    if outcome is not None:
        return 0 if outcome.passed else 2
    return 1 if getattr(repl, "last_turn_failed", False) else 0


def _make_repl(repl_factory, repo: Path, config: Config, overrides: dict[str, object], *, background: bool):
    """Call the factory with the saved ``config`` and the run-only ``overrides``.

    A factory that does not take ``overrides`` gets the effective config instead.
    """

    # The signature is inspected only so test fakes without these keywords still work.
    try:
        parameters = list(inspect.signature(repl_factory).parameters.values())
    except (TypeError, ValueError):
        parameters = []
    any_keyword = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters)
    names = {p.name for p in parameters}
    kwargs: dict = {}
    if any_keyword or "background" in names:
        kwargs["background"] = background
    if any_keyword or "overrides" in names:
        kwargs["overrides"] = overrides
    elif overrides:
        config = replace(config, **overrides)
    return repl_factory(repo, config, **kwargs)


def _interactive_loop(repl) -> None:
    """Run the loop; on a terminal, read lines with prompt_toolkit and keep task output above the prompt."""

    console = getattr(repl, "console", None)
    if console is None:
        repl.loop()
        return
    if getattr(repl, "input_fn", None) is input:  # nothing injected
        repl.input_fn = make_input_fn(console)
    with background_output(console):
        repl.loop()


def _task_line(record: TaskRecord) -> str:
    return f"{record.id}  {record.agent}  {record.status}  {' '.join(record.prompt.split())[:60]}"


def _updates_block(notices: list[str]) -> dict | None:
    if not notices:
        return None
    return {"type": "text", "text": _UPDATES_HEADER + "\n" + "\n\n".join(notices)}


def _requested_commands(interrupt) -> list[str]:
    """The command (or tool name) of each action an approval interrupt asks about."""

    value = getattr(interrupt, "value", None)
    actions = value.get("action_requests", []) if isinstance(value, dict) else []
    commands = []
    for action in actions:
        command = ""
        if isinstance(action, dict):
            command = str(action.get("args", {}).get("command", action.get("name", "")))
        commands.append(command)
    return commands


def _rejections(interrupts) -> dict[str, dict]:
    """Reject every action of every addressable interrupt."""

    return {
        item.id: {"decisions": [dict(_REJECTED) for _ in _requested_commands(item)]}
        for item in interrupts
        if getattr(item, "id", None)
    }


def _pending_interrupts(snapshot) -> list:
    found = []
    for task in getattr(snapshot, "tasks", ()) or ():
        found.extend(getattr(task, "interrupts", ()) or ())
    return found


def _resume_map(interrupts, answered: dict[str, dict]):
    """One pending interrupt takes its answer directly; several take ``{id: answer}``."""

    if len(interrupts) == 1 and len(answered) == 1:
        return next(iter(answered.values()))
    return answered
