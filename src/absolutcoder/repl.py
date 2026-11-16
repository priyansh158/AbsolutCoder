"""Terminal loop for AbsolutCoder."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import replace
from pathlib import Path

from langgraph.errors import GraphBubbleUp
from langgraph.types import Command
from rich.console import Console

from absolutcoder.agent import build_agent
from absolutcoder.agent_files import load_roster
from absolutcoder.attachments import build_user_content
from absolutcoder.config import Config, config_path, resolve_model, save_config
from absolutcoder.git_policy import PolicyDenied, TurnFlags, protect_context_file
from absolutcoder.loops import VerifyOutcome, parse_interval, run_interval_loop, run_until_verified
from absolutcoder.models import ModelFactory, ProviderMissing, missing_keys
from absolutcoder.render import StreamRenderer
from absolutcoder.sessions import (
    Session,
    checkpoint_saver,
    create_session,
    list_sessions,
    load_session,
    update_session,
)
from absolutcoder.shell import GuardedShellBackend, ShellNotFound
from absolutcoder.webtools import make_web_tools

_COMMANDS = "Commands: /model, /subagent, /auto, /resume, /attach, /agents, /verify, /loop, /exit"
_LOOP_USAGE = "Usage: /loop <N>(s|m|h) <prompt>"
_FAILURE_TAIL_LINES = 15


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
    ) -> None:
        self.repo = Path(repo).resolve()
        self.config = replace(
            config,
            main_model=resolve_model(config.main_model),
            subagent_model=resolve_model(config.subagent_model),
            agent_models={k: resolve_model(v) for k, v in config.agent_models.items()},
        )
        self.chat_model = chat_model
        self.model_factory = model_factory
        self.input_fn = input_fn or input
        self.console = console or Console()
        self.sleep_fn = sleep_fn
        self.pending_attachments: list[Path] = []
        self.last_turn_interrupted = False
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
        self._failures: list[str] = []
        self._quiet = True  # the caller prints a failed start itself
        agent = self._try_build(self.config)
        self._quiet = False
        if agent is None:
            raise ProviderMissing("\n".join(self._failures))
        self.session = create_session(
            self.repo,
            self.config.main_model,
            self.config.subagent_model,
            auto_run=self.config.auto_run,
        )
        self.thread_id = self.session.id
        self.agent = agent
        self._report_roster(warnings)

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
        while True:
            try:
                line = self._ask("absolut> ")
            except (EOFError, KeyboardInterrupt):
                self.console.print()
                return
            if line.strip() in {"/exit", "/quit"}:
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

    def run_turn(self, text: str, paths: list[Path] | None = None, *, flags: TurnFlags | None = None):
        """Run one user turn on the current thread and return the graph state.

        A tool or stream failure is printed and swallowed. The thread, agent,
        and checkpointer stay in place so the caller can prompt again.
        ``flags`` default to what ``text`` asks for; a follow-up turn that
        carries command output passes the user's own flags instead.
        """

        self.last_turn_interrupted = False
        attachments = list(self.pending_attachments if paths is None else paths)
        self.pending_attachments.clear()
        self.backend.flags = TurnFlags.from_message(text) if flags is None else flags
        start = len(self.backend.vet_log)
        config = {"configurable": {"thread_id": self.thread_id}}
        self._renderer = None
        try:
            content = build_user_content(text, attachments)
            return self._drive({"messages": [{"role": "user", "content": content}]}, config)
        except GraphBubbleUp:
            raise
        except KeyboardInterrupt:
            self.last_turn_interrupted = True
            if self._renderer is None:
                self.console.print("interrupted", markup=False, highlight=False)
            else:
                self._renderer.interrupted()
            return None
        except Exception as exc:
            if self._renderer is not None:
                self._renderer.end_line()
            self.console.print(f"Error: {exc}", markup=False, highlight=False)
            return None
        finally:
            self.turn_vet_log = self.backend.vet_log[start:]

    def run_task(self, text: str, paths: list[Path] | None = None) -> VerifyOutcome | None:
        """Run one user request; with a check command, repeat until the check passes.

        Every turn of the request, including the follow-ups that carry check
        output, runs with the flags of the user's own message.
        """

        self._task_stopped = False
        flags = TurnFlags.from_message(text)
        command = self.config.verify_command
        if not command:
            self.run_turn(text, paths, flags=flags)
            return None
        rounds = 0

        def turn(message: str):
            nonlocal rounds
            rounds += 1
            result = self.run_turn(message, paths if rounds == 1 else [], flags=flags)
            if self.last_turn_interrupted:
                raise KeyboardInterrupt
            return result

        def check():
            self._say(f"verify: {command}")
            self.backend.flags = flags
            return self.backend.execute(command)

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

    def _say(self, text: str) -> None:
        self.console.print(text, markup=False, highlight=False)

    def _verify(self, argument: str) -> None:
        if not argument:
            self._say(f"verify: {self.config.verify_command or 'off'}")
            return
        command = None if argument.lower() == "off" else argument
        self.config = replace(self.config, verify_command=command)
        save_config(config_path(), self.config)
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
        if command == "/agents":
            for definition in self.agents:
                self.console.print(
                    f"{definition.name}  [{definition.source}]  {definition.description}",
                    markup=False,
                )
            return
        self.console.print(_COMMANDS)

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
        previous = self.config
        updated = replace(
            previous,
            main_model=previous.main_model if main_model is None else main_model,
            subagent_model=previous.subagent_model if subagent_model is None else subagent_model,
            auto_run=previous.auto_run if auto_run is None else auto_run,
        )
        agent = self._try_build(updated)
        if agent is None:
            return False
        self.config = updated
        self.agent = agent
        save_config(config_path(), self.config)
        self.session = update_session(
            self.session.id,
            main_model=self.config.main_model,
            subagent_model=self.config.subagent_model,
            auto_run=self.config.auto_run,
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
        adopted = replace(
            self.config,
            main_model=loaded.main_model,
            subagent_model=loaded.subagent_model,
            auto_run=loaded.auto_run,
        )
        agent = self._try_build(adopted)
        if agent is None:
            return
        self.session = loaded
        self.thread_id = loaded.id
        self.config = adopted
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
            agents=self.agents,
            web_tools=self.web_tools,
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

    def _decisions(self, interrupts) -> dict[str, dict]:
        answered: dict[str, dict] = {}
        for item in interrupts:
            interrupt_id = getattr(item, "id", None)
            if not interrupt_id:
                continue  # _drive refuses to resume when any interrupt lacks an id
            value = item.value
            actions = value.get("action_requests", []) if isinstance(value, dict) else []
            decisions = []
            for action in actions:
                command = ""
                if isinstance(action, dict):
                    command = str(action.get("args", {}).get("command", action.get("name", "")))
                self.console.print(f"Run command? {command}")
                answer = self._ask("[y/n] ").strip().lower()
                if answer in {"y", "yes"}:
                    decisions.append({"type": "approve"})
                else:
                    decisions.append({"type": "reject", "message": "User rejected the command."})
            answered[interrupt_id] = {"decisions": decisions}
        return answered

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

    A one-shot task returns 2 when its check command still fails, else 0.
    ``--until`` and ``--max-rounds`` apply to that run only and are not saved.
    """

    from absolutcoder.config import ensure_config

    repo = Path.cwd()
    if config is None:
        config = ensure_config()
    task = getattr(args, "task", None)
    if task:
        until = getattr(args, "until", None)
        max_rounds = getattr(args, "max_rounds", None)
        if until is not None:
            config = replace(config, verify_command=until)
        if max_rounds is not None:
            config = replace(config, max_rounds=max_rounds)
    try:
        repl = repl_factory(repo, config)
    except ProviderMissing as error:
        print(error, file=sys.stderr)
        return 1
    except ShellNotFound as error:
        print(f"Config error: shell: {error}", file=sys.stderr)
        return 1
    attachments = [Path(item) for item in getattr(args, "attach", [])]
    if task:
        outcome = repl.run_task(task, paths=attachments)
        return 2 if outcome is not None and not outcome.passed else 0
    repl.pending_attachments.extend(attachments)
    repl.loop()
    return 0


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
