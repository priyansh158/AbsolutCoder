# AbsolutCoder: Any-Provider Models and a Full Coding Agent — Implementation Plan

> **For the implementer:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task by task. Tasks use checkbox (`- [ ]`) syntax for tracking.

## Context

AbsolutCoder (`C:\Users\priya\AbsolutCoder`, ~1,000 lines, 20 passing tests) is a
terminal coding agent on `deepagents` + LangGraph. Today it is hard-wired to
OpenAI (`config.resolve_model` rejects other providers, `cli.main` demands
`OPENAI_API_KEY`), has one worker subagent, no web access, and prints only
delegated task names and the final answer.

The user wants it to stay zero-config on OpenAI while accepting any provider
someone configures, and to grow into a full coding agent: several specialised
subagents in parallel, web fetch and search, work-until-verified looping, a
`/loop` command, a live terminal display, and background tasks that keep
running while the user types.

**Goal:** make AbsolutCoder provider-agnostic (OpenAI default) and add parallel
subagents, web tools, a streaming UI, looping and background tasks, in five
shippable phases.

**Architecture:** stay on `deepagents` 0.7.35 and use what it already provides:
`init_chat_model` provider resolution, concurrent `task` calls, always-on
summarisation, per-subagent stream metadata. New code is small focused modules
(`models`, `subagents`, `agent_files`, `webtools`, `render`, `loops`, `tasks`);
`repl.py` stays a thin driver. Background tasks are a thread-pool manager that
runs one standalone agent graph per task on its own checkpoint thread.

**Tech stack:** Python >=3.12, deepagents 0.7.35, langchain 1.5.0, langgraph
1.2.20, langgraph-checkpoint-sqlite 3.1.3, rich 15, httpx; new deps `ddgs`,
`pyyaml`, `prompt_toolkit` (P5).

**Spec:** this file is both design and plan (plan mode allows only this file).
Task 0 copies it into the repo as
`docs/superpowers/plans/2026-11-15-any-provider-full-agent.md`.

### Decisions made with the user

- **Looping:** both work-until-verified and `/loop <interval> <prompt>`.
- **Web:** `web_fetch` plus keyless DuckDuckGo `web_search`. Tavily deferred.
- **Subagents:** built-in roster plus custom agent files in
  `<repo>/.absolutcoder/agents/*.md` and `~/.absolutcoder/agents/*.md`.
- **Approach:** extend on `deepagents`, plus background tasks (phase 5).
  Rejected: rewriting the loop on bare LangGraph (reimplements file tools,
  summarisation and delegation for no user-visible gain).
- **Interface:** terminal only (Hallmark not used).
- **Delivery:** `git init`, baseline commit, five phases, tests green after each.

### My calls (not asked; correct at approval if wrong)

- Web tools live only in the `web-researcher` subagent, which has no shell and
  no file access. Its report is wrapped as untrusted content. `/auto` stays off
  by default.
- On Windows the shell prefers Git's `bash.exe` (found beside `git.exe`), else
  `cmd /c`. Config key `shell` overrides.
- Custom agent files may use `tools: [web]` only on its own in this version.
- Background-task completion is announced on the next turn; it does not start
  a turn by itself.

## Global Constraints

- A bare model id resolves to `openai:<id>`
  (`tests/test_config.py::test_bare_model_id_becomes_openai` stays unchanged).
- An existing `config.json` and the `sessions` SQLite table load unchanged; all
  new config keys are optional.
- Keep these signatures: `Config(main_model=, subagent_model=, auto_run=)`,
  `create_session(repo, main, sub, auto_run=)`,
  `Repl(repo, config, chat_model=, input_fn=, console=)`,
  `build_agent(repo, config, checkpointer, chat_model=, backend=)`,
  `vet(command, flags)`, `TurnFlags.from_message`.
- `Repl._drive` keeps calling the synchronous `self.agent.stream(...)`
  (`test_stream_exception_stays_in_the_session` monkeypatches it).
- `PolicyDenied` is raised from `backend.execute`, never returned.
  `turn_vet_log` entries stay `{"command", "allowed"[, "reason"]}`.
- Every subagent spec sets `"tools"` explicitly. A spec without the key inherits
  the parent's tools (`deepagents/graph.py`, declarative-subagent branch).
- Pass model **strings** to deepagents unless a provider override exists; an
  instance loses the provider/harness profile lookup.
- All tests run offline with `tests/support.py::scripted_model`; no test may
  reach the network.
- Git: commit as the user's configured identity. No co-author or
  "generated with" trailers. Commit messages contain no tool names and not the
  word "agent" (also avoid "subagent": write "worker roles", "delegation").
  Never commit `CLAUDE.md`, `CLAUDE.local.md` or `AbsolutCoder.md`. No remote
  exists, so nothing is pushed. Baseline goes on `main`; each phase is built on
  its own branch (`phase-1-providers` … `phase-5-background-tasks`), not merged.
- Every task follows one cycle: write the listed failing tests, run them and see
  them fail, implement to the listed interface, run the task's tests, run the
  full suite (`.venv\Scripts\python.exe -m pytest -q`), commit.

## Review Focus

Failure modes the feature tests would not catch; each has a pinning test in the
task that owns the code.

1. A fetched page tells the agent to run a command and the main agent obeys the
   researcher's report → `test_web_report_is_wrapped_as_untrusted` (P2 T3).
2. A provider streams list/reasoning content blocks and the display prints
   reprs → `test_render_handles_block_content` (P3 T1).
3. Internal model calls (summarisation) leak tokens into the display →
   `test_only_model_node_tokens_render` (P3 T1).
4. A background task finishes while an approval prompt is open and its notice is
   read as the answer → `test_notice_never_consumed_as_approval` (P5 T2).
5. A resumed session's provider package or key is now missing →
   `test_resume_with_unavailable_provider_prints_hint` (P1 T3).

## File Map

| File | Responsibility |
|---|---|
| `src/absolutcoder/models.py` (new) | Model id parsing, provider settings, model construction, key checks, install hints |
| `src/absolutcoder/config.py` | New defaulted fields; `resolve_model` delegates to `models.normalize_model_id` |
| `src/absolutcoder/cli.py` | Provider-aware key check after config load; `--until`, `--max-rounds` |
| `src/absolutcoder/subagents.py` (new) | Built-in roster, tool groups, spec building |
| `src/absolutcoder/agent_files.py` (new) | Discover, parse, validate `agents/*.md` |
| `src/absolutcoder/webtools.py` (new) | `web_fetch`, `web_search`, URL guard, HTML to text |
| `src/absolutcoder/agent.py` | `build_agent` wiring: model factory, roster, skills, extra tools |
| `src/absolutcoder/render.py` (new) | `StreamRenderer`: stream parts to console |
| `src/absolutcoder/loops.py` (new) | Verify loop, interval parsing, interval loop |
| `src/absolutcoder/shell.py` | Single-process execution, per-call env, git write lock, shell choice |
| `src/absolutcoder/git_policy.py` | Wrapper unwrapping in `vet`, tighter `_asked` |
| `src/absolutcoder/tasks.py` (new) | `TaskManager`, `TaskRecord`, task tools |
| `src/absolutcoder/repl.py` | Turn driving, slash dispatch, approvals, task notices |
| `tests/support.py` | Add `role_models(mapping)` and `task_call(agent, description, call_id)` |
| `tests/conftest.py` | Shared `git_repo` fixture replacing the two duplicated `init_repo()` helpers |
| `tests/test_{models,subagents,agent_files,webtools,render,loops,shell,tasks}.py` (new) | One per module |

`sessions.py` and `attachments.py` are unchanged.

## Interfaces

```python
# models.py
ModelFactory = Callable[[str, str], BaseChatModel | str]  # (role, model_id); role is "main" or the worker name
@dataclass(frozen=True)
class ProviderSettings:
    kind: str | None = None; base_url: str | None = None
    api: str = "responses"; api_key_env: str | None = None   # api: "responses" | "chat"
class ProviderMissing(RuntimeError): ...
def normalize_model_id(model_id: str) -> str
def provider_of(model_id: str) -> str
def build_model(model_id: str, providers: Mapping[str, ProviderSettings]) -> BaseChatModel | str
def missing_keys(config: Config, environ: Mapping[str, str]) -> list[str]
def install_hint(error: ImportError) -> str | None

# subagents.py
TOOL_GROUPS: dict[str, tuple[str, ...]]   # read, search, edit, shell, web
@dataclass(frozen=True)
class AgentDef:
    name: str; description: str; prompt: str; groups: tuple[str, ...]
    model: str | None = None; source: str = "builtin"
def builtin_agents() -> list[AgentDef]
def merge_agents(builtin, home, repo: list[AgentDef]) -> list[AgentDef]
def build_specs(defs: list[AgentDef], *, backend, config: Config,
                model_factory: ModelFactory, interrupt_on: dict | None) -> list[dict]

# agent_files.py
class AgentFileError(ValueError): ...
def parse_agent_file(path: Path) -> AgentDef
def load_agent_dir(directory: Path, source: str) -> tuple[list[AgentDef], list[str]]  # (defs, warnings)

# webtools.py
class WebToolError(ValueError): ...
def check_url(url: str) -> None
def html_to_text(html: str, *, max_chars: int = 20000) -> str
def make_web_tools(*, client: httpx.Client | None = None,
                   search_fn: Callable[[str, int], list[dict]] | None = None,
                   max_chars: int = 20000) -> list[BaseTool]

# agent.py (existing params unchanged)
def build_agent(repo, config, checkpointer, *, chat_model=None,
                model_factory: ModelFactory | None = None,
                backend: GuardedShellBackend | None = None,
                extra_tools: Sequence[BaseTool] = (),
                agents: list[AgentDef] | None = None) -> CompiledStateGraph

# render.py
class StreamRenderer:
    def __init__(self, console: Console, *, live: bool | None = None) -> None
    def feed(self, part: dict) -> None      # one {"type","ns","data"} stream part
    def finish(self, snapshot) -> None

# loops.py
@dataclass(frozen=True)
class VerifyOutcome: passed: bool; rounds: int; last_output: str
def parse_interval(text: str) -> float      # "30s" | "5m" | "2h"
def run_until_verified(run_turn, verify, prompt: str, *, max_rounds: int = 5) -> VerifyOutcome
def run_interval_loop(run_turn, prompt: str, interval: float, *,
                      sleep=time.sleep, max_iterations: int | None = None) -> int

# shell.py / git_policy.py
def find_shell(preference: str = "auto") -> list[str]
class GuardedShellBackend:  # __init__ gains shell: str = "auto", agent_label: str = "main"
def unwrap_shell(tokens: list[str]) -> str | None   # inner command of sh -c / bash -lc / cmd /c / env

# tasks.py
@dataclass
class TaskRecord:
    id: str; agent: str; prompt: str; status: str
    result: str | None; error: str | None; thread_id: str
    # status: queued | running | needs_approval | done | failed | cancelled
class TaskManager:
    def __init__(self, build: Callable[[str, GuardedShellBackend], CompiledStateGraph], *,
                 session_id: str, repo: Path, max_workers: int = 4,
                 on_event: Callable[[TaskRecord], None] | None = None) -> None
    def start(self, agent: str, prompt: str, *, flags: TurnFlags) -> TaskRecord
    def get(self, task_id: str) -> TaskRecord
    def list(self) -> list[TaskRecord]
    def cancel(self, task_id: str) -> bool
    def pending_approvals(self) -> list[tuple[TaskRecord, list]]
    def resume(self, task_id: str, decisions: dict) -> None
    def drain_notices(self) -> list[str]
    def shutdown(self, *, wait: bool = False) -> None
def make_task_tools(manager: TaskManager) -> list[BaseTool]  # start_task, check_task, list_tasks, cancel_task

# repl.py
Repl.__init__(..., model_factory: ModelFactory | None = None)
Repl.run_turn(text, paths=None, *, flags: TurnFlags | None = None)
```

### Config schema (all new keys optional)

```json
{"main_model": "openai:…", "subagent_model": "openai:…", "auto_run": false,
 "providers": {"local": {"kind": "openai", "base_url": "http://localhost:11434/v1",
                         "api": "chat", "api_key_env": null}},
 "agent_models": {"explore": "openai:…"},
 "shell": "auto", "verify_command": null, "max_rounds": 5,
 "web_max_chars": 20000, "max_background_tasks": 4}
```

- `normalize_model_id`: bare id → `openai:<id>`; `provider:rest` accepted when
  provider matches `^[a-z0-9_]+$` and `rest` is non-empty, split on the first
  colon (so Ollama tags are written `ollama:llama3:8b`).
- `providers.<name>.kind == "openai"` defines an alias used as `<name>:<model>`,
  built with `init_chat_model(model=<id>, model_provider="openai", base_url=…,
  use_responses_api=(api == "responses"))`. `providers.openai` may set
  `base_url`/`api` directly. Everything else is passed to deepagents as a string.
- Key check: `openai`→`OPENAI_API_KEY`, `anthropic`→`ANTHROPIC_API_KEY`,
  `google_genai`→`GOOGLE_API_KEY` or `GEMINI_API_KEY`; an alias uses its
  `api_key_env`; null or unknown provider is not checked.
- Model per worker: `agent_models[name]`, else `subagent_model`.

### Roster

| Name | Purpose | Tools | Permissions | Can prompt |
|---|---|---|---|---|
| `explore` | Read-only repo search, returns findings with paths | ls, read_file, glob, grep | secret denies + deny write `/**` | no |
| `plan` | Read-only design, returns a step plan | same | same | no |
| `general-purpose` | Edits files and runs commands, reports what changed and was verified | all eight filesystem tools | secret denies | yes (`execute`) |
| `web-researcher` | Fetches and searches the web, returns cited notes | web_fetch, web_search | no backend at all | no |

- Read-only spec: `"tools": []`, `"middleware": [FilesystemMiddleware(backend=routed,
  tools=["ls","read_file","glob","grep"])]`, permissions on the spec (they
  replace the parent's, so repeat `secret_permissions()`).
- `web-researcher` is a `CompiledSubAgent` whose runnable is
  `langchain.agents.create_agent(model, tools=make_web_tools(...), system_prompt=…,
  name="web-researcher")`: no filesystem middleware, no shell.

### Custom agent file

```
---
name: reviewer          # optional, defaults to file stem; ^[a-z][a-z0-9-]{1,39}$
description: …          # required, 1-300 chars
tools: [read, search]   # optional; groups read|search|edit|shell|web; default [read, search]
model: provider:model   # optional
---
<system prompt; required, non-empty>
```

`web` combined with any other group is an `AgentFileError`. Unknown keys warn.
An invalid file is skipped with one warning line and never stops startup.
Precedence by name: repo, then home, then built-in.

### Commands and flags

- `/agents` — roster with each entry's source.
- `/verify <cmd>` | `/verify off` | `/verify` — command run after each turn via
  `backend.execute` with that turn's flags; a failure is fed back as the next turn.
- `/loop <N>(s|m|h) <prompt>` — repeat until Ctrl+C.
- `/tasks` | `/tasks show <id>` | `/tasks cancel <id>`.
- `absolut "task" --until "<cmd>" [--max-rounds N]` — exit 0 if verified, 2 if
  rounds are exhausted.

---

## Task 0: Baseline

- [ ] `git init` in `C:\Users\priya\AbsolutCoder`; add `.gitignore` (`.venv/`,
  `__pycache__/`, `.pytest_cache/`, `*.egg-info/`, `AbsolutCoder.md`).
- [ ] Install the `no-agentic-coauthor` commit-msg hook
  (`~/.claude/skills/no-agentic-coauthor/scripts/strip-cursor-coauthor.sh` →
  `.git/hooks/commit-msg`).
- [ ] Copy this plan to `docs/superpowers/plans/2026-11-15-any-provider-full-agent.md`.
- [ ] Run the suite (expect 20 passed), commit on `main`, check
  `git log -1 --format=%B` has no trailer.

## Phase 1: Providers (branch `phase-1-providers`)

`pyproject.toml`: add extras `anthropic`, `google`, `ollama`, `all`.

- [ ] **T1 `models.py`, `config.py`.** Tests in `tests/test_models.py`:
  `test_provider_prefix_is_kept` (`anthropic:x` unchanged);
  `test_default_openai_passes_string_through` (`build_model` returns the string);
  `test_compatible_endpoint_uses_chat_completions` (instance has the `base_url`
  and `use_responses_api is False`); `test_missing_keys_follow_configured_providers`;
  `test_google_accepts_either_key`; `test_install_hint_names_extra`;
  `test_old_config_file_still_loads`. In `tests/test_config.py` replace
  `test_other_provider_is_rejected` with `test_malformed_provider_is_rejected`
  (`":x"`, `"a b:x"`).
- [ ] **T2 `cli.py`.** Key check moves after config load and uses `missing_keys`.
  `tests/test_cli_help.py::test_cli_reports_missing_anthropic_key` (stderr names
  `ANTHROPIC_API_KEY`, exit 1). `--help` still exits 0 with no key.
- [ ] **T3 `agent.py`, `repl.py`.** Add `model_factory`; turn a provider
  `ImportError` into a printed `pip install absolutcoder[<extra>]` hint. Replace
  the `_worker_model` closure walker in `tests/test_agent.py` with
  `test_subagent_model_comes_from_factory` (a recording factory sees role
  `general-purpose` and `openai:worker-test`). Add
  `tests/test_repl.py::test_resume_with_unavailable_provider_prints_hint`.
  Add `role_models` to `tests/support.py`; move `init_repo()` into a
  `git_repo` fixture in `tests/conftest.py`.

## Phase 2: Worker roles, web tools, agent files (branch `phase-2-roster-web`)

`pyproject.toml`: add `httpx`, `pyyaml`, `ddgs`.

- [ ] **T1 Spike.** `tests/test_subagents.py::test_parallel_subagent_interrupts_resume_by_id`:
  two `task` calls whose scripted workers each call `execute`; assert two
  prompts and both commands in `turn_vet_log`. In `repl._decisions`, key on
  `item.id` and skip interrupts without an id (no `""` fallback). If this spike
  fails, stop and revise the approval design before T2.
- [ ] **T2 `subagents.py`, `agent.py`.** `test_explore_has_no_write_or_execute_tools`;
  `test_explore_write_is_denied` (scripted `write_file` gets a permission error,
  no file on disk); `test_main_delegates_and_reads_report`;
  `test_every_spec_declares_tools`. Update `SYSTEM_PROMPT` to describe the
  roster and parallel delegation. Add `/agents`.
- [ ] **T3 `webtools.py`.** With `httpx.MockTransport`:
  `test_fetch_returns_text_without_script`; `test_fetch_truncates`;
  `test_fetch_rejects_file_and_private_hosts` (`file://`, `127.0.0.1`,
  `169.254.169.254`, and a redirect to one);
  `test_search_uses_injected_backend`; `test_web_tools_only_on_web_researcher`
  (main and `general-purpose` tool names contain neither);
  `test_web_report_is_wrapped_as_untrusted`. `html_to_text` uses the stdlib
  `html.parser` (no new dependency), dropping `script`/`style`/`nav`.
- [ ] **T4 `agent_files.py`.** `test_parse_valid_file`;
  `test_web_with_shell_is_rejected`; `test_repo_overrides_home`;
  `test_bad_file_is_skipped_with_warning`; `test_custom_agent_is_callable_by_task`.
- [ ] **T5 Skills.** Pass `skills=["/.absolutcoder/skills/"]` only when that
  directory exists. `test_skills_source_passed_only_when_dir_exists`.

## Phase 3: Streaming UI (branch `phase-3-streaming`)

- [ ] **T1 `render.py`.** Fed hand-built parts: `test_tokens_are_concatenated_per_agent`;
  `test_whole_message_is_printed_once` (scripted models yield whole messages;
  dedupe by message id against the final print); `test_subagent_lines_are_labelled`
  (`ns != ()` plus `metadata["lc_agent_name"]`); `test_tool_call_and_result_lines`;
  `test_plain_console_emits_no_control_codes`; `test_render_handles_block_content`;
  `test_only_model_node_tokens_render`. Rich `Live` only on a TTY; plain lines
  otherwise.
- [ ] **T2 `repl._drive`.** Call `self.agent.stream(payload, config,
  stream_mode=["messages","updates"], subgraphs=True, version="v2")` and feed
  the renderer; `get_state` stays the authority for interrupts.
  `test_turn_output_shows_subagent_activity`;
  `test_keyboard_interrupt_mid_stream_keeps_session` ("interrupted" printed, next
  turn works). `test_stream_exception_stays_in_the_session` passes unchanged.

## Phase 4: Looping and shell (branch `phase-4-loops-shell`)

- [ ] **T1 `shell.py`.** Vet per segment as now, rewrite commit segments, rejoin
  with the original operators and run **once** via our own
  `subprocess.run([*find_shell(), command], env=<per-call copy>)`, keeping the
  `[stderr]`/exit-code formatting of `deepagents/backends/local_shell.py:316-346`.
  A per-repo `threading.Lock` serialises git writes. `find_shell("auto")` on
  Windows prefers Git's `bash.exe` beside `git.exe`, else `cmd /c` with the
  existing stepwise executor as fallback. Tests in `tests/test_shell.py`:
  `test_cd_persists_across_and_chain`; `test_multiline_loop_runs_as_one_script`
  (skipped with no POSIX shell); `test_commit_env_is_per_call` (`backend._env`
  identical before and after); `test_concurrent_commits_both_land` (two threads,
  two commits, no `index.lock` error); `test_commit_inside_chain_still_scrubbed`.
  All of `tests/test_git_policy.py` and `tests/test_task_commit.py` pass unchanged.
- [ ] **T2 `git_policy.py`.** `test_wrapped_git_push_is_denied` (`sh -c "git push"`,
  `bash -lc`, `cmd /c`, `env X=1 git push`);
  `test_push_as_ordinary_verb_does_not_unlock` ("push this logic into a helper").
  `test_turn_flags_follow_the_user_message` passes unchanged.
- [ ] **T3 `loops.py`, `/verify`, `/loop`, `--until`, `--max-rounds`.**
  `tests/test_loops.py`: `test_verify_passes_first_round`;
  `test_failure_output_becomes_next_turn`; `test_stops_at_max_rounds`;
  `test_parse_interval`; `test_interval_loop_stops_on_keyboard_interrupt`;
  `test_until_flag_sets_exit_code`. Add a line to `SYSTEM_PROMPT`: run the
  project's checks before reporting done.

## Phase 5: Background tasks (branch `phase-5-background-tasks`)

`pyproject.toml`: add `prompt_toolkit`.

- [ ] **T1 `tasks.py`.** Thread pool; one standalone `create_deep_agent` graph
  per task, thread id `<session>:task:<n>` on the shared `SqliteSaver` (it holds
  its own lock), its own `GuardedShellBackend` (own flags and vet log).
  Cancellation is cooperative via LangGraph `RunControl.request_drain()` passed
  as `stream(..., control=)`; a running subprocess is bounded by the shell
  timeout, not killed. `tests/test_tasks.py`: `test_task_runs_to_done_with_result`;
  `test_two_tasks_run_concurrently` (barrier in the responders);
  `test_cancel_marks_cancelled`; `test_task_uses_own_thread_id`;
  `test_task_backend_has_own_flags_and_log`; `test_approval_parks_then_resumes`.
  If `request_drain` does not stop a run as documented, fall back to marking the
  record cancelled and discarding its result.
- [ ] **T2 Tools and REPL.** `make_task_tools` goes to the main agent only via
  `extra_tools`. Approvals from tasks are asked only at a prompt boundary.
  Completion notices are prepended to the next turn. `/tasks` commands; `/exit`
  cancels running tasks. Tests: `test_main_agent_starts_and_checks_task`;
  `test_completion_notice_is_prepended_to_next_turn`;
  `test_subagents_do_not_get_task_tools`; `test_exit_cancels_running_tasks`;
  `test_notice_never_consumed_as_approval`.
- [ ] **T3 Input.** `prompt_toolkit` `PromptSession` + `patch_stdout` when stdin
  is a TTY and no `input_fn` was given. Manual check only.

## Deferred

Tavily search; `web` combined with read tools in custom agents; auto-starting a
turn when a task finishes; killing a subprocess on cancel; moving image
attachments out of checkpoints; `stream_events` v3; a UI for `agent_models`
(config-only).

## Verification

After every task: `.venv\Scripts\python.exe -m pytest -q` is green.

End of each phase, by hand in a scratch git repo with a real key:

- **P1:** `absolut "say hi"` works with only `OPENAI_API_KEY` and an untouched
  `config.json`. With `main_model` set to `anthropic:<id>` and no key, the error
  names `ANTHROPIC_API_KEY`. A `providers.local` alias reaches an
  OpenAI-compatible server if one is available.
- **P2:** ask for a repo survey and see `explore` workers run in parallel; ask a
  question needing the web and see `web-researcher` used; drop a file in
  `.absolutcoder/agents/` and see it in `/agents`.
- **P3:** tokens stream; tool calls and labelled worker lines appear; Ctrl+C
  mid-turn returns to the prompt.
- **P4:** `absolut "make the tests pass" --until "python -m pytest -q"` loops
  and exits 0; `/loop 30s <prompt>` repeats until Ctrl+C; a chained
  `cd sub && <cmd>` runs in `sub`.
- **P5:** start a background task, keep typing, see the completion notice on the
  next turn; `/tasks cancel <id>` works; an approval from a task appears at the
  prompt, not mid-line.

## Execution

Recommended: **subagent-driven**, one fresh implementer and reviewer per task,
because the tasks share interfaces across seven new modules and a guard
regression (git policy, path checks) would be costly. Stop for the user's
review at the end of each phase before starting the next branch.
