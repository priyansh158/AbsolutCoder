# AbsolutCoder: built-in Hallmark design rules and a Graphify code map — Implementation Plan

> **For the implementer:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development to implement this plan task by task (the same process as the rebuild: a brief per task, an implementer, a task review, one final whole-branch review). Tasks use checkbox (`- [ ]`) syntax.

## Context

The rebuilt AbsolutCoder (branch `phase-5-background-tasks`, tip `04e6e18`, 415
tests, not merged) works, and the user wants two things added as global,
built-in capabilities:

1. **UI that does not look machine-generated**, using the user's Hallmark design
   skill (`~/.claude/skills/hallmark`, MIT licensed, 187 files, Node scripts
   with no npm dependencies).
2. **Lower token use**, using Graphify (`https://github.com/Graphify-Labs/graphify`,
   branch `v8`, PyPI package `graphifyy`, Apache-2.0/MIT): it parses a codebase
   locally into a graph that the assistant queries before grepping or reading files.

Findings that shape the design:

- AbsolutCoder loads skills only from `<repo>/.absolutcoder/skills/`
  (`agent.py:35,161`), only for the main model; worker roles get none
  (`subagents.py:218-232`). There is no home-level skills folder.
- Hallmark's text uses a `<skill-dir>` placeholder that nothing fills in, and
  its full flow reads about 100k characters of guidance (~25k tokens) per build.
- Graphify saves tokens when the assistant must understand **existing** code.
  It would not have changed the observed snake-and-ladder run (an almost empty
  folder; cost came from 52 round trips). Branch `v8` publishes no per-query
  token-reduction figure for the open-source tool, so this plan adds a way to
  measure it.
- The observed run used the OLD code: plain `absolut` runs the `main` branch
  from the system Python (`C:\Users\priya\AppData\Local\Programs\Python\Python312`),
  which also lacks the `ddgs` package.

**Goal:** every UI the assistant writes follows Hallmark's core rules, the full
Hallmark flow is one command away, existing codebases are navigated through a
locally built code map, token use is visible per turn, and plain `absolut` runs
the new build.

### Decisions made with the user

- **Hallmark source:** a new global skills folder `~/.absolutcoder/skills/`,
  loaded in every project; Hallmark is copied into it.
- **Hallmark use:** its core rules (the 16-rule "Floor", about 1k tokens) apply
  to every UI the assistant writes; the full flow runs only when the user names
  Hallmark.
- **Graphify packaging:** always installed (a required dependency).
- **Code map build:** automatic, in the background, refreshed after edits.
- **Merge:** at the end, fast-forward `main` and install dependencies into the
  Python that runs `absolut`. Nothing is pushed.

### My calls (correct at approval if wrong)

- The global skills folder is mounted at the virtual path
  `/.absolutcoder-global/skills/` (a plain `/skills/` route would hide a real
  `skills/` folder in a repository). It is read-only to the file tools.
- A skill may ship an `always.md` (at most 4,000 characters). Its text is added
  to the system prompt of the main model and of every worker that can edit
  files. Hallmark's `always.md` is the Floor, adapted; that is how "core rules
  always" is implemented, and any other skill can use the same mechanism.
- Typing `/<skill-name> <request>` in the loop runs that request with the skill's
  full instructions loaded (`/hallmark build a pricing page`).
- The copied Hallmark's listing description is changed to say it is used only
  when the user names it. The `LICENSE` file is kept in the copy.
- Tiny folders are not mapped: the code map is built only when a folder has at
  least 20 source files (config `code_map_min_files`).
- Graphify runs with `GRAPHIFY_BACKEND=none` (it can never call a model) and
  `GRAPHIFY_QUERY_LOG_DISABLE=1` (no query log outside the project). Only its
  code-only, local parsing is used; documents, PDFs and images are not sent anywhere.
- `graphify-out/` is added to `.git/info/exclude`, like `AbsolutCoder.md`.
- A one-line token total is printed after each turn, so savings can be seen.

## Global Constraints

- All existing tests (415) keep passing with unchanged assertions unless a task
  names the assertion.
- Existing `config.json` files load unchanged; new keys are optional:
  `code_map` (`"auto"` | `"off"`, default `"auto"`), `code_map_min_files`
  (positive int, default `20`).
- Tool isolation is preserved: the web role gets no file, shell or code-map
  tools; read-only roles get no write or shell tools; task-manager tools stay on
  the main model only. Every worker spec still sets `"tools"` explicitly.
- Global skills are never writable through the file tools (deny write on
  `/.absolutcoder-global/**` for every role).
- Graphify is invoked with an argument list, never through a shell string; model
  text is passed as arguments only.
- No test reaches the network or calls a model. Tests that need the real
  `graphify` package are skipped when it is not importable.
- Git: commits as the configured identity, subject and optional body only, no
  attribution or session lines, no standalone word "agent"/"agents" in messages.
  Work on a new branch `phase-6-hallmark-code-map` from `04e6e18`.
- Each task: write the listed failing tests, see them fail, implement, run the
  task's tests, run the full suite
  (`PYTHONPATH="$PWD/src" /c/Users/priya/AbsolutCoder/.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider`), commit.

## Review Focus

1. A repository with its own `skills/` or `.absolutcoder/skills/` folder: the
   global route must not hide either → `test_global_route_does_not_shadow_repo_folders` (T2).
2. A hostile or oversized `always.md` in a project skill inflating every
   request → `test_always_text_is_capped_and_project_always_is_ignored` (T2):
   only GLOBAL skills contribute always-on text; project skills do not.
3. Code map missing, still building, stale, or Graphify failing: tools must
   answer plainly and the turn must continue → `test_map_tools_report_not_ready_and_errors` (T4).
4. Model-authored query text reaching a shell → `test_query_is_passed_as_one_argument` (T4).
5. The map refresh racing a commit (`index.lock`) or running after every
   read-only turn → `test_update_runs_only_after_turns_that_changed_files` (T5).

## File Map

| File | Responsibility |
|---|---|
| `src/absolutcoder/skills.py` (new) | Global skills folder: discovery, `always.md` text, install/remove a skill, `<skill-dir>` paths |
| `src/absolutcoder/codemap.py` (new) | Graphify wrapper (`CodeMap`: build, update, query, path, explain, status) and the three model tools |
| `src/absolutcoder/agent.py` | Second backend route, skill sources, always-on text and code-map rule in prompts, code-map tools for main |
| `src/absolutcoder/subagents.py` | Skills for editing roles; `map` tools for roles with `read`; always-on text in their prompts |
| `src/absolutcoder/repl.py` | `/skills`, `/<skill> <request>`, `/map`, map lifecycle (start, after-turn refresh), token line |
| `src/absolutcoder/render.py` | Token totals per turn |
| `src/absolutcoder/config.py` | `code_map`, `code_map_min_files` |
| `src/absolutcoder/git_policy.py` | `graphify-out/` in `.git/info/exclude` (extend `protect_context_file`) |
| `pyproject.toml` | `graphifyy>=0.9,<0.10` |
| `README.md` | Global skills, Hallmark, code map, token line |
| `tests/test_skills.py`, `tests/test_codemap.py` (new); additions to `test_repl.py`, `test_subagents.py`, `test_render.py`, `test_config.py` | |

Reused as they are: `RepoFilesystemBackend` (`shell.py:198`, can be rooted
anywhere), `workspace_backend` (`agent.py:243`), `secret_permissions` and
`role_middleware`/`role_permissions` (`subagents.py`), `config.data_dir`
(`config.py:13`), `protect_context_file` (`git_policy.py`), the `extra_tools`
path of `build_agent`, `StreamRenderer` (`render.py`), `scripted_model(bound=…)`
and `role_models` (`tests/support.py`).

## Interfaces

```python
# skills.py
GLOBAL_ROUTE = "/.absolutcoder-global/skills/"
PROJECT_SOURCE = "/.absolutcoder/skills/"
ALWAYS_LIMIT = 4000
@dataclass(frozen=True)
class SkillInfo: name: str; description: str; scope: str; folder: Path   # scope: "global" | "project"
def global_skills_dir() -> Path                                  # data_dir() / "skills"
def list_skills(repo: Path) -> list[SkillInfo]                   # project entries override global by name
def always_text(repo: Path) -> str                               # global skills' always.md, each capped, joined
def skill_paths_note(repo: Path) -> str                          # prompt line mapping <skill-dir> to real folders
def install_skill(source: Path) -> SkillInfo                     # copy into the global folder; ValueError on bad SKILL.md
def remove_skill(name: str) -> bool

# codemap.py
class CodeMap:
    def __init__(self, repo: Path, *, min_files: int = 20, runner: Callable[..., CompletedProcess] | None = None) -> None
    def status(self) -> str            # "off" | "too small" | "missing" | "building" | "ready" | "failed: …"
    def ensure(self) -> None           # start a background build when missing and large enough
    def refresh(self) -> None          # background `graphify update .` when ready
    def query(self, question: str, budget: int = 2000) -> str
    def path(self, source: str, target: str) -> str
    def explain(self, name: str) -> str
    def close(self) -> None
def make_map_tools(code_map: CodeMap) -> list[BaseTool]   # code_map_query, code_map_path, code_map_explain
CODE_MAP_RULE: str                                         # the prompt paragraph

# agent.py (parameters added)
def build_agent(..., code_map: CodeMap | None = None)
def build_role_graph(..., code_map: CodeMap | None = None)
```

Prompt paragraph `CODE_MAP_RULE` (added only when a map is ready or building):
`A code map of this repository is available. Before your first search or source-file read on a question about existing code, call code_map_query with the question; use code_map_path to see how two things connect and code_map_explain for one symbol. Then read only the files it points to. Put this rule in the brief of any worker that explores code.`

## Tasks

- [ ] **T0 Branch and spike (controller).** Create `phase-6-hallmark-code-map`.
  Install `graphifyy` into the project virtualenv. On a throwaway copy of a
  small project, record for the T4 brief: how to invoke it (`python -m graphify`
  or the console script path), the exact code-only build command and its
  outputs (`graphify extract . --code-only`, `--no-viz`), where it writes, exit
  codes and output shape of `query --budget N`, `path`, `explain` and
  `update .`, timings on AbsolutCoder itself, and whether an output directory
  can be chosen. Also run Hallmark's `sloplint.mjs --tier floor` on a sample
  page to learn whether the Floor tier requires the stamp line. If Graphify
  cannot build offline with no model, stop and return to the user.

- [ ] **T1 Config.** `code_map`, `code_map_min_files` with validation at load.
  Tests (`test_config.py`): defaults when absent; `"code_map": "sometimes"` and
  a non-positive `code_map_min_files` are config errors; round trip.

- [ ] **T2 Global skills.** `skills.py`; in `agent.workspace_backend` add the
  route `GLOBAL_ROUTE → RepoFilesystemBackend(global_skills_dir())` when that
  folder exists; skill sources become
  `[(GLOBAL_ROUTE, "Global"), (PROJECT_SOURCE, "Project")]` (each only when its
  folder exists; project wins by name); deny write on `/.absolutcoder-global/**`
  for every role; `skill_paths_note` and `always_text` appended to the main
  prompt; roles with the `edit` group get the same skill sources
  (`spec["skills"]`) and the always-on text in their prompt, including
  background role graphs. Tests (`test_skills.py`, `test_subagents.py`):
  `test_global_skill_is_listed_and_readable_through_the_route`;
  `test_project_skill_overrides_global_of_same_name`;
  `test_global_route_does_not_shadow_repo_folders`;
  `test_global_skills_are_read_only_to_file_tools`;
  `test_always_text_is_capped_and_project_always_is_ignored`;
  `test_always_text_reaches_main_and_editing_roles_only` (not `explore`, `plan`,
  `web-researcher`); `test_skill_dir_note_names_real_folders`;
  `test_install_skill_copies_and_validates` (skips `fixtures`, `node_modules`,
  `.hallmark`, `*.test.mjs`; rejects a folder with no valid `SKILL.md`);
  `test_remove_skill`.

- [ ] **T3 Skill commands and Hallmark.** In the loop: `/skills` (name, scope,
  description), `/skills add <path>`, `/skills remove <name>` (rebuild the
  graph afterwards), and `/<skill-name> <request>` which runs the request with
  that skill pinned (deepagents `pinned_skills`); an unknown `/name` keeps the
  current "Commands:" reply. One-shot runs add the prompt sentence
  `Nobody can answer questions in this run; decide and say what you assumed.`
  Then install Hallmark into `~/.absolutcoder/skills/hallmark` with
  `install_skill`, write its `always.md` (the Floor rules adapted to plain
  HTML/CSS/JSX work, with the real lint command from T0's finding), and change
  the copy's description to
  `Full design flow for new websites and pages. Use only when the user names Hallmark, for example /hallmark <request>.`
  Tests (`test_repl.py`): `test_skills_command_lists_scopes`;
  `test_skill_slash_runs_request_with_skill_loaded` (the scripted model's input
  contains the skill body and the request); `test_unknown_slash_is_not_a_skill`;
  `test_one_shot_prompt_says_nobody_can_answer`. Hallmark's own install is
  verified by hand (see Verification).

- [ ] **T4 Code map wrapper and tools.** `codemap.py` per T0's findings; three
  tools for the main model and for every role with the `read` group (new tool
  group `map`, added automatically with `read`); not for `web-researcher`.
  Results are cut to a character budget with a truncation note. Tests
  (`test_codemap.py`, fake `runner`): `test_build_command_is_code_only_and_offline`
  (environment has `GRAPHIFY_BACKEND=none` and the log opt-out);
  `test_query_is_passed_as_one_argument`;
  `test_map_tools_report_not_ready_and_errors` (missing, building, non-zero
  exit, timeout, package missing);
  `test_too_small_folder_is_not_mapped`; `test_build_runs_in_background_once`;
  `test_real_graphify_builds_and_answers` (skipped without the package: a
  five-file project, then a query names the right file). `test_subagents.py`:
  `test_map_tools_on_reading_roles_not_on_web_role`.

- [ ] **T5 Lifecycle, prompt and `/map`.** `Repl` owns one `CodeMap` (when
  `code_map` is `"auto"`): `ensure()` at start, a `map: building…` /
  `map: ready` line, `refresh()` after any turn whose tool calls included
  `write_file`, `edit_file`, `delete` or `execute`, under the repository git
  lock; `CODE_MAP_RULE` in the main prompt and in reading roles' prompts once a
  map is ready or building; `graphify-out/` excluded from git; `/map` (status),
  `/map build`, `/map update`, `/map off`, `/map on`; `close()` on exit. Tests
  (`test_repl.py`): `test_map_rule_only_when_a_map_exists`;
  `test_update_runs_only_after_turns_that_changed_files`;
  `test_map_commands`; `test_code_map_off_adds_no_tools_or_rule`;
  `test_graphify_out_is_git_excluded`.

- [ ] **T6 Token line.** `StreamRenderer` sums `usage_metadata` from every
  model message of the turn (main and workers, de-duplicated by message id) and
  `finish` prints
  `tokens: {input} in ({cached} cached), {output} out, {calls} model calls`.
  Tests (`test_render.py`): totals across main and a worker; a message seen
  twice is counted once; no line when no usage was reported.

- [ ] **T7 README, final review, merge.** README sections for global skills,
  Hallmark (`/hallmark`), the code map and the token line. One whole-branch
  review over `04e6e18..HEAD` with one fix pass and a scoped re-review. Then
  fast-forward `main` to the branch tip in the main checkout and run
  `pip install -e .` with the system Python that provides `absolut`, and confirm
  `absolut --help` shows the new build. If the session's worktree isolation
  blocks the merge, give the user the two commands instead of working around it.

## Not in this plan

The other speed findings from the observed run (parallel delegation by default,
bulk writing by workers, progress during long file writes, a workspace snapshot
up front, batched edits, screenshots stored outside the session database). They
are separate work and likely the larger token saving for new-project tasks.

## Verification

After every task: the full suite is green.

End to end, by hand with the real key, in `C:\Users\priya\AbsolutTests`:

- **Hallmark core rules:** ask for a small web page without naming Hallmark.
  The page's CSS declares colours and fonts as tokens, has no `transition: all`
  and no purple-to-blue gradient; run
  `node ~/.absolutcoder/skills/hallmark/scripts/sloplint.mjs <folder> --tier floor`
  and see no FAIL lines.
- **Full Hallmark:** `/hallmark make a landing page for a bakery`; the model
  reads the skill, deals a theme and prints its Picks block.
- **Code map:** start in `C:\Users\priya\AbsolutCoder` (over 20 source files):
  see `map: building…` then `map: ready`; ask "where are shell commands vetted
  before they run?"; the tool lines show `code_map_query` before any `grep` or
  `read_file`. Repeat with `/map off` and compare the two `tokens:` lines.
- **Small folder:** start in an empty folder; no map is built and no
  `graphify-out/` appears.
- **Merge:** plain `absolut --help` lists `/skills` and `/map`.

## Execution

Subagent-driven, tasks in order T0 to T7, without stopping between tasks; the
user reviews at the end.
