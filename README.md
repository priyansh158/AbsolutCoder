# AbsolutCoder

A terminal coding assistant. It plans work, reads and edits files in the
repository you start it in, runs shell commands you approve, delegates to
worker roles in parallel, can search and read the web, and commits with your
own git identity.

It runs on OpenAI out of the box and on any other model provider you configure.

## Install

Python 3.12 or newer.

```
pip install -e .
pip install -e ".[anthropic]"   # optional: Anthropic models
pip install -e ".[google]"      # optional: Google models
pip install -e ".[ollama]"      # optional: Ollama models
```

## Run

```
absolut                     # interactive loop in the current directory
absolut "fix the failing test"            # one task, then exit
absolut "make the tests pass" --until "python -m pytest -q"
```

The first launch asks for two model ids (main and worker) and saves them. Set
the API key for the provider you chose first, for example `OPENAI_API_KEY`.

AbsolutCoder creates `AbsolutCoder.md` in the directory you start it in and
uses it as project memory. That file is kept out of git.

## Models and providers

A model id is `provider:model`. A bare id means OpenAI.

| You type | Provider | Key it needs |
|---|---|---|
| `gpt-…` or `openai:gpt-…` | OpenAI | `OPENAI_API_KEY` |
| `anthropic:<model-id>` | Anthropic | `ANTHROPIC_API_KEY` |
| `google_genai:<model-id>` | Google | `GOOGLE_API_KEY` or `GEMINI_API_KEY` |
| `ollama:llama3:8b` | Ollama | none |

Switch during a session with `/model <id>` and `/subagent <id>`. If a
provider's package is missing you get the `pip install` line to run.

### OpenAI-compatible servers

Declare the server under `providers` in `config.json` and use its name as the
prefix. `"kind": "openai"` is required.

```json
{
  "main_model": "local:llama3",
  "subagent_model": "local:llama3",
  "providers": {
    "local": {
      "kind": "openai",
      "base_url": "http://localhost:11434/v1",
      "api_key_env": null
    }
  }
}
```

Your OpenAI key is never sent to such a server. Set `"api_key_env"` to the name
of an environment variable if the server needs its own key. These servers are
called with the Chat Completions API unless you add `"api": "responses"`.

## Settings

`config.json` lives in `~/.absolutcoder`, or in `$ABSOLUTCODER_HOME` when set.

| Key | Default | Meaning |
|---|---|---|
| `main_model`, `subagent_model` | asked on first launch | Models for the main loop and for workers |
| `agent_models` | `{}` | A model per worker role, for example `{"explore": "openai:…"}` |
| `providers` | `{}` | Named endpoints, see above |
| `auto_run` | `false` | Run shell commands without asking (`/auto`) |
| `shell` | `"auto"` | `auto`, `bash`, `sh`, `cmd`, or a path to a POSIX shell |
| `verify_command` | none | Check command run after every request (`/verify`) |
| `verify_timeout` | `1800` | Seconds the check command may run |
| `max_rounds` | `5` | Turns the check loop may take |
| `web_max_chars` | `20000` | Most text returned from one web page |
| `max_background_tasks` | `4` | Background tasks running at once |

Sessions are stored in the same folder; `/resume` lists the ones for the
current repository.

## In the loop

| Command | What it does |
|---|---|
| `/model <id>`, `/subagent <id>` | Switch models |
| `/auto [on\|off]` | Stop or resume asking before shell commands |
| `/agents` | List worker roles |
| `/tasks`, `/tasks show <id>`, `/tasks cancel <id>` | Background tasks |
| `/verify <command>`, `/verify off`, `/verify` | Set, clear or show the check command |
| `/loop <N>(s\|m\|h) <prompt>` | Repeat a prompt until Ctrl+C |
| `/attach <path>` | Attach a file to the next message |
| `/resume [N]` | Resume an earlier session |
| `/exit`, `/quit` | Leave |

Ctrl+C during a turn stops that turn and keeps the session. Ctrl+C at the
prompt leaves.

## Approvals and the git policy

By default you are asked before each shell command, except plain local
`git commit`, `git branch`, `git switch` and `git merge`. `/auto` or `--auto`
turns the asking off; the git policy below still applies.

- **Push, pull, fetch** run only when your own request asks for them. Say
  "push", "push it", "commit and push the changes" or "git push". "Push this
  logic into a helper" does not count.
- **`git config`, `git reset --hard`, forced `git clean`** run only when your
  request asks for them.
- **Commits** use the git identity configured on your machine. Attribution
  trailers (`Co-authored-by` and similar) are removed from messages.
- Text that did not come from you (check output fed back to the model,
  background task results, web pages) never unlocks any of the above.

The policy is a pattern check on the command text. It looks inside `sh -c`,
`eval`, `env` and similar wrappers, and refuses what it cannot see into, but it
is not a sandbox: a script file or another interpreter can run git without it
noticing. Keep approvals on in repositories you do not trust.

## Worker roles

The main model can start several workers at once.

| Role | Can do |
|---|---|
| `explore` | Read and search the repository |
| `plan` | Read and search, return a plan |
| `general-purpose` | Edit files and run commands |
| `web-researcher` | Search and read the public web; cannot see the repository |

Web content reaches the main model only as marked, untrusted text from
`web-researcher`, which has no file or shell access.

### Your own roles

Put a markdown file in `<repo>/.absolutcoder/agents/` or
`~/.absolutcoder/agents/`. A repository file overrides a home file of the same
name, which overrides a built-in role. Restart to load changes.

```
---
name: reviewer
description: Reviews a diff and lists problems with file and line.
tools: [read, search]
model: anthropic:<model-id>
---
You review code. Report problems; do not change files.
```

`tools` takes groups: `read`, `search`, `edit`, `shell`, `web`. `web` cannot be
combined with the others. `name` and `model` are optional.

Skills in `<repo>/.absolutcoder/skills/<name>/SKILL.md` are offered to the main
model.

## Check loop

With a check command set (`/verify <command>`, or `--until` for one run), the
command runs after every request. When it fails, its output goes back to the
model, up to `max_rounds` turns.

```
absolut "make the tests pass" --until "python -m pytest -q" --max-rounds 8
```

Exit codes for a one-shot run: `0` done, `1` the turn failed, `2` the check
still fails, `130` interrupted.

`/verify` is saved in `config.json` and applies in every repository until
`/verify off`.

## Background tasks

In the interactive loop the main model can start a worker in the background
and keep talking to you. A line appears when a task changes state, and its
result is given to the model at the start of your next request. If a task needs
approval for a command, you are asked the next time you press Enter. `/exit`
cancels running tasks.

## Without a terminal

Piped output is UTF-8. With no terminal to ask on, commands that need approval
are rejected; pass `--auto` to allow them for that run.

## Development

```
pip install -e ".[dev]"
python -m pytest -q
```

The tests run offline with scripted models.
