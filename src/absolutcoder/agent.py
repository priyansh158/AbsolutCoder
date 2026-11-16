"""Build the AbsolutCoder deep agent and its one worker."""

from __future__ import annotations

from pathlib import Path

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import CompositeBackend
from langchain.agents.middleware import TodoListMiddleware
from langgraph.graph.state import CompiledStateGraph

from absolutcoder.config import Config
from absolutcoder.git_policy import execute_needs_approval, protect_context_file
from absolutcoder.models import ModelFactory, ProviderMissing, build_model, install_hint
from absolutcoder.shell import GuardedShellBackend, RepoFilesystemBackend

MEMORY_PATH = "/AbsolutCoder.md"
SECRET_PATHS = [
    "/.env",
    "/**/.env",
    "/**/.ssh/**",
    "/**/*credentials*",
]

SYSTEM_PROMPT = """You are AbsolutCoder, a local coding agent working in this repository.

Plan longer tasks with the todo list. Delegate investigation and multi-step edits to the general-purpose worker with the task tool, read the worker's report, and decide what lands in the repo.

Update /AbsolutCoder.md as you learn the project. That file is local memory. It stays untracked: never add it, commit it, or push it.

Commit with the git identity already configured on this machine. Never run git config. Never set GIT_AUTHOR_* or GIT_COMMITTER_* and never pass --author. Do not add Co-authored-by, Assisted-by, or Generated-by trailers, and do not attribute work to AbsolutCoder.

You may create branches, commit, and merge locally, including into local main, unless the user said otherwise. Push, including force-push, only when the user explicitly asks to push. Pull or fetch only when the user asks to pull or fetch. Do not run git reset --hard or git clean -fd unless the user asks for that operation.
"""

WORKER_PROMPT = """You are AbsolutCoder's worker. Investigate the repo, edit files, and run commands, then report what you changed and what you verified.

Do not commit AbsolutCoder.md. Do not change git config, author environment variables, or commit attribution. Push only when the task says the user asked to push.
"""


def secret_permissions() -> list[FilesystemPermission]:
    """Deny reads and writes of env files, ssh keys, and credential paths."""

    return [
        FilesystemPermission(operations=["read", "write"], paths=list(SECRET_PATHS), mode="deny")
    ]


def build_agent(
    repo: Path | str,
    config: Config,
    checkpointer,
    *,
    chat_model=None,
    model_factory: ModelFactory | None = None,
    backend: GuardedShellBackend | None = None,
) -> CompiledStateGraph:
    """Compile one deep agent. The main model supervises a single worker.

    The installed deepagents release rejects filesystem permissions on a shell
    backend unless every rule sits under a CompositeBackend route. Files are
    routed at ``/`` so the secret-path denies apply, and the guarded shell stays
    the default backend so ``execute`` still runs on this machine. The route
    backend strips Windows ``\\\\?\\`` prefixes before the root check.
    """

    root = Path(repo).resolve()
    protect_context_file(root)
    routed = workspace_backend(root, backend)
    factory = model_factory or default_model_factory(config)
    try:
        return _compile(root, config, checkpointer, routed, chat_model, factory)
    except ImportError as error:
        hint = install_hint(error)
        if hint is None:
            raise
        raise ProviderMissing(f"{error}\nInstall it with: {hint}") from error


def default_model_factory(config: Config) -> ModelFactory:
    """Build models from the configured provider settings."""

    def factory(role: str, model_id: str):
        return build_model(model_id, config.providers)

    return factory


def _compile(root, config, checkpointer, routed, chat_model, factory) -> CompiledStateGraph:
    main_model = chat_model if chat_model is not None else factory("main", config.main_model)
    worker_model = factory(
        "general-purpose", config.agent_models.get("general-purpose", config.subagent_model)
    )
    interrupt_on = None if config.auto_run else {"execute": _execute_interrupt()}
    kwargs = {
        "model": main_model,
        "system_prompt": SYSTEM_PROMPT,
        "middleware": [TodoListMiddleware()],
        "subagents": [
            {
                "name": "general-purpose",
                "description": (
                    "The only worker. Use it for research and multi-step repo edits. "
                    "It reports back; you decide what lands in the repository."
                ),
                "system_prompt": WORKER_PROMPT,
                "model": worker_model,
            }
        ],
        "memory": [MEMORY_PATH],
        "permissions": secret_permissions(),
        "backend": routed,
        "checkpointer": checkpointer,
    }
    if interrupt_on is not None:
        kwargs["interrupt_on"] = interrupt_on
    return create_deep_agent(**kwargs)


def workspace_backend(root: Path | str, shell: GuardedShellBackend | None = None) -> CompositeBackend:
    """Route repo files at ``/`` and keep command execution on the guarded shell."""

    guarded = shell or GuardedShellBackend(root_dir=root, virtual_mode=True, inherit_env=True)
    filesystem = RepoFilesystemBackend(root_dir=root, virtual_mode=True)
    return CompositeBackend(default=guarded, routes={"/": filesystem})


def _execute_interrupt() -> dict:
    return {
        "allowed_decisions": ["approve", "reject"],
        "description": "Approve this shell command before it runs.",
        "when": _execute_needs_prompt,
    }


def _execute_needs_prompt(request) -> bool:
    tool_call = getattr(request, "tool_call", None) or {}
    args = tool_call.get("args", {}) if isinstance(tool_call, dict) else {}
    command = str(args.get("command", ""))
    return execute_needs_approval(command)
