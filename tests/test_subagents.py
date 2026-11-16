"""The worker roster: role tool sets, parallel delegation and approvals."""

import io

from langchain_core.messages import AIMessage
from rich.console import Console

from absolutcoder.config import Config
from absolutcoder.repl import Repl
from tests.support import (
    last_human_text,
    role_models,
    scripted_model,
    task_call,
    tool_call,
    tool_calls,
    tool_names_since_last_human,
)


def _console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, no_color=True, width=100)


def _task_messages(messages) -> list:
    return [m for m in messages if getattr(m, "type", None) == "tool" and m.name == "task"]


def test_parallel_subagent_interrupts_resume_by_id(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def main(messages):
        if len(_task_messages(messages)) >= 2:
            return AIMessage(content="all done")
        return tool_calls(
            task_call("general-purpose", "run: echo one", "t1"),
            task_call("general-purpose", "run: echo two", "t2"),
        )

    def worker(messages):
        if "execute" not in tool_names_since_last_human(messages):
            command = last_human_text(messages).partition("run:")[2].strip()
            return tool_call("execute", {"command": command}, "exec-" + command.split()[-1])
        return AIMessage(content="ran")

    output = io.StringIO()
    prompts: list[str] = []

    def ask(prompt=""):
        # Repl prints "Run command? <cmd>" and then asks "[y/n] " once per approval.
        if prompt.startswith("[y/n]"):
            prompts.append(output.getvalue().splitlines()[-1])
        return "y"

    repl = Repl(
        repo,
        Config(main_model="openai:main-test", subagent_model="openai:worker-test", auto_run=False),
        chat_model=scripted_model(main),
        model_factory=role_models({}, default=scripted_model(worker)),
        input_fn=ask,
        console=Console(file=output, force_terminal=False, no_color=True, width=100),
    )
    snapshot = repl.run_turn("run both")
    assert snapshot is not None
    assert len(prompts) == 2
    assert all(line.startswith("Run command?") for line in prompts)
    allowed = {entry["command"] for entry in repl.turn_vet_log if entry["allowed"]}
    assert any("echo one" in command for command in allowed)
    assert any("echo two" in command for command in allowed)
    assert snapshot.values["messages"][-1].content == "all done"
