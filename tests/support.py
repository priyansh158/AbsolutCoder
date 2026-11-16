"""Fake chat models for tests. They never call the network."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from absolutcoder.models import ModelFactory


def scripted_model(responder, *, bound: list | None = None):
    """Return a chat model whose replies come from ``responder(messages)``.

    When ``bound`` is given, every ``bind_tools`` call appends the list of tool
    names it was given.
    """

    class FakeChatModel(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "fake"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=responder(messages))])

        def bind_tools(self, tools, **kwargs):
            if bound is not None:
                bound.append([_tool_name(tool) for tool in tools])
            return self

    return FakeChatModel()


def _tool_name(tool) -> str:
    if isinstance(tool, dict):
        return str(tool.get("name") or tool.get("function", {}).get("name", ""))
    return str(getattr(tool, "name", ""))


def tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def task_call(agent: str, description: str, call_id: str) -> dict:
    """One tool-call dict for the ``task`` tool."""

    return {
        "name": "task",
        "args": {"description": description, "subagent_type": agent},
        "id": call_id,
        "type": "tool_call",
    }


def tool_calls(*calls: dict) -> AIMessage:
    """One AIMessage carrying several tool calls."""

    return AIMessage(content="", tool_calls=list(calls))


def last_human_text(messages) -> str:
    for message in reversed(messages):
        if getattr(message, "type", None) != "human":
            continue
        content = message.content
        if isinstance(content, str):
            return content
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(parts)
    return ""


def tool_names_since_last_human(messages) -> list[str]:
    names: list[str] = []
    for message in reversed(messages):
        if getattr(message, "type", None) == "human":
            break
        if getattr(message, "type", None) == "tool":
            names.append(message.name)
    names.reverse()
    return names


_NO_DEFAULT = object()


def role_models(
    mapping: dict[str, BaseChatModel], *, seen: list | None = None, default=_NO_DEFAULT
) -> ModelFactory:
    """Return a model factory that hands out ``mapping[role]`` and records requests.

    A role missing from ``mapping`` gets ``default`` when one is given, else KeyError.
    """

    def factory(role: str, model_id: str):
        if seen is not None:
            seen.append((role, model_id))
        if role not in mapping and default is not _NO_DEFAULT:
            return default
        return mapping[role]

    return factory
