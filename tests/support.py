"""Fake chat models for tests. They never call the network."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


def scripted_model(responder):
    """Return a chat model whose replies come from ``responder(messages)``."""

    class FakeChatModel(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "fake"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=responder(messages))])

        def bind_tools(self, tools, **kwargs):
            return self

    return FakeChatModel()


def tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


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
