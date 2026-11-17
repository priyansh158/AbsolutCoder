"""Keep images the model has already seen out of the conversation history.

``read_file`` on a PNG returns a ``ToolMessage`` whose content holds an image
block with the file's base64 data (deepagents: ``{"type": "image", "base64":
..., "mime_type": "image/png"}``, with the path in ``additional_kwargs``).
Left alone, that data is stored at every checkpoint and sent again with every
later model call. Once a later model call has answered after it, the model has
seen the image, so the image blocks are replaced with a short note.
"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage

PLACEHOLDER = "[image from {source} was shown to you earlier; read the file again to see it]"
_IMAGE_TYPES = frozenset({"image", "image_url"})
_DATA_KEYS = ("base64", "data")


def is_image_block(block: Any) -> bool:
    """An image content block in any of the shapes langchain and the providers use."""

    if not isinstance(block, dict):
        return False
    if block.get("type") in _IMAGE_TYPES:
        return True
    mime = str(block.get("mime_type") or block.get("media_type") or "")
    if mime.startswith("image/") and any(block.get(key) for key in _DATA_KEYS):
        return True
    source = block.get("source")
    if isinstance(source, dict) and str(source.get("media_type", "")).startswith("image/"):
        return True
    url = block.get("url") or block.get("image_url")
    if isinstance(url, dict):
        url = url.get("url")
    return isinstance(url, str) and url.startswith("data:image/")


def _source(message: ToolMessage, blocks: list) -> str:
    extra = message.additional_kwargs if isinstance(message.additional_kwargs, dict) else {}
    path = extra.get("read_file_path")
    if not path:
        path = next(
            (b.get("path") or b.get("file_path") for b in blocks if isinstance(b, dict) and (b.get("path") or b.get("file_path"))),
            None,
        )
    return str(path or message.name or "a tool")


def without_images(message: ToolMessage) -> ToolMessage | None:
    """``message`` with its image blocks replaced by one note; None when it has none."""

    content = message.content
    if not isinstance(content, list) or not any(is_image_block(block) for block in content):
        return None
    note = {"type": "text", "text": PLACEHOLDER.format(source=_source(message, content))}
    kept: list = []
    for block in content:
        if not is_image_block(block):
            kept.append(block)
        elif note is not None:
            kept.append(note)  # the first image's place; later images are dropped
            note = None
    return ToolMessage(
        content=kept,
        id=message.id,
        tool_call_id=message.tool_call_id,
        name=message.name,
        status=message.status,
        artifact=message.artifact,
        additional_kwargs=dict(message.additional_kwargs or {}),
        response_metadata=dict(message.response_metadata or {}),
    )


def seen_image_update(messages: list) -> dict | None:
    """The state update that trims every image tool result the model has already answered after."""

    last_ai = None
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], AIMessage):
            last_ai = index
            break
    if last_ai is None:
        return None
    replaced = []
    for message in messages[:last_ai]:
        if not isinstance(message, ToolMessage) or not message.id:
            continue  # without an id the reducer could not replace it
        trimmed = without_images(message)
        if trimmed is not None:
            replaced.append(trimmed)
    return {"messages": replaced} if replaced else None


class SeenImagesMiddleware(AgentMiddleware):
    """Before each model call, replace images from earlier calls with a note (same message id)."""

    def before_model(self, state, runtime) -> dict[str, Any] | None:  # noqa: ANN001
        try:
            messages = state.get("messages") if isinstance(state, dict) else getattr(state, "messages", None)
            if not isinstance(messages, list):
                return None
            return seen_image_update(messages)
        except Exception:  # noqa: BLE001 - trimming history must never stop a model call
            return None

    async def abefore_model(self, state, runtime) -> dict[str, Any] | None:  # noqa: ANN001
        return self.before_model(state, runtime)
