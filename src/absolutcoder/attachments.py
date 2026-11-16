"""Turn user text and file paths into message content blocks."""

from __future__ import annotations

import base64
from pathlib import Path

_IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def build_user_content(text: str, paths: list[Path]) -> list[dict]:
    """Build the user message.

    Image suffixes become image blocks. Every other path becomes a text line
    naming the path so the agent can read it with ``read_file``.
    """

    blocks: list[dict] = [{"type": "text", "text": text}]
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        mime = _IMAGE_TYPES.get(path.suffix.lower())
        if mime is not None:
            blocks.append({"type": "text", "text": f"Image attachment path: {path}"})
            blocks.append(
                {
                    "type": "image",
                    "mime_type": mime,
                    "base64": base64.b64encode(path.read_bytes()).decode("ascii"),
                    "path": str(path),
                }
            )
            continue
        blocks.append(
            {
                "type": "text",
                "text": f"Attachment path: {path}. Read it with read_file.",
            }
        )
    return blocks
