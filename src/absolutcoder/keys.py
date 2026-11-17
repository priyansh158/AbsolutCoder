"""Arrow keys read while a turn runs, so the activity view can be expanded.

A listener runs a daemon thread that reads the keyboard without echoing and
calls ``on_key`` with ``"up"``, ``"down"``, ``"left"`` or ``"right"``; every
other key is dropped. ``stop()`` returns within about 0.2 seconds and leaves
no thread reading the keyboard, so the next prompt gets every keystroke.
Ctrl+C still interrupts the turn: on Windows ``getwch`` never reads it, and on
POSIX cbreak mode keeps signal keys working.
"""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Callable
from typing import Protocol

POLL_SECONDS = 0.05
STOP_SECONDS = 0.2

_WINDOWS_PREFIXES = ("\x00", "\xe0")
_WINDOWS_ARROWS = {"H": "up", "P": "down", "K": "left", "M": "right"}
_POSIX_ARROWS = {b"A": "up", b"B": "down", b"C": "right", b"D": "left"}


class KeyListener(Protocol):
    def start(self, on_key: Callable[[str], None]) -> None: ...

    def stop(self) -> None: ...


class NoKeys:
    """A listener that never reports a key (no terminal, or no way to read one)."""

    def start(self, on_key: Callable[[str], None]) -> None:
        return None

    def stop(self) -> None:
        return None


def decode_windows(chars: str) -> list[str]:
    """Arrow keys in characters read with ``msvcrt.getwch``; other keys are dropped.

    An arrow arrives as a prefix (``"\\x00"`` or ``"\\xe0"``) and then a letter.
    """

    keys: list[str] = []
    index = 0
    while index < len(chars):
        char = chars[index]
        if char in _WINDOWS_PREFIXES:
            follow = chars[index + 1 : index + 2]
            key = _WINDOWS_ARROWS.get(follow)
            if key is not None:
                keys.append(key)
            index += 2  # the second half of a function key is never a key of its own
            continue
        index += 1
    return keys


class PosixDecoder:
    """Turn bytes read from a cbreak terminal into arrow keys.

    Arrows are ``ESC [ A..D`` (and ``ESC O A..D`` in application mode). A
    sequence split across reads is kept until the rest arrives; other bytes,
    and other escape sequences, are dropped.
    """

    def __init__(self) -> None:
        self._pending = b""

    def feed(self, data: bytes) -> list[str]:
        buffer = self._pending + data
        self._pending = b""
        keys: list[str] = []
        index = 0
        while index < len(buffer):
            if buffer[index : index + 1] != b"\x1b":
                index += 1
                continue
            rest = buffer[index + 1 : index + 3]
            if len(rest) < 2 and (rest in (b"", b"[", b"O")):
                self._pending = buffer[index:]  # wait for the rest of the sequence
                break
            if rest[:1] in (b"[", b"O"):
                key = _POSIX_ARROWS.get(rest[1:2])
                if key is not None:
                    keys.append(key)
                    index += 3
                    continue
                index += 2  # some other sequence: skip its introducer
                continue
            index += 1
        return keys


def decode_posix(data: bytes) -> list[str]:
    """Arrow keys in one complete chunk of bytes."""

    return PosixDecoder().feed(data)


class _ThreadListener:
    """Shared start/stop for the platform listeners: one daemon thread, one stop event."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self, on_key: Callable[[str], None]) -> None:
        if self._thread is not None:
            return
        self._stop = threading.Event()
        self._open()
        thread = threading.Thread(
            target=self._guarded, args=(on_key, self._stop), name="absolutcoder-keys", daemon=True
        )
        self._thread = thread
        try:
            thread.start()
        except BaseException:
            self._thread = None
            self._close()
            raise

    def stop(self) -> None:
        thread = self._thread
        self._thread = None
        self._stop.set()
        try:
            if thread is not None and thread is not threading.current_thread():
                thread.join(STOP_SECONDS)
        finally:
            self._close()

    def _guarded(self, on_key: Callable[[str], None], stop: threading.Event) -> None:
        try:
            self._run(on_key, stop)
        except Exception:  # noqa: BLE001 - a keyboard error only means the view cannot expand
            pass

    def _emit(self, on_key: Callable[[str], None], keys: list[str]) -> None:
        for key in keys:
            try:
                on_key(key)
            except Exception:  # noqa: BLE001 - the view records its own errors
                pass

    # subclasses
    def _open(self) -> None: ...

    def _close(self) -> None: ...

    def _run(self, on_key: Callable[[str], None], stop: threading.Event) -> None: ...


class WindowsKeys(_ThreadListener):
    """Poll ``msvcrt.kbhit`` every 50 ms; read with ``getwch`` (no echo)."""

    def __init__(self) -> None:
        super().__init__()
        import msvcrt  # noqa: PLC0415 - Windows only

        self._msvcrt = msvcrt

    def _run(self, on_key: Callable[[str], None], stop: threading.Event) -> None:
        msvcrt = self._msvcrt
        while not stop.is_set():
            if not msvcrt.kbhit():
                stop.wait(POLL_SECONDS)
                continue
            char = msvcrt.getwch()
            if char in _WINDOWS_PREFIXES:
                char += msvcrt.getwch()
            if stop.is_set():
                return
            self._emit(on_key, decode_windows(char))


class PosixKeys(_ThreadListener):
    """stdin in cbreak mode, read with ``select``; the saved attributes always come back."""

    def __init__(self, fd: int | None = None) -> None:
        super().__init__()
        import select  # noqa: PLC0415 - POSIX only
        import termios  # noqa: PLC0415
        import tty  # noqa: PLC0415

        self._select = select
        self._termios = termios
        self._tty = tty
        self._fd = sys.stdin.fileno() if fd is None else fd
        self._saved = None
        self._lock = threading.Lock()

    def _open(self) -> None:
        with self._lock:
            if self._saved is not None:
                return
            saved = self._termios.tcgetattr(self._fd)
            try:
                self._tty.setcbreak(self._fd, self._termios.TCSANOW)
            except BaseException:
                self._termios.tcsetattr(self._fd, self._termios.TCSADRAIN, saved)
                raise
            self._saved = saved

    def _close(self) -> None:
        with self._lock:
            saved, self._saved = self._saved, None
            if saved is not None:
                try:
                    self._termios.tcsetattr(self._fd, self._termios.TCSADRAIN, saved)
                except Exception:  # noqa: BLE001 - the terminal is gone; nothing to restore
                    pass

    def _run(self, on_key: Callable[[str], None], stop: threading.Event) -> None:
        decoder = PosixDecoder()
        while not stop.is_set():
            ready, _, _ = self._select.select([self._fd], [], [], POLL_SECONDS)
            if not ready or stop.is_set():
                continue
            data = os.read(self._fd, 64)
            if not data:
                return
            self._emit(on_key, decoder.feed(data))


def default_key_listener() -> KeyListener:
    """The listener for this platform; a no-op one without a keyboard to read."""

    try:
        if not sys.stdin.isatty():
            return NoKeys()
        if sys.platform == "win32":
            return WindowsKeys()
        return PosixKeys()
    except Exception:  # noqa: BLE001 - an import error or an odd stdin: the view still works
        return NoKeys()
