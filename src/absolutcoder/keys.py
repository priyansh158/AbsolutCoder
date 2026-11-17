"""Arrow keys read while a turn runs, so the activity view can be expanded.

A listener runs a daemon thread that reads the keyboard without echoing and
calls ``on_key`` with ``"up"``, ``"down"``, ``"left"`` or ``"right"``.
``stop()`` returns within about 0.2 seconds and leaves no thread reading the
keyboard. On Windows only arrow records are taken from the console input
queue, so anything typed during a turn stays for the next prompt; on POSIX
other keys are read and dropped. Ctrl+C still interrupts the turn: on Windows
the console handles it outside the input queue, and on POSIX cbreak mode keeps
signal keys working.
"""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

POLL_SECONDS = 0.05
STOP_SECONDS = 0.2

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


VK_ARROWS = {0x26: "up", 0x28: "down", 0x25: "left", 0x27: "right"}
# Shift, Ctrl, Alt, the Windows keys, Caps/Num/Scroll Lock: a key-down of one of these
# alone types nothing, so prompt_toolkit would drop it too.
_VK_MODIFIERS = frozenset({0x10, 0x11, 0x12, 0x14, 0x5B, 0x5C, 0x90, 0x91, 0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5})
KEY_EVENT = 0x0001
MOUSE_EVENT = 0x0002
WINDOW_BUFFER_SIZE_EVENT = 0x0004
MENU_EVENT = 0x0008
FOCUS_EVENT = 0x0010
_PEEK_LIMIT = 64


@dataclass(frozen=True)
class ConsoleEvent:
    """The parts of a Windows console input record the reader decides on."""

    event_type: int
    key_down: bool = False
    vk: int = 0
    char: str = "\0"


def plan_console_events(events: Sequence[ConsoleEvent]) -> tuple[list[str], int]:
    """Which arrow keys the first records hold, and how many records to take out of the queue.

    Taken: arrow keys (reported on key-down), key-up records (prompt_toolkit
    ignores them, and the Enter that submitted the prompt leaves one behind),
    a modifier pressed alone, and focus and menu records. The first record of
    anything else (a typed character such as ``à``, Enter, a mouse or window
    event) stops the plan: it and everything after it stay for the next prompt.
    """

    keys: list[str] = []
    taken = 0
    for event in events:
        if event.event_type in (FOCUS_EVENT, MENU_EVENT):
            taken += 1
            continue
        if event.event_type != KEY_EVENT:
            break
        arrow = VK_ARROWS.get(event.vk)
        if arrow is not None:
            if event.key_down:
                keys.append(arrow)
            taken += 1
            continue
        if not event.key_down:
            taken += 1
            continue
        if event.vk in _VK_MODIFIERS and event.char == "\0":
            taken += 1
            continue
        break
    return keys, taken


def _console_types():
    import ctypes  # noqa: PLC0415 - Windows only
    from ctypes import wintypes  # noqa: PLC0415

    class KEY_EVENT_RECORD(ctypes.Structure):  # noqa: N801 - the Win32 name
        _fields_ = [
            ("bKeyDown", wintypes.BOOL),
            ("wRepeatCount", wintypes.WORD),
            ("wVirtualKeyCode", wintypes.WORD),
            ("wVirtualScanCode", wintypes.WORD),
            ("uChar", wintypes.WCHAR),
            ("dwControlKeyState", wintypes.DWORD),
        ]

    class _EVENT(ctypes.Union):
        # KEY_EVENT_RECORD and MOUSE_EVENT_RECORD are 16 bytes, the largest members.
        _fields_ = [("KeyEvent", KEY_EVENT_RECORD), ("raw", ctypes.c_byte * 16)]

    class INPUT_RECORD(ctypes.Structure):  # noqa: N801
        _fields_ = [("EventType", wintypes.WORD), ("Event", _EVENT)]

    return ctypes, wintypes, KEY_EVENT_RECORD, INPUT_RECORD


class WindowsConsoleKeys(_ThreadListener):
    """Read arrow keys from the console input records with kernel32; never blocks.

    Every 50 ms the thread peeks at the waiting records and reads only those
    ``plan_console_events`` takes, which are already there, so no call waits.
    The console mode is left alone: with ``ENABLE_PROCESSED_INPUT`` (the
    normal mode outside a prompt) Ctrl+C is handled by the system, never put
    in the input queue, and still raises KeyboardInterrupt in the main thread.
    Raises OSError when there is no console to read (redirected stdin).
    """

    def __init__(self, handle: int | None = None) -> None:
        super().__init__()
        ctypes, wintypes, _key, record = _console_types()
        self._ctypes = ctypes
        self._wintypes = wintypes
        self._record = record
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetStdHandle.argtypes = [wintypes.DWORD]
        kernel32.GetStdHandle.restype = wintypes.HANDLE
        for name in ("PeekConsoleInputW", "ReadConsoleInputW"):
            function = getattr(kernel32, name)
            function.argtypes = [wintypes.HANDLE, ctypes.POINTER(record), wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
            function.restype = wintypes.BOOL
        kernel32.GetNumberOfConsoleInputEvents.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.GetNumberOfConsoleInputEvents.restype = wintypes.BOOL
        self._kernel32 = kernel32
        if handle is None:
            handle = kernel32.GetStdHandle(wintypes.DWORD(-10 & 0xFFFFFFFF))  # STD_INPUT_HANDLE
        if not handle or handle == ctypes.c_void_p(-1).value:
            raise OSError("no console input handle")
        self._handle = handle
        self._count()  # fails on a handle that is not a console (redirected stdin)

    def _count(self) -> int:
        count = self._wintypes.DWORD()
        if not self._kernel32.GetNumberOfConsoleInputEvents(self._handle, self._ctypes.byref(count)):
            raise OSError(self._ctypes.get_last_error(), "GetNumberOfConsoleInputEvents failed")
        return count.value

    def _peek(self, limit: int) -> list[ConsoleEvent]:
        ctypes = self._ctypes
        buffer = (self._record * limit)()
        got = self._wintypes.DWORD()
        if not self._kernel32.PeekConsoleInputW(self._handle, buffer, limit, ctypes.byref(got)):
            raise OSError(ctypes.get_last_error(), "PeekConsoleInputW failed")
        events = []
        for record in buffer[: got.value]:
            if record.EventType == KEY_EVENT:
                key = record.Event.KeyEvent
                events.append(ConsoleEvent(KEY_EVENT, bool(key.bKeyDown), key.wVirtualKeyCode, key.uChar or "\0"))
            else:
                events.append(ConsoleEvent(record.EventType))
        return events

    def _take(self, count: int) -> None:
        ctypes = self._ctypes
        buffer = (self._record * count)()
        got = self._wintypes.DWORD()
        # These records were just peeked: they are waiting, so this returns at once.
        if not self._kernel32.ReadConsoleInputW(self._handle, buffer, count, ctypes.byref(got)):
            raise OSError(ctypes.get_last_error(), "ReadConsoleInputW failed")

    def _run(self, on_key: Callable[[str], None], stop: threading.Event) -> None:
        while not stop.is_set():
            waiting = self._count()
            if waiting:
                keys, taken = plan_console_events(self._peek(min(waiting, _PEEK_LIMIT)))
                if taken and not stop.is_set():
                    self._take(taken)
                    self._emit(on_key, keys)
                    continue  # more may be waiting behind them
            stop.wait(POLL_SECONDS)


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


def _in_foreground(fd: int) -> bool:
    """Whether this process group owns the terminal ``fd``; False when that cannot be told."""

    try:
        return os.tcgetpgrp(fd) == os.getpgrp()
    except (OSError, AttributeError):
        return False


def default_key_listener() -> KeyListener:
    """The listener for this platform; a no-op one without a keyboard to read."""

    try:
        if not sys.stdin.isatty():
            return NoKeys()
        if sys.platform == "win32":
            return WindowsConsoleKeys()
        if not _in_foreground(sys.stdin.fileno()):
            return NoKeys()  # a background job (`absolut ... &`) touching the tty would be stopped
        return PosixKeys()
    except Exception:  # noqa: BLE001 - an import error or an odd stdin: the view still works
        return NoKeys()
