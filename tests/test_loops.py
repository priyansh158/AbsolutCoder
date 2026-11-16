"""The check loop and the interval loop, driven by fakes."""

import pytest
from deepagents.backends.protocol import ExecuteResponse

from absolutcoder.loops import (
    VerifyOutcome,
    failure_prompt,
    parse_interval,
    run_interval_loop,
    run_until_verified,
)


def test_parse_interval():
    assert parse_interval("30s") == 30
    assert parse_interval("5m") == 300
    assert parse_interval("2H") == 7200
    assert parse_interval(" 1s ") == 1
    for bad in ("0s", "5", "m", "-1s", "1.5m", "5 minutes"):
        with pytest.raises(ValueError):
            parse_interval(bad)


def _checks(*responses):
    queue = list(responses)
    calls = []

    def verify():
        calls.append(1)
        return queue.pop(0)

    return verify, calls


def test_verify_passes_first_round():
    turns = []
    verify, _ = _checks(ExecuteResponse(output="1 passed", exit_code=0))
    outcome = run_until_verified(lambda t: turns.append(t) or "state", verify, "fix it", command="pytest")
    assert turns == ["fix it"]
    assert outcome == VerifyOutcome(True, 1, "1 passed")


def test_failure_output_becomes_next_turn():
    turns = []
    output = "E assert 1 == 2"
    verify, _ = _checks(
        ExecuteResponse(output=output, exit_code=1),
        ExecuteResponse(output="ok", exit_code=0),
    )
    outcome = run_until_verified(lambda t: turns.append(t) or "state", verify, "fix it", command="pytest -q")
    assert len(turns) == 2
    assert turns[1] == failure_prompt("pytest -q", 1, output)
    assert "E assert 1 == 2" in turns[1]
    assert "pytest -q" in turns[1]
    assert outcome.passed is True
    assert outcome.rounds == 2


def test_stops_at_max_rounds():
    turns = []

    def verify():
        return ExecuteResponse(output="still broken", exit_code=1)

    outcome = run_until_verified(lambda t: turns.append(t) or "state", verify, "go", max_rounds=3)
    assert len(turns) == 3
    assert outcome == VerifyOutcome(False, 3, "still broken")


def test_max_rounds_below_one_runs_once():
    turns = []
    outcome = run_until_verified(
        lambda t: turns.append(t) or "state",
        lambda: ExecuteResponse(output="no", exit_code=1),
        "go",
        max_rounds=0,
    )
    assert len(turns) == 1
    assert outcome == VerifyOutcome(False, 1, "no")


def test_failed_turn_stops_the_loop():
    verify, calls = _checks()
    outcome = run_until_verified(lambda t: None, verify, "go")
    assert outcome == VerifyOutcome(False, 1, "turn failed")
    assert calls == []


def test_failure_prompt_keeps_only_the_tail():
    output = "a" * 6000 + "b" * 4000
    prompt = failure_prompt("make test", 2, output)
    assert prompt == (
        "The check command failed (exit code 2). Find the cause, fix it, and make the check pass. "
        "Do not weaken or delete the check.\n\n"
        "Command: make test\n\n"
        "Output (last 4000 characters):\n" + "b" * 4000
    )
    assert "last 4000 characters" in prompt
    assert "a" not in prompt.split("characters):\n", 1)[1]


def test_interval_loop_runs_until_max_iterations():
    turns, sleeps, seen = [], [], []
    count = run_interval_loop(
        turns.append, "hi", 7.0, sleep=sleeps.append, max_iterations=3, on_iteration=seen.append
    )
    assert count == 3
    assert turns == ["hi", "hi", "hi"]
    assert sleeps == [7.0, 7.0]
    assert seen == [1, 2, 3]


def test_interval_loop_stops_on_keyboard_interrupt():
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 2:
            raise KeyboardInterrupt

    turns = []
    assert run_interval_loop(turns.append, "hi", 1.0, sleep=sleep) == 2
    assert len(turns) == 2

    def interrupted_turn(_prompt):
        raise KeyboardInterrupt

    assert run_interval_loop(interrupted_turn, "hi", 1.0, sleep=sleeps.append) == 1
