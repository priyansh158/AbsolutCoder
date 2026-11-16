"""The absolut console script prints help and exits 0."""

import os
import subprocess
import sys
from pathlib import Path


def test_absolut_help_exits_zero():
    scripts = Path(sys.executable).parent
    executable = scripts / "absolut.exe"
    if not executable.exists():
        executable = scripts / "absolut"
    env = os.environ.copy()
    env["PATH"] = str(scripts) + os.pathsep + env.get("PATH", "")
    env.pop("OPENAI_API_KEY", None)
    completed = subprocess.run(
        [str(executable), "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0
    assert "absolut" in completed.stdout.lower()
