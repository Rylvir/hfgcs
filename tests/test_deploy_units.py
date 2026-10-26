"""deploy/: the systemd units must not hardcode the interpreter, and the wrapper must
honour HFGCS_PYTHON."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "deploy" / "bin" / "hfgcs-python"
UNITS = sorted((ROOT / "deploy" / "systemd").glob("*.service"))


@pytest.mark.parametrize("unit", UNITS, ids=lambda p: p.name)
def test_units_use_overridable_wrapper(unit):
    lines = [ln.strip() for ln in unit.read_text().splitlines()]
    execs = [ln for ln in lines if ln.startswith("ExecStart=")]
    assert len(execs) == 1 and execs[0].startswith("ExecStart=%h/hfgcs-app/deploy/bin/hfgcs-python -m ")
    assert ".venv" not in execs[0]
    env = lines.index("Environment=HFGCS_PYTHON=%h/hfgcs-app/.venv/bin/python")
    files = [i for i, ln in enumerate(lines) if ln.startswith("EnvironmentFile=")]
    assert files and all(i > env for i in files)   # the env file's HFGCS_PYTHON wins over the default


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell wrapper")
def test_wrapper_runs_hfgcs_python(tmp_path):
    assert os.access(WRAPPER, os.X_OK)
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path), "HFGCS_PYTHON": sys.executable}
    out = subprocess.run([str(WRAPPER), "-c", "import sys; print(sys.executable)"], env=env,
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == sys.executable
    env.pop("HFGCS_PYTHON")                         # default: $HOME/hfgcs-app/.venv/bin/python
    r = subprocess.run([str(WRAPPER), "-c", "pass"], env=env, capture_output=True, text=True)
    assert r.returncode == 127 and f"{tmp_path}/hfgcs-app/.venv/bin/python" in r.stderr
    venv_py = tmp_path / "hfgcs-app" / ".venv" / "bin" / "python"
    venv_py.parent.mkdir(parents=True)
    venv_py.symlink_to(sys.executable)
    r = subprocess.run([str(WRAPPER), "-c", "print('ok')"], env=env, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "ok"
