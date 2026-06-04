"""Helpers for benchmark code extraction and sandboxed execution."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any


_PYTHON_BLOCK_RE = re.compile(
    r"```(?:python)?[ \t]*\r?\n?(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
_DEFAULT_SANDBOX_TEMP_DIR = Path(".cache/sol_eval/code_exec")
_DEFAULT_MEMORY_LIMIT_MB = 4096
_DEFAULT_MAX_OUTPUT_BYTES = 1_000_000

_RUNNER_SOURCE = r"""
import builtins
import json
import os
import pathlib
import socket
import subprocess
import sys
import traceback

sandbox_root = pathlib.Path(os.getcwd()).resolve()
payload_path = pathlib.Path(sys.argv[1])
result_path = pathlib.Path(sys.argv[2])

_orig_open = builtins.open

def _resolve_path(path_like):
    path = pathlib.Path(path_like)
    if path.is_absolute():
        return path.resolve()
    return (pathlib.Path.cwd() / path).resolve()

def _guard_open(file, mode="r", *args, **kwargs):
    if any(flag in mode for flag in ("w", "a", "x", "+")):
        resolved = _resolve_path(file)
        if resolved != sandbox_root and sandbox_root not in resolved.parents:
            raise PermissionError(f"write outside sandbox is blocked: {resolved}")
    return _orig_open(file, mode, *args, **kwargs)

builtins.open = _guard_open

def _blocked(*args, **kwargs):
    raise PermissionError("operation blocked in sandbox")

os.system = _blocked
os.popen = _blocked
subprocess.Popen = _blocked
subprocess.call = _blocked
subprocess.run = _blocked
subprocess.check_call = _blocked
subprocess.check_output = _blocked
socket.socket = _blocked
socket.create_connection = _blocked

def _write_result(payload):
    with _orig_open(result_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

try:
    with _orig_open(payload_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    namespace = {}
    exec(payload["program"], namespace)
    fn = namespace[payload["entry_point"]]
    outputs = []
    for args in payload["inputs"]:
        if isinstance(args, list):
            outputs.append(fn(*args))
        elif isinstance(args, tuple):
            outputs.append(fn(*args))
        else:
            outputs.append(fn(args))
    _write_result({"ok": True, "outputs": outputs})
except BaseException as exc:
    _write_result(
        {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=20),
        }
    )
    sys.exit(1)
"""

_STDIO_RUNNER_SOURCE = r"""
import builtins
import io
import json
import os
import pathlib
import socket
import subprocess
import sys
import traceback

sandbox_root = pathlib.Path(os.getcwd()).resolve()
payload_path = pathlib.Path(sys.argv[1])
result_path = pathlib.Path(sys.argv[2])

_orig_open = builtins.open
_orig_stdout = sys.stdout
_orig_stderr = sys.stderr
_orig_stdin = sys.stdin

def _resolve_path(path_like):
    path = pathlib.Path(path_like)
    if path.is_absolute():
        return path.resolve()
    return (pathlib.Path.cwd() / path).resolve()

def _guard_open(file, mode="r", *args, **kwargs):
    if any(flag in mode for flag in ("w", "a", "x", "+")):
        resolved = _resolve_path(file)
        if resolved != sandbox_root and sandbox_root not in resolved.parents:
            raise PermissionError(f"write outside sandbox is blocked: {resolved}")
    return _orig_open(file, mode, *args, **kwargs)

builtins.open = _guard_open

def _blocked(*args, **kwargs):
    raise PermissionError("operation blocked in sandbox")

os.system = _blocked
os.popen = _blocked
subprocess.Popen = _blocked
subprocess.call = _blocked
subprocess.run = _blocked
subprocess.check_call = _blocked
subprocess.check_output = _blocked
socket.socket = _blocked
socket.create_connection = _blocked

def _write_result(payload):
    with _orig_open(result_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

try:
    with _orig_open(payload_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    stdin_text = payload["stdin"]
    sys.stdin = io.StringIO(stdin_text)
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    globals_dict = {"__name__": "__main__", "__file__": "candidate.py"}
    exec(compile(payload["program"], "candidate.py", "exec"), globals_dict, globals_dict)
    _write_result(
        {
            "ok": True,
            "stdout": sys.stdout.getvalue(),
            "stderr": sys.stderr.getvalue(),
        }
    )
except BaseException as exc:
    _write_result(
        {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "stdout": getattr(sys.stdout, "getvalue", lambda: "")(),
            "stderr": getattr(sys.stderr, "getvalue", lambda: "")(),
            "traceback": traceback.format_exc(limit=20),
        }
    )
    sys.exit(1)
finally:
    sys.stdout = _orig_stdout
    sys.stderr = _orig_stderr
    sys.stdin = _orig_stdin
"""


def extract_python_code(response_text: str | None) -> str | None:
    """Best-effort extraction of Python code from a model response."""

    if response_text is None:
        return None

    text = str(response_text).strip()
    if not text:
        return None

    matches = _PYTHON_BLOCK_RE.findall(text)
    if matches:
        candidate = matches[-1].strip("\n")
    else:
        candidate = text

    if candidate.startswith("<answer>") and candidate.endswith("</answer>"):
        candidate = candidate[len("<answer>") : -len("</answer>")].strip()

    return candidate or None


def compare_outputs(expected: Any, actual: Any, *, atol: float = 0.0) -> bool:
    """Recursively compare execution outputs with numeric tolerance."""

    if type(expected) in (int, float) and type(actual) in (int, float):
        try:
            return math.isclose(float(expected), float(actual), abs_tol=float(atol))
        except Exception:
            return False

    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) != len(actual):
            return False
        return all(compare_outputs(e, a, atol=atol) for e, a in zip(expected, actual))

    if isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            return False
        return all(compare_outputs(expected[k], actual[k], atol=atol) for k in expected)

    return expected == actual


def _sandbox_preexec_fn(*, memory_limit_mb: int, cpu_limit_sec: int) -> None:
    import resource

    mem_bytes = int(memory_limit_mb) * 1024 * 1024
    file_bytes = _DEFAULT_MAX_OUTPUT_BYTES

    os.setsid()
    os.umask(0o077)

    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit_sec, cpu_limit_sec))
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    resource.setrlimit(resource.RLIMIT_FSIZE, (file_bytes, file_bytes))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    if hasattr(resource, "RLIMIT_NPROC"):
        resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))


def run_callable_program(
    program: str,
    *,
    entry_point: str,
    inputs: list[Any],
    timeout_sec: float,
    sandbox_temp_dir: str | os.PathLike[str] | None = None,
    memory_limit_mb: int = _DEFAULT_MEMORY_LIMIT_MB,
    preserve_exec_artifacts: bool = False,
) -> tuple[bool, Any]:
    """Execute a callable-style Python program in a constrained subprocess."""

    base_temp_dir = Path(sandbox_temp_dir or _DEFAULT_SANDBOX_TEMP_DIR)
    base_temp_dir.mkdir(parents=True, exist_ok=True)
    sandbox_dir = Path(
        tempfile.mkdtemp(prefix="sol_eval_exec_", dir=str(base_temp_dir))
    )

    payload = {
        "program": program,
        "entry_point": entry_point,
        "inputs": inputs,
    }
    payload_path = sandbox_dir / "payload.json"
    runner_path = sandbox_dir / "runner.py"
    result_path = sandbox_dir / "result.json"

    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    runner_path.write_text(_RUNNER_SOURCE, encoding="utf-8")

    cpu_limit_sec = max(1, int(math.ceil(float(timeout_sec))))
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }

    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                str(runner_path),
                str(payload_path),
                str(result_path),
            ],
            cwd=str(sandbox_dir),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=float(timeout_sec),
            preexec_fn=lambda: _sandbox_preexec_fn(
                memory_limit_mb=int(memory_limit_mb),
                cpu_limit_sec=cpu_limit_sec,
            ),
            check=False,
        )
    except subprocess.TimeoutExpired:
        if not preserve_exec_artifacts:
            shutil.rmtree(sandbox_dir, ignore_errors=True)
        return False, "timed out"

    if not result_path.exists():
        if not preserve_exec_artifacts:
            shutil.rmtree(sandbox_dir, ignore_errors=True)
        return False, f"process exited with code {proc.returncode}"

    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as exc:
        if not preserve_exec_artifacts:
            shutil.rmtree(sandbox_dir, ignore_errors=True)
        return False, f"invalid sandbox result: {exc}"

    if not preserve_exec_artifacts:
        shutil.rmtree(sandbox_dir, ignore_errors=True)

    if not payload.get("ok"):
        return False, payload.get("error", "unknown error")
    return True, payload.get("outputs", [])


def run_stdio_program(
    program: str,
    *,
    stdin_text: str,
    timeout_sec: float,
    sandbox_temp_dir: str | os.PathLike[str] | None = None,
    memory_limit_mb: int = _DEFAULT_MEMORY_LIMIT_MB,
    preserve_exec_artifacts: bool = False,
) -> tuple[bool, dict[str, Any]]:
    """Execute a stdin/stdout style Python program in a constrained subprocess."""

    base_temp_dir = Path(sandbox_temp_dir or _DEFAULT_SANDBOX_TEMP_DIR)
    base_temp_dir.mkdir(parents=True, exist_ok=True)
    sandbox_dir = Path(
        tempfile.mkdtemp(prefix="sol_eval_stdio_", dir=str(base_temp_dir))
    )

    payload = {
        "program": program,
        "stdin": stdin_text,
    }
    payload_path = sandbox_dir / "payload.json"
    runner_path = sandbox_dir / "runner.py"
    result_path = sandbox_dir / "result.json"

    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    runner_path.write_text(_STDIO_RUNNER_SOURCE, encoding="utf-8")

    cpu_limit_sec = max(1, int(math.ceil(float(timeout_sec))))
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }

    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                str(runner_path),
                str(payload_path),
                str(result_path),
            ],
            cwd=str(sandbox_dir),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=float(timeout_sec),
            preexec_fn=lambda: _sandbox_preexec_fn(
                memory_limit_mb=int(memory_limit_mb),
                cpu_limit_sec=cpu_limit_sec,
            ),
            check=False,
        )
    except subprocess.TimeoutExpired:
        if not preserve_exec_artifacts:
            shutil.rmtree(sandbox_dir, ignore_errors=True)
        return False, {"error": "timed out", "stdout": "", "stderr": ""}

    if not result_path.exists():
        if not preserve_exec_artifacts:
            shutil.rmtree(sandbox_dir, ignore_errors=True)
        return False, {
            "error": f"process exited with code {proc.returncode}",
            "stdout": "",
            "stderr": "",
        }

    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as exc:
        if not preserve_exec_artifacts:
            shutil.rmtree(sandbox_dir, ignore_errors=True)
        return False, {"error": f"invalid sandbox result: {exc}", "stdout": "", "stderr": ""}

    if not preserve_exec_artifacts:
        shutil.rmtree(sandbox_dir, ignore_errors=True)

    if not payload.get("ok"):
        return False, {
            "error": payload.get("error", "unknown error"),
            "stdout": payload.get("stdout", ""),
            "stderr": payload.get("stderr", ""),
        }
    return True, {
        "stdout": payload.get("stdout", ""),
        "stderr": payload.get("stderr", ""),
    }


def build_functional_program(
    *,
    prompt: str,
    code: str,
    contract: str | None = None,
) -> str:
    """Assemble a function-completion program for execution."""

    def _normalize_body_block(block: str) -> str:
        normalized = textwrap.dedent(block).strip("\n")
        if not normalized:
            return ""
        return textwrap.indent(normalized, "    ")

    pieces = [prompt.rstrip(), ""]
    if contract:
        contract_block = _normalize_body_block(contract)
        if contract_block:
            pieces.extend([contract_block, ""])
    pieces.append(_normalize_body_block(code))
    pieces.append("")
    return "\n".join(pieces)
