"""Autonomous coding agent built with Google ADK and Vertex AI (Gemini).

Provides sandboxed filesystem exploration, regex code search, file I/O, and
subprocess code execution tools alongside a structured system prompt for
root-cause analysis, boundary-condition rigor, and iterative self-verification.
"""

import os
import re
import subprocess
import sys
import warnings

from google.adk.agents import Agent
from google.genai import types

# Default model shared with eval.py as a single source of truth.
DEFAULT_MODEL = os.environ.get("AGENT_MODEL", "gemini-3.1-flash-lite")

# Silence ADK function-declaration schema warnings to keep CLI output clean.
warnings.filterwarnings(
    "ignore", message=r".*JSON_SCHEMA_FOR_FUNC_DECL.*", category=UserWarning
)


def _resolve(sandbox: str, path: str) -> str:
    """Resolve a sandbox-relative path to an absolute path, blocking path traversal."""
    sandbox = os.path.realpath(sandbox)
    full = os.path.realpath(os.path.join(sandbox, path))
    if full != sandbox and not full.startswith(sandbox + os.sep):
        raise ValueError(f"path escapes sandbox: {path!r}")
    return full


def _read(sandbox: str, path: str) -> str:
    """Read a UTF-8 text file inside the sandbox, returning errors as strings."""
    try:
        with open(_resolve(sandbox, path), encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return f"Error: no such file: {path}"
    except (ValueError, OSError) as exc:
        return f"Error: {exc}"


def _write(sandbox: str, path: str, contents: str) -> str:
    """Write a UTF-8 text file inside the sandbox, creating parent directories."""
    try:
        full = _resolve(sandbox, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as fh:
            fh.write(contents)
    except (ValueError, OSError) as exc:
        return f"Error: {exc}"
    return f"Wrote {len(contents)} chars to {path}"


def _list_files(sandbox: str, directory: str = ".") -> str:
    """Recursively list all non-bytecode files under a sandbox directory."""
    try:
        root = _resolve(sandbox, directory)
        if not os.path.exists(root):
            return f"Error: no such directory: {directory}"
        matches = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(
                d for d in dirnames if d not in {"__pycache__", ".git"}
            )
            for fname in sorted(filenames):
                if fname.endswith(".pyc"):
                    continue
                rel = os.path.relpath(os.path.join(dirpath, fname), sandbox)
                matches.append(rel)
        return "\n".join(matches) if matches else "(empty directory)"
    except (ValueError, OSError) as exc:
        return f"Error: {exc}"


def _search_code(sandbox: str, pattern: str, directory: str = ".") -> str:
    """Search files in the sandbox for lines matching a regular expression."""
    try:
        root = _resolve(sandbox, directory)
        regex = re.compile(pattern)
        hits = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(
                d for d in dirnames if d not in {"__pycache__", ".git"}
            )
            for fname in sorted(filenames):
                if fname.endswith(".pyc"):
                    continue
                fpath = os.path.join(dirpath, fname)
                rel = os.path.relpath(fpath, sandbox)
                try:
                    with open(fpath, encoding="utf-8") as fh:
                        for lineno, line in enumerate(fh, 1):
                            if regex.search(line):
                                hits.append(f"{rel}:{lineno}: {line.rstrip()}")
                except OSError:
                    continue
        return "\n".join(hits) if hits else "No matches found."
    except (ValueError, OSError, re.error) as exc:
        return f"Error: {exc}"


def _run_command(sandbox: str, command: str, stdin_data: str = "") -> dict:
    """Execute a shell or Python command inside the sandbox with optional stdin."""
    try:
        sandbox_real = os.path.realpath(sandbox)
        src_dir = os.path.join(sandbox_real, "src")
        env = {
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": f"{src_dir}:{sandbox_real}:"
            + os.environ.get("PYTHONPATH", ""),
        }
        proc = subprocess.run(
            command,
            shell=True,
            cwd=sandbox_real,
            input=stdin_data,
            capture_output=True,
            text=True,
            timeout=15,
            env=env,
        )
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except subprocess.TimeoutExpired:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": "Error: command timed out after 15s",
        }
    except Exception as exc:
        return {"returncode": -1, "stdout": "", "stderr": f"Error: {exc}"}


SYSTEM_INSTRUCTION = """You are an autonomous software engineering agent working inside a project sandbox.
Always write your code changes using `write_file` and verify them using `run_command` before finishing.

## 1. Root-Cause Investigation (Bug-Fix Tasks)
- **Explore the whole package first**: Start by calling `list_files` (and `search_code` if needed) to see every file in `src/`.
- **Read beyond the file named in the prompt**: Bug reports often name the downstream function where the wrong result surfaces, while the real bug lives upstream in a sibling helper (such as a parsing, loading, or normalization module). Read `__init__.py` and ALL sibling files in the package before editing anything.
- **Trace data flow upstream**: Follow how raw inputs (strings, amounts, timestamps with timezone offsets) are parsed and transformed before reaching the reporting function. Fix the root cause where data is lost or corrupted (e.g., integer truncation during parsing or stripping `tzinfo`), rather than patching around the symptom downstream.
- **Preserve existing return types and API contracts (`float` vs `Decimal`)**: In bug-fix tasks where existing library functions use `float` and `round(..., 2)` (such as `int(float(raw))` in a loader and `round(sum(amounts), 2)` in a reporter), keep `float` as the return type (e.g., `return float(raw)`) and do NOT change the functions to return `Decimal`. In Python, `Decimal("125.49") == 125.49` evaluates to `False`, so returning `Decimal` when tests assert equality against a `float` literal will fail! In your reproduction test for bug-fix tasks, verify `isinstance(result, float)` when the original function returned a `float`.

## 2. Systematic Edge-Case & Format Rigor (Build & CLI Tasks)
Before writing code, enumerate boundary conditions and enforce exact output contracts:
- **Sequence & range direction**: When expanding a range `a-b`, support both ascending (`a <= b`, step `+1`) and descending/reversed ranges (`a > b`, e.g., `9-7` must expand descending as `9 8 7` with step `-1`), as well as single-number ranges (`a == b`) and negative integers. Note that splitting `a-b` when numbers can be negative requires care, or `int(start_str)` / `int(end_str)` stepping by `1 if start <= end else -1`.
- **Exact decimal math & half-up rounding for CLI output**: In CLI programs that print formatted numbers or money to stdout, avoid binary `float` rounding traps by using `decimal.Decimal` initialized from strings and `ROUND_HALF_UP` (`val.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)` so `5.005` prints as `5.01`).
- **Safe aggregation on empty input**: In Python, `sum(generator)` on an empty sequence returns integer `0` (`int`), which crashes with `AttributeError` if you call `.quantize()`. Always seed `sum(..., Decimal("0"))` so empty inputs remain `Decimal("0")` and format cleanly as `0.00`. Also guard `average` on empty input to print `0.00` instead of dividing by zero.
- **Trailing-zero trimming**: To format a `Decimal` with trailing zeros removed (e.g., `4.0` -> `4`, `1.50` -> `1.5`), convert with fixed-point `s = f"{d:f}"`, strip trailing zeros and decimal point (`s = s.rstrip("0").rstrip(".") if "." in s else s`), and normalize `"-0"` to `"0"`.
- **Exact exit codes & error streams**: Write all error messages to `sys.stderr` (never `stdout`) and exit with the exact required code (`sys.exit(2)` for usage errors, missing/unknown commands, malformed lines, or `start > end`; `sys.exit(1)` when a queried item/category is not found).

## 3. Mandatory Verification Loop
- **Reproduce first**: In bug-fix tasks, use `run_command` to run a small reproduction script exercising the full pipeline (from upstream parse/load functions through the reporting function) to observe the failure before editing.
- **Test examples and edge cases**: After writing or editing files with `write_file`, use `run_command` (passing `stdin_data` or shell pipes) to test:
  1. All examples from the prompt.
  2. Empty input (`stdin_data=""` or blank lines).
  3. Reversed/descending ranges (e.g., `9-7`), touching intervals, and half-up rounding ties (e.g., `5.005`).
  4. Invalid inputs, unknown commands, and missing arguments to verify `stderr` output and exact nonzero exit codes (`1` vs `2`).
- **Iterate until green**: If any test fails, fix the code with `write_file` and re-run `run_command` until every check passes. Clean up any temporary test scripts you created outside `src/` before finishing.
"""


def build_agent(sandbox: str, model: str) -> Agent:
    """Build and configure the ADK coding agent scoped to `sandbox`."""

    def list_files(directory: str = ".") -> dict:
        """List all files recursively under `directory` (relative to the project root)."""
        return {"result": _list_files(sandbox, directory)}

    def search_code(pattern: str, directory: str = ".") -> dict:
        """Search files under `directory` for lines matching regex `pattern`."""
        return {"result": _search_code(sandbox, pattern, directory)}

    def read_file(path: str) -> dict:
        """Read a text file (path is relative to the project root) and return its contents."""
        return {"result": _read(sandbox, path)}

    def write_file(path: str, contents: str) -> dict:
        """Create or overwrite a text file (path is relative to the project root)."""
        return {"result": _write(sandbox, path, contents)}

    def run_command(command: str, stdin_data: str = "") -> dict:
        """Run a shell or python command inside the project root with optional `stdin_data`, returning returncode, stdout, and stderr."""
        return {"result": _run_command(sandbox, command, stdin_data)}

    return Agent(
        name="coding_agent",
        model=model,
        instruction=SYSTEM_INSTRUCTION,
        tools=[list_files, search_code, read_file, write_file, run_command],
        generate_content_config=types.GenerateContentConfig(
            temperature=0.0,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.LOW
            ),
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=5)
            ),
        ),
    )


if __name__ == "__main__":
    import asyncio
    import glob
    from google.adk.runners import InMemoryRunner

    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")
    os.environ.setdefault(
        "GOOGLE_CLOUD_PROJECT", os.environ.get("GCP_PROJECT_ID", "your-gcp-project-id")
    )
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")

    sandbox = os.path.join(os.path.dirname(os.path.realpath(__file__)), "out")
    os.makedirs(os.path.join(sandbox, "src"), exist_ok=True)
    prompt = " ".join(sys.argv[1:]) or "Write a hello world python script"

    async def _run_agent() -> None:
        runner = InMemoryRunner(agent=build_agent(sandbox, DEFAULT_MODEL))
        session = await runner.session_service.create_session(
            app_name=runner.app_name, user_id="developer"
        )
        message = types.Content(role="user", parts=[types.Part(text=prompt)])
        async for event in runner.run_async(
            user_id="developer", session_id=session.id, new_message=message
        ):
            for part in (event.content and event.content.parts) or []:
                print(part.text or "", end="", flush=True)

    asyncio.run(_run_agent())
    for script in glob.glob(f"{sandbox}/**/*.py", recursive=True):
        proc = subprocess.run(
            [sys.executable, script],
            cwd=sandbox,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(
            f"\n=== output of {os.path.relpath(script, sandbox)} ===\n{proc.stdout}",
            end="",
        )
