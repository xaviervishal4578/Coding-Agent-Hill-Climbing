"""Evaluation harness for the autonomous coding agent.

Discovers benchmark tasks under `tasks/`, provisions isolated temporary sandboxes,
runs `build_agent` asynchronously across all tasks in parallel, records full ADK
trajectories (`trajectory.evalset.json`, `tool_uses.json`, `repro.sh`), and grades
shipped solutions against hidden test suites.
"""

import argparse
import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from typing import NamedTuple

# Configure Vertex AI environment defaults prior to importing ADK / GenAI SDKs.
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")
os.environ.setdefault(
    "GOOGLE_CLOUD_PROJECT", os.environ.get("GCP_PROJECT_ID", "your-gcp-project-id")
)
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")

from google.adk.agents.run_config import RunConfig  # noqa: E402
from google.adk.evaluation.eval_case import (  # noqa: E402
    EvalCase,
    SessionInput,
    get_all_tool_calls,
    get_all_tool_responses,
)
from google.adk.evaluation.eval_set import EvalSet  # noqa: E402
from google.adk.evaluation.evaluation_generator import EvaluationGenerator  # noqa: E402
from google.adk.runners import InMemoryRunner  # noqa: E402
from google.genai import types  # noqa: E402

from agent import DEFAULT_MODEL, build_agent  # noqa: E402

HERE = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "src" else HERE
TASKS_DIR = os.path.join(REPO_ROOT, "tasks")
RUNS_DIR = os.path.join(REPO_ROOT, "runs")
USER_ID = "evaluator"


def silence_no_cap_warning() -> None:
    """Silence ADK's informational log when max_llm_calls=0 is used."""
    logging.getLogger("google_adk.google.adk.agents.run_config").setLevel(
        logging.ERROR
    )


class TaskResult(NamedTuple):
    passed: bool
    tool_calls: int
    seconds: float
    run_dir: str
    stop_reason: str


def discover_tasks(task_id: str) -> list[str]:
    """Return sorted IDs of all tasks, or a single matching `task_id`."""
    ids = sorted(
        d
        for d in os.listdir(TASKS_DIR)
        if os.path.isfile(os.path.join(TASKS_DIR, d, "prompt.md"))
    )
    if task_id:
        return [task_id] if task_id in ids else []
    return ids


def make_sandbox(task_dir: str, sandboxes_dir: str) -> str:
    """Create an isolated temporary sandbox seeded with starter code if present."""
    sandbox = tempfile.mkdtemp(prefix="task_", dir=sandboxes_dir)
    src = os.path.join(task_dir, "src")
    if os.path.isdir(src):
        shutil.copytree(src, os.path.join(sandbox, "src"))
    else:
        os.makedirs(os.path.join(sandbox, "src"))
    return sandbox


def grade(task_dir: str, sandbox: str) -> bool:
    """Copy the hidden test suite into the sandbox and execute it (exit 0 == PASS)."""
    dest = os.path.join(sandbox, "tests")
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(os.path.join(task_dir, "tests"), dest)
    try:
        proc = subprocess.run(
            [sys.executable, "tests/hidden.py"],
            cwd=sandbox,
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        return proc.returncode == 0
    except subprocess.SubprocessError:
        return False


_REPRO_TEMPLATE = """#!/usr/bin/env bash
# Reproduce the hidden-test result for task '{task_id}' against shipped ./src.
set -u
here="$(cd "$(dirname "$0")" && pwd)"
work="$(mktemp -d)"
mkdir -p "$work/src" "$work/tests"
cp -r "$here/src/." "$work/src/" 2>/dev/null || true
cp -r "{tests_dir}/." "$work/tests/"
( cd "$work" && {python} tests/hidden.py ); code=$?
echo "hidden.py exit code: $code (0 = PASS)"
rm -rf "$work"
exit $code
"""


def write_trajectory(
    task_id: str,
    task_dir: str,
    sandbox: str,
    app_name: str,
    events: list,
    runs_root: str,
) -> str:
    """Persist ADK trajectory, flattened tool uses, shipped code, and repro script."""
    invocations = EvaluationGenerator.convert_events_to_eval_invocations(events)
    eval_set = EvalSet(
        eval_set_id=task_id,
        name=f"{task_id} trajectory",
        description=(
            f"Actual trajectory the agent took on task '{task_id}'. "
            f"Graded out-of-band by tasks/{task_id}/tests/hidden.py."
        ),
        eval_cases=[
            EvalCase(
                eval_id=task_id,
                conversation=invocations,
                session_input=SessionInput(app_name=app_name, user_id=USER_ID),
            )
        ],
    )

    task_run_dir = os.path.join(runs_root, task_id)
    os.makedirs(task_run_dir, exist_ok=True)

    with open(
        os.path.join(task_run_dir, "trajectory.evalset.json"),
        "w",
        encoding="utf-8",
    ) as fh:
        fh.write(eval_set.model_dump_json(exclude_none=True, indent=2))

    flat = [
        {
            "invocation_id": inv.invocation_id,
            "tool_uses": [
                tc.model_dump(exclude_none=True)
                for tc in get_all_tool_calls(inv.intermediate_data)
            ],
            "tool_responses": [
                tr.model_dump(exclude_none=True)
                for tr in get_all_tool_responses(inv.intermediate_data)
            ],
        }
        for inv in invocations
    ]
    with open(
        os.path.join(task_run_dir, "tool_uses.json"), "w", encoding="utf-8"
    ) as fh:
        json.dump(flat, fh, indent=2, default=str)

    dest_src = os.path.join(task_run_dir, "src")
    shutil.rmtree(dest_src, ignore_errors=True)
    src = os.path.join(sandbox, "src")
    if os.path.isdir(src):
        shutil.copytree(src, dest_src)
    else:
        os.makedirs(dest_src, exist_ok=True)

    repro_path = os.path.join(task_run_dir, "repro.sh")
    with open(repro_path, "w", encoding="utf-8") as fh:
        fh.write(
            _REPRO_TEMPLATE.format(
                task_id=task_id,
                tests_dir=os.path.join(task_dir, "tests"),
                python=sys.executable,
            )
        )
    os.chmod(repro_path, 0o755)

    return task_run_dir


async def run_task(
    task_id: str, model: str, sandboxes_dir: str, runs_root: str
) -> TaskResult:
    """Run the coding agent on a single task and grade the resulting sandbox."""
    start = time.perf_counter()
    task_dir = os.path.join(TASKS_DIR, task_id)
    sandbox = make_sandbox(task_dir, sandboxes_dir)
    with open(os.path.join(task_dir, "prompt.md"), encoding="utf-8") as fh:
        prompt = fh.read().strip()

    agent = build_agent(sandbox, model)
    runner = InMemoryRunner(agent=agent)
    session = await runner.session_service.create_session(
        app_name=runner.app_name, user_id=USER_ID
    )
    msg = types.Content(role="user", parts=[types.Part(text=prompt)])

    tool_calls = 0
    stop_reason, detail = "completed", ""
    try:
        async for ev in runner.run_async(
            user_id=USER_ID,
            session_id=session.id,
            new_message=msg,
            run_config=RunConfig(max_llm_calls=0),
        ):
            tool_calls += len(ev.get_function_calls() or [])
    except Exception as exc:
        stop_reason, detail = "error", str(exc)
        print(f"warning: task {task_id!r} errored: {exc}", file=sys.stderr)

    full_session = await runner.session_service.get_session(
        app_name=runner.app_name, user_id=USER_ID, session_id=session.id
    )
    events = (
        list(full_session.events)
        if full_session and full_session.events
        else []
    )
    run_dir = write_trajectory(
        task_id, task_dir, sandbox, runner.app_name, events, runs_root
    )

    passed = await asyncio.to_thread(grade, task_dir, sandbox)
    seconds = time.perf_counter() - start

    with open(os.path.join(run_dir, "run.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "task_id": task_id,
                "model": model,
                "passed": passed,
                "tool_calls": tool_calls,
                "seconds": round(seconds, 1),
                "stop_reason": stop_reason,
                "detail": detail,
            },
            fh,
            indent=2,
        )

    return TaskResult(passed, tool_calls, seconds, run_dir, stop_reason)


def format_row(task: str, result: str, tool_calls, latency) -> str:
    return f"{task:<18}{result:<8}{tool_calls:<12}{latency}"


async def main_async(args) -> None:
    task_ids = discover_tasks(args.task)
    if not task_ids:
        print("No tasks matched.")
        return

    silence_no_cap_warning()
    runs_root = os.path.join(RUNS_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(runs_root, exist_ok=True)
    print(
        f"Please wait: agent.py is solving {len(task_ids)} tasks in parallel with {args.model}..."
    )
    sandboxes_dir = tempfile.mkdtemp(prefix="eval_base_")
    header = format_row("task_id", "result", "tool_calls", "duration_seconds")
    rule = "-" * len(header)
    print(header)
    print(rule)
    solved = 0
    completed = []
    early_stops = []
    try:
        running = [
            asyncio.create_task(
                run_task(tid, args.model, sandboxes_dir, runs_root)
            )
            for tid in task_ids
        ]
        for tid, task in zip(task_ids, running):
            r = await task
            solved += int(r.passed)
            print(
                format_row(
                    tid,
                    "PASS" if r.passed else "FAIL",
                    r.tool_calls,
                    f"{r.seconds:.1f}",
                )
            )
            completed.append((tid, r.run_dir))
            if r.stop_reason != "completed":
                early_stops.append((tid, r.stop_reason, r.run_dir))
    finally:
        shutil.rmtree(sandboxes_dir, ignore_errors=True)
    print(rule)
    print(f"Evaluation completed. Solved {solved} out of {len(task_ids)} tasks.")

    summary = {
        "solved": solved,
        "total": len(task_ids),
        "timestamp": datetime.now().isoformat(),
    }
    summary_path = os.path.join(runs_root, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    if early_stops:
        print("\nnote: some tasks stopped early (details in each task's run.json):")
        for tid, reason, run_dir in early_stops:
            print(
                f"  {tid}: {reason}  ->  {os.path.relpath(run_dir, os.getcwd())}/run.json"
            )

    if completed:
        print("Trajectories and generated code for all tasks:")
        for tid, run_dir in completed:
            rel = os.path.relpath(run_dir, os.getcwd())
            print(f"  {tid}")
            print(f"    trajectory : {rel}/trajectory.evalset.json")
            print(f"    code       : {rel}/src/")


def main() -> None:
    p = argparse.ArgumentParser(description="Run coding agent evaluation suite")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument(
        "--task", default="", help="run a single task id (default: all)"
    )
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
