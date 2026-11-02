"""A stand-in for an agent CLI. Behaviour is set by FAKE_MODE:

solve   - copy files from FAKE_SOLUTION (a directory) into the working directory, commit one of them
fail    - change nothing
crash   - write a file, exit with code 3
sleep   - sleep FAKE_SLEEP seconds (for timeouts)
leak    - write a CLAUDE.md and mention its vendor in code (for blind export)

Prints a usage report in FAKE_FORMAT: claude (one JSON object), codex or opencode (JSONL), text.
"""
import json
import os
import shutil
import subprocess
import sys
import time

prompt = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read()
mode = os.environ.get("FAKE_MODE", "solve")
fmt = os.environ.get("FAKE_FORMAT", "claude")
if os.environ.get("FAKE_TRACE"):
    with open(os.environ["FAKE_TRACE"], "w", encoding="utf-8") as f:
        json.dump({"prompt": prompt, "cwd": os.getcwd(), "args": sys.argv[2:],
                   "log": subprocess.run(["git", "log", "--all", "--format=%s"],
                                         capture_output=True, text=True).stdout}, f)

if mode in ("solve", "leak"):
    src = os.environ["FAKE_SOLUTION"]
    for name in os.listdir(src):
        shutil.copyfile(os.path.join(src, name), name)
    if mode == "leak":
        with open("CLAUDE.md", "w") as f:
            f.write("agent notes\n")
        with open("calc.py", "a") as f:
            f.write("# written by Claude Code (Anthropic), not Codex\n")
    if os.environ.get("FAKE_COMMIT"):
        subprocess.run(["git", "add", "-A"], check=True)
        subprocess.run(["git", "commit", "-qm", "agent commit"], check=True)
elif mode == "crash":
    with open("half_done.txt", "w") as f:
        f.write("oops\n")
    print("boom", file=sys.stderr)
    sys.exit(3)
elif mode == "sleep":
    time.sleep(float(os.environ.get("FAKE_SLEEP", "30")))

if fmt == "claude":
    print(json.dumps({"type": "result", "subtype": "success", "num_turns": 4, "total_cost_usd": 0.25,
                      "usage": {"input_tokens": 1000, "cache_read_input_tokens": 8000,
                                "cache_creation_input_tokens": 1000, "output_tokens": 500}}))
elif fmt == "codex":
    print(json.dumps({"type": "thread.started", "thread_id": "t"}))
    for _ in range(2):
        print(json.dumps({"type": "turn.completed",
                          "usage": {"input_tokens": 3000, "cached_input_tokens": 1000, "output_tokens": 200}}))
elif fmt == "opencode":
    for _ in range(2):
        print(json.dumps({"type": "step_finish", "part": {"cost": 0.01, "tokens": {
            "input": 100, "output": 50, "reasoning": 10, "cache": {"read": 900, "write": 0}}}}))
else:
    print("done, no usage info")
