"""Reading results/agents/*.jsonl."""
import glob
import json
import os


def expand(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "*.jsonl")))
        else:
            files.append(p)
    return files


def load_records(paths):
    records = []
    for path in expand(paths):
        with open(path, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except ValueError as e:
                    raise ValueError(f"{path}:{n}: bad JSON line: {e}") from None
                rec["_source"] = path
                records.append(rec)
    return records
