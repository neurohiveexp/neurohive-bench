"""Clean repository copies for agent runs.

Every run gets a fresh clone that holds history only up to the task commit:
the fix commit of a bugfix task and later commits are not reachable, and the
clone has no remote, so an agent cannot peek at the answer with `git log --all`.
"""
import hashlib
import os
import re
import shutil
import subprocess

# Kept out of the diff: virtualenvs, caches and the bench's own files.
EXCLUDES = [".venv/", "venv/", "__pycache__/", "*.pyc", ".pytest_cache/", ".mypy_cache/",
            ".ruff_cache/", "node_modules/", ".nh-bench/"]
BENCH_EMAIL = "bench@neurohive.local"


class WorkspaceError(Exception):
    pass


def git(*args, cwd=None, check=True, input=None):
    p = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, input=input,
                       env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C"})
    if check and p.returncode != 0:
        raise WorkspaceError(f"git {' '.join(args)} failed in {cwd or os.getcwd()}:\n{p.stderr.strip()}")
    return p


def cache_path(cache_dir, repo):
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", repo.rstrip("/").split("/")[-1].removesuffix(".git"))
    digest = hashlib.sha1(repo.encode()).hexdigest()[:8]
    return os.path.join(cache_dir, f"{name}-{digest}.git")


def ensure_cache(repo, cache_dir):
    """Bare mirror of the repository, cloned once and reused by all runs."""
    path = cache_path(cache_dir, repo)
    if not os.path.isdir(path):
        os.makedirs(cache_dir, exist_ok=True)
        tmp = path + ".tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        git("clone", "--quiet", "--bare", repo, tmp)
        os.replace(tmp, path)
    return path


def resolve(cache, rev, repo=None):
    """Full SHA for `rev` (a SHA, `<sha>^`, a tag); fetches once if it is missing."""
    p = git("rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}", cwd=cache, check=False)
    if p.returncode != 0:
        git("fetch", "--quiet", "--tags", "origin", "+refs/heads/*:refs/heads/*", cwd=cache, check=False)
        p = git("rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}", cwd=cache, check=False)
    if p.returncode != 0:
        raise WorkspaceError(f"commit {rev!r} not found in {repo or cache}")
    return p.stdout.strip()


def _configure(dest):
    git("config", "user.name", "nh-bench", cwd=dest)
    git("config", "user.email", BENCH_EMAIL, cwd=dest)
    git("config", "commit.gpgsign", "false", cwd=dest)
    exclude = os.path.join(dest, ".git", "info", "exclude")
    os.makedirs(os.path.dirname(exclude), exist_ok=True)
    with open(exclude, "a", encoding="utf-8") as f:
        f.write("\n# neurohive agent bench\n" + "\n".join(EXCLUDES) + "\n")


def prepare(task, dest, cache_dir):
    """Create a clean copy for `task` in `dest` (removed first). Returns the base commit SHA."""
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    if task.type == "greenfield":
        os.makedirs(dest)
        git("init", "--quiet", cwd=dest)
        git("symbolic-ref", "HEAD", "refs/heads/main", cwd=dest)
        _configure(dest)
        git("commit", "--quiet", "--allow-empty", "-m", "empty repository", cwd=dest)
    else:
        cache = ensure_cache(task.repo, cache_dir)
        sha = resolve(cache, task.commit, task.repo)
        branch = f"nh-base-{sha[:12]}"
        git("branch", "--force", branch, sha, cwd=cache)
        url = "file://" + os.path.abspath(cache)
        git("clone", "--quiet", "--single-branch", "--branch", branch, url, dest)
        git("remote", "remove", "origin", cwd=dest)
        git("branch", "--quiet", "-m", branch, "main", cwd=dest)
        _configure(dest)
    base = git("rev-parse", "HEAD", cwd=dest).stdout.strip()
    # a fresh clone is clean, but make sure nothing from a previous run survived
    git("clean", "-fdxq", cwd=dest)
    return base


def commit_all(dest, message):
    git("add", "--all", cwd=dest)
    git("commit", "--quiet", "--allow-empty", "--no-verify", "-m", message, cwd=dest)
    return git("rev-parse", "HEAD", cwd=dest).stdout.strip()


def snapshot_diff(dest, base):
    """Everything the agent changed relative to `base`: commits, staged, unstaged and new files."""
    git("add", "--all", cwd=dest)
    diff = git("-c", "core.quotepath=false", "diff", "--cached", "--no-color", "--no-ext-diff",
               base, cwd=dest).stdout
    numstat = git("diff", "--cached", "--numstat", base, cwd=dest).stdout
    files = added = removed = 0
    for line in numstat.splitlines():
        a, r, _ = line.split("\t", 2)
        files += 1
        added += int(a) if a.isdigit() else 0
        removed += int(r) if r.isdigit() else 0
    return diff, {"files": files, "added": added, "removed": removed}


def install_hidden_tests(task, dest, cache_dir):
    """Copy the hidden tests into the workspace after the agent has finished."""
    installed = []
    for h in task.hidden_tests:
        target = os.path.join(dest, h["path"])
        os.makedirs(os.path.dirname(target) or dest, exist_ok=True)
        if "file" in h:
            shutil.copyfile(h["file"], target)
        else:
            cache = ensure_cache(task.repo, cache_dir)
            sha = resolve(cache, h["from_commit"], task.repo)
            blob = subprocess.run(["git", "show", f"{sha}:{h['path']}"], cwd=cache, capture_output=True)
            if blob.returncode != 0:
                raise WorkspaceError(f"{h['path']} not found in {sha}: {blob.stderr.decode().strip()}")
            with open(target, "wb") as f:
                f.write(blob.stdout)
        installed.append(h["path"])
    return installed
