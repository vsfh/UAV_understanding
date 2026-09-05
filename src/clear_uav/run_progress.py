"""Console progress for slow startup and child experiment processes."""
from contextlib import contextmanager
import os
import subprocess
import time


def elapsed(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


@contextmanager
def phase(label):
    start = time.monotonic()
    print(f"[stage] {label}", flush=True)
    try:
        yield
    except BaseException:
        print(f"[stopped] {label} after {elapsed(time.monotonic() - start)}", flush=True)
        raise
    else:
        print(f"[done] {label} ({elapsed(time.monotonic() - start)})", flush=True)


def run_with_progress(command, *, cwd, env, label, index, total, interval=15):
    """Heartbeat lives in the parent even if native loading holds the child's GIL.

    Elapsed time is NOT a percentage or proof that a child is making progress.
    Keep inherited stdout/stderr so the child's real tqdm bars remain live.
    """
    child_env = dict(os.environ if env is None else env)
    child_env["PYTHONUNBUFFERED"] = "1"
    start = time.monotonic()
    print(f"\n[task {index}/{total}] {label}", flush=True)
    print("[stage] Starting Python and importing dependencies...", flush=True)
    process = subprocess.Popen(command, cwd=cwd, env=child_env)
    try:
        while True:
            try:
                code = process.wait(timeout=interval)
                break
            except subprocess.TimeoutExpired:
                print(f"[waiting] {label} | elapsed {elapsed(time.monotonic()-start)}"
                      f" | process {process.pid} has not exited; see last stage above",
                      flush=True)
    except KeyboardInterrupt:
        # The terminal normally delivers Ctrl+C to both parent and child.
        # If the child does not exit, terminate only this runner's child.
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        print(f"[interrupted] {label}; existing checkpoints kept.", flush=True)
        raise SystemExit(130) from None
    if code:
        print(f"[failed] {label}: exit {code}", flush=True)
        raise subprocess.CalledProcessError(code, command)
    print(f"[task {index}/{total} done] {label}"
          f" | elapsed {elapsed(time.monotonic()-start)}", flush=True)
