"""A to-do list for generating pairs of sections from many songs, and a runner that works through it one task at a time (so it can be stopped, killed and resumed).

    batch.py plan [--force]  write batch/queue.json from out/codec_texture/musthaves.json: per song one `pool` task (cut the stems by the structure; done when the
                             song's levels file exists) and PAIRS `track` tasks. --force re-plans and keeps the state of tasks that are still in the plan.
    batch.py status          counts, one line per song, the running task, an ETA
    batch.py run             work through the queue (foreground; `make start` runs it detached under caffeinate); resumes where it left off
    batch.py stop            let the running task finish, then exit      batch.py kill   stop now: the running task goes back to pending
    batch.py retry [SONG]    failed tasks (of one song) back to pending
    batch.py do TASK_ID      one task, what the runner launches (e.g. `pool:slide`, `track:slide:3`)

Pairs     Structure.pairs: every track has at least one must-have KIND (any segment of that kind was flagged on segments.html); the other kind and the order come from
          the transition chain; the kind of a closing section that occurs once (a fade-out) is never used; kinds more than MAX_STEP_DB apart in level are not joined.
          A song's PAIRS pairs are drawn together (seeded by the song name) so that kind pairs do not repeat while others are possible; task n takes pair n.
Output    compare/Y_{song}_pair{FIRST_SEED + n}.wav/.json; logs in batch/logs/. Only the runner writes queue.json; a `running` task with no live runner is reset on start.
"""
import json
import os
import signal
import subprocess
import sys
import time
import zlib
from dataclasses import asdict, dataclass, replace
from pathlib import Path

os.environ.setdefault("TF", "8")  # the banks kept in atoms/ are 8-frame; snippets.py defaults to 4, which would rebuild every bank (and the pool tasks inherit this too)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ROOT = HERE.parent
OUT = ROOT / "out/codec_texture"
ATOMS = OUT / "atoms"
BATCH = OUT / "batch"
QUEUE, PID, STOP, LOGS = BATCH / "queue.json", BATCH / "runner.pid", BATCH / "STOP", BATCH / "logs"
MUSTHAVES = OUT / "musthaves.json"
PAIRS = 4
FIRST_SEED = 100  # tracks are Y_{song}_pair100..107: clear of the earlier pair0-3 demos


@dataclass(frozen=True)
class Task:
    song: str
    kind: str  # "pool" | "track"
    n: int = 0  # which of the song's pairs
    state: str = "pending"  # pending | running | done | failed
    started: float | None = None
    took: float | None = None
    error: str | None = None

    @property
    def id(self) -> str:
        return f"pool:{self.song}" if self.kind == "pool" else f"track:{self.song}:{self.n}"

    @property
    def output(self) -> Path:
        return ROOT / f"out/codec_texture/compare/Y_{self.song}_pair{FIRST_SEED + self.n}.json"


def load() -> tuple[Task, ...]:
    return tuple(Task(**row) for row in json.loads(QUEUE.read_text()))


def save(queue: tuple[Task, ...]) -> None:
    BATCH.mkdir(parents=True, exist_ok=True)
    tmp = QUEUE.with_suffix(".tmp")
    tmp.write_text(json.dumps([asdict(t) for t in queue], indent=1))
    os.replace(tmp, QUEUE)


def updated(queue: tuple[Task, ...], task: Task) -> tuple[Task, ...]:
    return tuple(task if t.id == task.id else t for t in queue)


def runner_pid() -> int | None:
    try:
        pid = int(PID.read_text())
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError):
        return None


def pooled(song: str) -> bool:
    return (ATOMS / f"{song}_levels.json").exists()


def tasks_of(song: str) -> tuple[Task, ...]:
    return (Task(song, "pool", state="done" if pooled(song) else "pending"), *(Task(song, "track", n) for n in range(PAIRS)))


def plan(force: bool) -> None:
    if QUEUE.exists() and not force:
        raise SystemExit(f"{QUEUE} exists: `plan --force` re-plans and keeps the state of tasks that remain")
    musthaves = json.loads(MUSTHAVES.read_text())
    songs = sorted((s for s in musthaves if (ATOMS / f"{s}_structure.json").exists()), key=lambda s: (not pooled(s), s))  # songs that are ready first
    old = {t.id: t for t in load()} if QUEUE.exists() else {}
    save(tuple(old.get(t.id, t) for song in songs for t in tasks_of(song)))
    print(f"planned {len(songs)} songs, {len(songs) * PAIRS} tracks -> {QUEUE}")


def sections_of(song: str) -> tuple:
    """The song's PAIRS pairs (the same on every call): must-have kinds from musthaves.json, the fade-out kind dropped."""
    import numpy as np
    from stitch import kinds_within
    from structure import Structure

    structure = Structure.load(ATOMS / f"{song}_structure.json")
    levels = json.loads((ATOMS / f"{song}_levels.json").read_text())
    fade = structure.fade_kind()
    must = {m["kind"] for m in json.loads(MUSTHAVES.read_text())[song]} - {fade}
    return structure.pairs(PAIRS, np.random.default_rng(zlib.crc32(song.encode())), must, kinds_within(levels), frozenset() if fade is None else frozenset({fade}))


def do(task_id: str) -> None:
    kind, song, *rest = task_id.split(":")
    if kind == "pool":
        subprocess.run([sys.executable, str(HERE / "segment.py"), song, "pool"], check=True)
        return
    import stitch

    n = int(rest[0])
    levels = json.loads((ATOMS / f"{song}_levels.json").read_text())
    stitch.make_track(song, levels, stitch.stems_in(levels), FIRST_SEED + n, list(sections_of(song)[n]))


def runnable(queue: tuple[Task, ...]) -> Task | None:
    pool_done = {t.song: t.state == "done" for t in queue if t.kind == "pool"}
    return next((t for t in queue if t.state == "pending" and (t.kind == "pool" or pool_done.get(t.song))), None)


def last_line(log: Path) -> str:
    lines = [line for line in log.read_text(errors="replace").splitlines() if line.strip()]
    return lines[-1][:300] if lines else "no output"


def run() -> None:
    if pid := runner_pid():
        raise SystemExit(f"a runner is already going (pid {pid})")
    BATCH.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    PID.write_text(str(os.getpid()))
    STOP.unlink(missing_ok=True)
    save(tuple(replace(t, state="pending") if t.state == "running" else t for t in load()))  # left over from a runner that died
    current: list[tuple[subprocess.Popen, Task]] = []

    def terminate(*_) -> None:
        for proc, task in current:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait()
            save(updated(load(), replace(task, state="pending", started=None)))
        PID.unlink(missing_ok=True)
        print("killed: the running task is back to pending", flush=True)
        os._exit(143)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    while not STOP.exists() and (task := runnable(load())):
        started = time.time()
        save(updated(load(), replace(task, state="running", started=started, error=None)))
        print(f"{time.strftime('%H:%M:%S')} start {task.id}", flush=True)
        log = LOGS / f"{task.id.replace(':', '_')}.log"
        with log.open("w") as sink:
            proc = subprocess.Popen([sys.executable, str(HERE / "batch.py"), "do", task.id], stdout=sink, stderr=subprocess.STDOUT, start_new_session=True)
            current.append((proc, task))
            code = proc.wait()
            current.clear()
        outcome = replace(task, state="done" if code == 0 else "failed", started=started, took=time.time() - started, error=None if code == 0 else last_line(log))
        save(updated(load(), outcome))
        print(f"{time.strftime('%H:%M:%S')} {outcome.state} {task.id} ({outcome.took:.0f} s)" + (f": {outcome.error}" if outcome.error else ""), flush=True)
    PID.unlink(missing_ok=True)
    print("stopped by request" if STOP.exists() else "queue finished", flush=True)


def stop() -> None:
    BATCH.mkdir(parents=True, exist_ok=True)
    STOP.touch()
    print("will exit after the running task" if runner_pid() else "no runner is going")


def kill() -> None:
    if pid := runner_pid():
        os.kill(pid, signal.SIGTERM)
        print(f"killed runner {pid}")
        return
    queue = load()
    save(tuple(replace(t, state="pending", started=None) if t.state == "running" else t for t in queue))
    print("no runner is going (any stale running task reset)")


def freeze(sig: int, word: str) -> None:
    """Suspend / continue the running task's whole process group (it was started as its own session, so its pid is the group id). Memory stays held."""
    pids = subprocess.run(["pgrep", "-f", f"{HERE / 'batch.py'} do "], capture_output=True, text=True).stdout.split()
    for pid in pids:
        os.killpg(int(pid), sig)
    print(f"{word} {len(pids)} running task(s)")


def retry(song: str | None) -> None:
    queue = load()
    save(tuple(replace(t, state="pending", error=None) if t.state == "failed" and song in (None, t.song) else t for t in queue))
    print(f"{sum(t.state == 'failed' and song in (None, t.song) for t in queue)} failed tasks back to pending")


def skipped_stems(task: Task) -> int:
    return len(json.loads(task.output.read_text())["skipped"]) if task.state == "done" and task.output.exists() else 0


def status() -> None:
    queue = load()
    pid = runner_pid()
    count = {s: sum(t.state == s for t in queue) for s in ("done", "running", "failed", "pending")}
    tracks = [t for t in queue if t.kind == "track"]
    took = [t.took for t in tracks if t.state == "done" and t.took]
    todo = sum(t.state in ("pending", "running") for t in tracks) + sum(t.state == "pending" for t in queue if t.kind == "pool")
    eta = f", about {todo * sum(took) / len(took) / 3600:.1f} h left at {sum(took) / len(took) / 60:.1f} min per track (pooling not counted)" if took else ""
    print(f"runner: {f'going (pid {pid})' if pid else 'not running'}{'; STOP requested' if STOP.exists() else ''}")
    print("tasks: " + ", ".join(f"{n} {s}" for s, n in count.items()) + eta)
    glyph = {"done": "■", "running": "▶", "failed": "✗", "pending": "□"}
    for song in dict.fromkeys(t.song for t in queue):
        mine = [t for t in queue if t.song == song]
        pool, rest = mine[0], mine[1:]
        warned = sum(skipped_stems(t) for t in rest)
        print(f"  {song:26s} pool {glyph[pool.state]}  {''.join(glyph[t.state] for t in rest)}  {sum(t.state == 'done' for t in rest)}/{len(rest)}" + (f"  ({warned} stem-sections skipped)" if warned else ""))
    for t in queue:
        if t.state == "running" and t.started:
            print(f"now: {t.id} for {(time.time() - t.started) / 60:.1f} min  (log: {LOGS / t.id.replace(':', '_')}.log)")
        if t.state == "failed":
            print(f"failed: {t.id}: {t.error}")


def dry() -> None:
    """The sections of every pooled song's pairs (a song that is not pooled has no levels file yet, which the level filter needs)."""
    for song in dict.fromkeys(t.song for t in load()):
        if not pooled(song):
            print(f"  {song:26s} not pooled yet")
            continue
        print(f"  {song:26s} " + "  ".join("+".join(f"{chr(65 + s.kind)}{s.seconds:.0f}" for s in pair) for pair in sections_of(song)))


if __name__ == "__main__":
    command, *args = sys.argv[1:] or ["status"]
    {
        "dry": dry, "pause": lambda: freeze(signal.SIGSTOP, "paused"), "resume": lambda: freeze(signal.SIGCONT, "resumed"),
        "plan": lambda: plan("--force" in args), "status": status, "run": run, "stop": stop, "kill": kill,
        "retry": lambda: retry(args[0] if args else None), "do": lambda: do(args[0]),
    }[command]()
