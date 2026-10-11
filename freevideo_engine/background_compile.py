"""Compile recent request shapes while an interactive session is idle.

The first request after an update, a repair, a new installation or a GPU
driver update compiles its kernels; on a 12 GB Windows laptop that cost 50 s of
a 564 s request. Nobody waits for this compile here. Once ComfyUI has had no
work for START_DELAY seconds, precompile runs at the lowest CPU and disk
priority. A queued prompt stops it within POLL seconds, and a generation, a
prompt rewrite or a calibration stops it before taking the GPU. Kernels
compiled by then stay in the caches, and the request compiles the rest as it
always did.

It runs only before the session's first generation. After that, the resident
encoder holds GPU memory that a request would release, so the plan would no
longer be the request's. An interrupted run writes no receipt. When the run
was stopped or found no room (background_room), the session waits twice as
long before the next try, up to MAX_DELAY; the next start tries again too.
"""
import logging
from pathlib import Path
import subprocess
import threading
import time

from . import processes

START_DELAY = 30.
MAX_DELAY = 600.
POLL = .25
# Longer than the Linux supervisor takes to end every descendant (4.5 s):
# killing it first would orphan a worker in its own session. A tree that
# exits sooner returns as soon as it does.
STOP_GRACE = 6.
# Windows ends a stopped tree asynchronously: CUDA memory and the lease can
# outlive the top process by seconds.
RELEASE_SECONDS = 15.
ATTEMPTS = 8


class BackgroundCompile:
    def __init__(self):
        self.guard = threading.Lock()
        self.closed = threading.Event()
        self.process = None
        self.thread = None
        self.lock = None
        self.attempts = []

    def start(self, command, *, environment, cwd, log, lock, busy, delay=START_DELAY):
        """Run `command` once the session is quiet; False when this session already did.

        `lock` is the engine lease the compile holds while it runs.
        """
        with self.guard:
            if self.thread is not None or self.closed.is_set():
                return False
            self.lock = Path(lock)
            self.thread = threading.Thread(target=self._work, args=(command, environment, cwd, Path(log), busy, delay),
                                           name='FreeVideo background compile', daemon=True)
            self.thread.start()
            return True

    def stop(self, *, final=False):
        """Stop a running compile and return once its processes, GPU memory and lease are gone.

        `final` when a generation starts: the session compiles nothing more.
        """
        if final:
            self.closed.set()
        tick = time.monotonic()  # One deadline from here, a concurrent stop included.
        with self.guard:
            process, self.process = self.process, None
            if process is None:
                return None
            row = self.attempts[-1]
            if process.poll() is None:
                # Marked before the signal: the worker thread then retries
                # instead of reading the exit as a finished run.
                row['stopped'] = True
                processes.stop(process, grace=STOP_GRACE)
                row['stop_seconds'] = time.monotonic() - tick
            # Every process of the tree holds the lease; once it is free the
            # tree is gone, even when the top process had already exited. On a
            # timeout the request reports the busy lease. One deadline for all.
            row['released'] = self._released(tick + RELEASE_SECONDS)
            row['release_seconds'] = time.monotonic() - tick
            if not row.get('stopped'):
                return None
            logging.info('[FreeVideo] Background compile stopped in %.1f s; the GPU lease is %s.',
                         row['release_seconds'], 'free' if row['released'] else 'still held')
            return dict(row)

    def _released(self, deadline):
        from .locking import runtime_lock
        while True:
            try:
                with runtime_lock(self.lock, inherit=False):
                    return True
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    return False
                time.sleep(.1)

    def _quiet(self, busy, delay):
        """True after `delay` seconds without work; False once closed."""
        since = time.monotonic()
        while time.monotonic() - since < delay:
            if self.closed.wait(1.):
                return False
            if busy():
                since = time.monotonic()
        return True

    def _work(self, command, environment, cwd, log, busy, delay):
        for attempt in range(ATTEMPTS):
            if attempt:
                delay = min(2 * delay, MAX_DELAY)
            if not self._quiet(busy, delay):
                return
            with self.guard:
                if self.closed.is_set():
                    return
                logging.info('[FreeVideo] Compiling GPU kernels for recent video sizes in the background; '
                             'a generation request stops it.')
                row = dict(started=time.time())
                self.attempts.append(row)
                with log.open('ab') as stream:
                    process = self.process = processes.popen(
                        command, env=environment, cwd=cwd, stdin=subprocess.DEVNULL, stdout=stream,
                        stderr=subprocess.STDOUT, start_new_session=True, supervise=True)
            while process.poll() is None:
                if busy():
                    self.stop()
                    break
                try:
                    process.wait(timeout=POLL)
                except subprocess.TimeoutExpired:
                    pass
            with self.guard:
                # A stop in progress holds the guard and has marked the row.
                stopped = row.get('stopped')
                if not stopped and self.process is process:
                    # Ended by itself: confirm its tree let go before forgetting
                    # it, so a later stop never waits on someone else's lease.
                    row['released'] = self._released(time.monotonic() + RELEASE_SECONDS)
                    if row['released']:
                        self.process = None
            row.update(returncode=process.poll(), seconds=time.time() - row['started'])
            from .precompile import BUSY_EXIT
            if not stopped and row['returncode'] != BUSY_EXIT:
                return  # Compiled, or failed with a receipt that says why.


COMPILE = BackgroundCompile()
