#!/usr/bin/env python3
"""Keep one FreeVideo LoRA run moving in a detached local process."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import errno
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import base64

import freevideo_lora

ROOT = Path(__file__).resolve().parent
STATE = freevideo_lora.STATE
TERMINAL = {"succeeded", "failed", "cancelled", "needs_input"}
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{4,64}$")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, separators=(",", ":"))
            stream.write("\n")
        os.replace(name, path)
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


@contextmanager
def driver_lock(path):
    """A separate, nonblocking per-run lock. Never holds Pipeline's state lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    acquired = False
    try:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                acquired = True
            except OSError:
                acquired = False
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                acquired = False
        yield acquired
    finally:
        if acquired:
            if os.name == "nt":
                import msvcrt
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def _signature(state):
    return tuple(state.get(key) for key in (
        "stage", "training_status", "phase", "progress", "video_status", "failure", "driver_error"))


def _log_change(path, run_id, state, clock):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"time": clock(), "run_id": run_id, "state": _signature_dict(state)},
                                ensure_ascii=False, separators=(",", ":")) + "\n")


def _signature_dict(state):
    keys = ("stage", "training_status", "phase", "progress", "video_status", "failure", "driver_error")
    return {key: state[key] for key in keys if key in state}


def _windows_pid_alive(pid):
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    open_process = kernel.OpenProcess
    open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    open_process.restype = wintypes.HANDLE
    get_exit_code = kernel.GetExitCodeProcess
    get_exit_code.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    get_exit_code.restype = wintypes.BOOL
    close_handle = kernel.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL
    # PROCESS_QUERY_LIMITED_INFORMATION. Do not use os.kill(pid, 0) on Windows.
    handle = open_process(0x1000, False, int(pid))
    if not handle:
        return False
    try:
        exit_code = wintypes.DWORD()
        return bool(get_exit_code(handle, ctypes.byref(exit_code)) and exit_code.value == 259)
    finally:
        close_handle(handle)


def _pid_alive(pid, platform=None):
    platform = platform or os.name
    if platform == "nt":
        return _windows_pid_alive(pid)
    try:
        os.kill(int(pid), 0)
        return True
    except OSError as error:
        return error.errno == errno.EPERM


def run_driver(run_id, pipeline, *, state_dir=STATE, interval=30, max_failures=3,
               clock=time.time, sleep=time.sleep):
    """Poll/continue a run until terminal or three consecutive transient errors."""
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError("invalid run ID")
    state_dir = Path(state_dir)
    lock_path = state_dir / (run_id + ".driver.lock")
    driver_path = state_dir / (run_id + ".driver.json")
    log_path = state_dir / (run_id + ".driver.log")
    with driver_lock(lock_path) as acquired:
        if not acquired:
            return 3
        state = pipeline.get(run_id)
        previous = None
        failures = 0
        meta = {"run_id": run_id, "pid": os.getpid(), "status": "running", "heartbeat": clock(),
                "consecutive_failures": 0}
        atomic_json(driver_path, meta)
        while True:
            try:
                current = pipeline.get(run_id)
                if current.get("stage") in TERMINAL:
                    if _signature(current) != previous:
                        _log_change(log_path, run_id, current, clock)
                    meta.update(status=current["stage"], heartbeat=clock(), consecutive_failures=0)
                    atomic_json(driver_path, meta)
                    return 0
                pipeline.continue_run(run_id, wait=0)
                current = pipeline.get(run_id)
                failures = 0
                signature = _signature(current)
                if signature != previous:
                    _log_change(log_path, run_id, current, clock)
                    previous = signature
                if current.get("stage") in TERMINAL:
                    meta.update(status=current["stage"], heartbeat=clock(), consecutive_failures=0)
                    atomic_json(driver_path, meta)
                    return 0
                meta.update(status="running", heartbeat=clock(), consecutive_failures=0)
                atomic_json(driver_path, meta)
            except Exception as error:
                failures += 1
                summary = str(error).replace("\n", " ")[:300]
                meta.update(status="running", heartbeat=clock(), consecutive_failures=failures,
                            last_error=summary)
                if failures >= max_failures:
                    try:
                        with freevideo_lora.run_lock(state_dir / (run_id + ".lock")):
                            current = pipeline.get(run_id)
                            current["driver_error"] = summary
                            pipeline.save(current)
                    except Exception:
                        pass
                    meta.update(status="driver_error", heartbeat=clock())
                    atomic_json(driver_path, meta)
                    _log_change(log_path, run_id, {"stage": "driver_error", "driver_error": summary}, clock)
                    return 2
                atomic_json(driver_path, meta)
                sleep(min(30, 5 * (2 ** (failures - 1))))
                continue
            sleep(interval)


def _ps_quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def start_driver(config_path, run_id, *, state_dir=STATE, platform=None,
                 popen=subprocess.Popen, run=subprocess.run):
    """Start detached; reuse a recent live heartbeat and return a compact launch result."""
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError("invalid run ID")
    project_root = ROOT.resolve()
    config_path = Path(config_path).expanduser().resolve()
    state_dir = Path(state_dir).resolve()
    driver_file = state_dir / (run_id + ".driver.json")
    state_dir.mkdir(parents=True, exist_ok=True)
    run_file = state_dir / (run_id + ".json")
    if run_file.is_file():
        saved_run = freevideo_lora.read_json(run_file)
        if saved_run.get("stage") in TERMINAL:
            return {"status": saved_run["stage"], "pid": None}
    if driver_file.is_file():
        try:
            meta = freevideo_lora.read_json(driver_file)
            pid = int(meta.get("pid", 0))
            age = time.time() - float(meta.get("heartbeat", 0))
            alive = pid > 0 and _pid_alive(pid, platform)
            if alive and 0 <= age < 120 and meta.get("status") == "running":
                return {"status": "already_running", "pid": pid}
        except (OSError, ValueError, TypeError):
            pass
    python = Path(sys.executable).resolve()
    command = [str(python), str((project_root / "driver.py").resolve()),
               "--config", str(config_path), "--run", run_id]
    platform = platform or os.name
    if platform == "nt":
        # WMI starts a process outside the SSH-created process job object.
        command_line = subprocess.list2cmdline(command)
        script = "$commandLine=" + _ps_quote(command_line) + "; $workingDirectory=" + _ps_quote(str(project_root)) + "; "
        script += "$startup=([wmiclass]'Win32_ProcessStartup').CreateInstance(); $startup.ShowWindow=0; "
        script += "$result=([wmiclass]'Win32_Process').Create($commandLine,$workingDirectory,$startup); "
        script += "if($result.ReturnValue -ne 0){throw ('WMI process creation failed: '+$result.ReturnValue)}; "
        script += "[Console]::Out.Write($result.ProcessId)"
        encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
        result = run(["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                     capture_output=True, text=True, timeout=20)
        if result.returncode:
            raise RuntimeError((result.stderr or result.stdout or "WMI process launch failed")[:300])
        pid = int(result.stdout.strip())
    else:
        log_path = state_dir / (run_id + ".driver-process.log")
        with log_path.open("ab") as output:
            process = popen(command, cwd=str(project_root), stdin=subprocess.DEVNULL,
                            stdout=output, stderr=subprocess.STDOUT, close_fds=True,
                            start_new_session=True)
        pid = process.pid
    # The child writes its own PID/heartbeat. Avoid a parent write racing and
    # overwriting the child's first running heartbeat.
    return {"status": "started", "pid": pid}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", required=True)
    args = parser.parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    config = freevideo_lora.read_json(config_path)
    pipeline = freevideo_lora.Pipeline(freevideo_lora.default_transport(config), wait=0)
    return run_driver(args.run, pipeline)


if __name__ == "__main__":
    raise SystemExit(main())
