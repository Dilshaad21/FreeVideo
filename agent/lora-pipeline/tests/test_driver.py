import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import driver


class FakePipeline:
    def __init__(self, directory, states=None, outcomes=None):
        self.state_dir = Path(directory)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state = {"run_id": "run-1234", "stage": "training"}
        self.outcomes = list(outcomes or [])
        self.calls = 0
        if states:
            self.state.update(states)
        self.save(self.state)

    def get(self, run_id):
        return json.loads((self.state_dir / (run_id + ".json")).read_text())

    def save(self, state):
        self.state = dict(state)
        (self.state_dir / (state["run_id"] + ".json")).write_text(json.dumps(state))

    def continue_run(self, run_id, wait=0):
        self.calls += 1
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            self.state.update(outcome)
            self.save(self.state)


class Process:
    pid = 7777


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state = self.root / "runs"
        self.state.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_driver_moves_from_training_through_video_to_terminal(self):
        pipeline = FakePipeline(self.state, outcomes=[{"stage": "generating", "training_status": "succeeded"},
                                                       {"stage": "succeeded", "video_status": "succeeded"}])
        sleeps = []
        result = driver.run_driver("run-1234", pipeline, state_dir=self.state,
                                   sleep=sleeps.append, clock=lambda: 100)
        self.assertEqual(0, result)
        self.assertEqual(2, pipeline.calls)
        self.assertEqual([30], sleeps)
        meta = json.loads((self.state / "run-1234.driver.json").read_text())
        self.assertEqual("succeeded", meta["status"])
        lines = (self.state / "run-1234.driver.log").read_text().splitlines()
        self.assertEqual(2, len(lines))

    def test_three_consecutive_transient_errors_persist_driver_error(self):
        pipeline = FakePipeline(self.state, outcomes=[OSError("offline"), OSError("offline"), OSError("offline")])
        sleeps = []
        result = driver.run_driver("run-1234", pipeline, state_dir=self.state,
                                   sleep=sleeps.append, clock=lambda: 100)
        self.assertEqual(2, result)
        self.assertEqual([5, 10], sleeps)
        self.assertEqual("offline", pipeline.get("run-1234")["driver_error"])
        meta = json.loads((self.state / "run-1234.driver.json").read_text())
        self.assertEqual("driver_error", meta["status"])

    def test_driver_error_run_state_write_uses_pipeline_state_lock(self):
        lock_depth = []
        original_lock = driver.freevideo_lora.run_lock

        @contextmanager
        def tracked_lock(path):
            with original_lock(path):
                lock_depth.append(True)
                try:
                    yield
                finally:
                    lock_depth.pop()

        class TrackingPipeline(FakePipeline):
            def __init__(self, *args, **kwargs):
                self.lock_observed = False
                super().__init__(*args, **kwargs)

            def save(self, state):
                if "driver_error" in state:
                    self.lock_observed = bool(lock_depth)
                super().save(state)

        pipeline = TrackingPipeline(self.state, outcomes=[OSError("offline")] * 3)
        with mock.patch.object(driver.freevideo_lora, "run_lock", tracked_lock):
            result = driver.run_driver("run-1234", pipeline, state_dir=self.state, sleep=lambda _: None)
        self.assertEqual(2, result)
        self.assertTrue(pipeline.lock_observed)

    def test_windows_pid_check_never_calls_os_kill(self):
        (self.state / "run-1234.driver.json").write_text(json.dumps({
            "pid": 9876, "status": "running", "heartbeat": __import__("time").time()}))
        with mock.patch.object(driver, "_windows_pid_alive", return_value=True) as check:
            with mock.patch.object(driver.os, "kill", side_effect=AssertionError("os.kill used on Windows")):
                result = driver.start_driver("/tmp/config.json", "run-1234", state_dir=self.state,
                                             platform="nt", popen=lambda *a, **k: self.fail("spawned duplicate"))
        check.assert_called_once_with(9876)
        self.assertEqual({"status": "already_running", "pid": 9876}, result)

    def test_transient_error_resets_after_a_success(self):
        pipeline = FakePipeline(self.state, outcomes=[OSError("brief"), {"stage": "succeeded"}])
        sleeps = []
        self.assertEqual(0, driver.run_driver("run-1234", pipeline, state_dir=self.state,
                                              sleep=sleeps.append, clock=lambda: 100))
        self.assertEqual([5], sleeps)
        self.assertNotIn("driver_error", pipeline.get("run-1234"))

    def test_terminal_needs_input_stops_without_polling(self):
        pipeline = FakePipeline(self.state, states={"stage": "needs_input"})
        self.assertEqual(0, driver.run_driver("run-1234", pipeline, state_dir=self.state,
                                              sleep=lambda _: self.fail("terminal run slept")))
        self.assertEqual(0, pipeline.calls)

    def test_duplicate_driver_lock_exits_immediately(self):
        pipeline = FakePipeline(self.state)
        with driver.driver_lock(self.state / "run-1234.driver.lock") as acquired:
            self.assertTrue(acquired)
            self.assertEqual(3, driver.run_driver("run-1234", pipeline, state_dir=self.state))
        self.assertEqual(0, pipeline.calls)

    def test_start_driver_reuses_recent_live_heartbeat(self):
        meta_path = self.state / "run-1234.driver.json"
        meta_path.write_text(json.dumps({"pid": os.getpid(), "status": "running", "heartbeat": __import__("time").time()}))
        # Exercise heartbeat reuse without sending a POSIX signal on Windows.
        with mock.patch.object(driver, "_pid_alive", return_value=True) as alive:
            result = driver.start_driver("/tmp/config.json", "run-1234", state_dir=self.state,
                                         platform="posix", popen=lambda *a, **k: self.fail("spawned duplicate"))
        alive.assert_called_once_with(os.getpid(), "posix")
        self.assertEqual({"status": "already_running", "pid": os.getpid()}, result)

    def test_start_driver_detaches_posix_process_with_absolute_config(self):
        command = {}
        def fake_popen(args, **kwargs):
            command.update(args=args, kwargs=kwargs)
            return Process()
        result = driver.start_driver("relative/config.json", "run-1234", state_dir=self.state,
                                     platform="posix", popen=fake_popen)
        self.assertEqual({"status": "started", "pid": 7777}, result)
        self.assertTrue(Path(command["args"][3]).is_absolute())
        self.assertTrue(command["kwargs"]["start_new_session"])
        self.assertTrue(command["kwargs"]["close_fds"])

    def test_windows_launcher_uses_encoded_quoted_wmi_command(self):
        captured = {}
        def fake_run(args, **kwargs):
            captured.update(args=args, kwargs=kwargs)
            return subprocess.CompletedProcess(args, 0, stdout="4321", stderr="")
        result = driver.start_driver("config with spaces.json", "run-1234", state_dir=self.state,
                                     platform="nt", run=fake_run)
        self.assertEqual({"status": "started", "pid": 4321}, result)
        encoded = captured["args"][-1]
        script = base64.b64decode(encoded).decode("utf-16le")
        self.assertIn("Win32_Process", script)
        self.assertIn(str(Path("config with spaces.json").resolve()), script)


if __name__ == "__main__":
    unittest.main()
