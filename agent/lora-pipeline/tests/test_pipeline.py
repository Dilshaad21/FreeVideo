import json
from pathlib import Path
import tempfile
import unittest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import freevideo_lora as cli


class FakeTransport:
    def __init__(self, status="succeeded", fail_after_photo=False):
        self.calls = []
        self.datasets = []
        self.jobs = []
        self.status = status
        self.fail_after_photo = fail_after_photo
        self.failed_once = False

    def request(self, method, path, body=None):
        self.calls.append((method, path, body))
        if method == "GET" and path == "/training":
            return {"ready": True, "api_version": "1.1", "client_source_sha256": True}
        if method == "GET" and path == "/training/datasets":
            return {"datasets": self.datasets}
        if method == "POST" and path == "/training/datasets":
            row = {"id": "d123", **body, "photos": []}
            self.datasets.append(row)
            return row
        if method == "GET" and path == "/training/datasets/d123":
            return self.datasets[0]
        if method == "POST" and path == "/media":
            return {"file": "input/photo.jpg"}
        if method == "POST" and path == "/training/datasets/d123/photos":
            row = {"id": "p123", **body}
            row["caption"] = self.datasets[0]["trigger"] + ", " + body["caption"]
            self.datasets[0]["photos"].append(row)
            if self.fail_after_photo and not self.failed_once:
                self.failed_once = True
                raise OSError("connection dropped after commit")
            return self.datasets[0]
        if method == "GET" and path == "/training/jobs":
            return {"jobs": self.jobs}
        if method == "POST" and path == "/training/jobs":
            row = {"id": "t123", **body}
            self.jobs.append(row)
            return row
        if method == "GET" and path == "/training/jobs/t123?wait=0":
            return {"id": "t123", "status": self.status,
                    "lora": "trained/subject.safetensors" if self.status == "succeeded" else None,
                    "adapter": {"verified": True} if self.status == "succeeded" else None}
        if method == "POST" and path == "/training/jobs/t123/cancel":
            return {"status": "cancelled"}
        if method == "GET" and path == "/training/jobs/t-existing":
            return {"id": "t-existing", "status": "succeeded", "lora": "trained/subject.safetensors",
                    "adapter": {"verified": True, "family": "minimax-h3"}}
        if method == "POST" and path == "/training/jobs/t-existing/generate":
            return {"id": "j123", **body}
        if method == "GET" and path == "/jobs/j123?wait=0":
            return {"id": "j123", "status": "complete", "video": "videos/result.mp4"}
        raise AssertionError(f"Unexpected request {method} {path}")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.photos = self.root / "photos"
        self.photos.mkdir()
        for index in range(5):
            image = self.photos / f"view{index}.jpg"
            image.write_bytes(f"jpeg bytes {index}".encode())
            image.with_suffix(".txt").write_text(f"person view {index}", encoding="utf-8")
        self.request_file = self.root / "request.json"
        self.request_file.write_text(json.dumps({"version": 1, "subject": "Mira",
            "photos": str(self.photos)}), encoding="utf-8")
        self.request, missing = cli.load_request(self.request_file)
        self.assertIsNone(missing)

    def tearDown(self):
        self.tmp.cleanup()

    def test_repeat_request_reuses_local_and_remote_ids(self):
        api = FakeTransport()
        pipeline = cli.Pipeline(api, self.root / "state")
        first = pipeline.run(self.request)
        second = pipeline.run(self.request)
        self.assertEqual(first["run_id"], second["run_id"])
        self.assertEqual("succeeded", second["stage"])
        self.assertEqual(1, len([c for c in api.calls if c[1] == "/training/jobs" and c[0] == "POST"]))
        self.assertEqual(5, len(api.datasets[0]["photos"]))

    def test_new_scene_reuses_training_but_submits_distinct_video(self):
        class Videos(FakeTransport):
            def request(self, method, path, body=None):
                if method == 'POST' and path == '/training/jobs/t123/generate':
                    self.calls.append((method, path, body))
                    return {'job': {'id': 'j123'}}
                if method == 'GET' and path == '/jobs/j123?wait=0':
                    return {'status': 'succeeded', 'result': {'library_id': 'day/video',
                            'video': {'file': 'video.mp4', 'url': '/agent/video'}}}
                return super().request(method, path, body)
        api=Videos()
        pipeline=cli.Pipeline(api,self.root/'state')
        first=pipeline.run(dict(self.request,video={'prompt':'Garden','seconds':5,'strength':.8}))
        second=pipeline.run(dict(self.request,video={'prompt':'Beach','seconds':5,'strength':.8}))
        self.assertNotEqual(first['run_id'],second['run_id'])
        self.assertEqual(first['training_id'],second['training_id'])
        self.assertEqual('/agent/video',second['video']['url'])
        self.assertEqual(1,len([c for c in api.calls if c[:2]==('POST','/training/jobs')]))
        prompts=[c[2]['prompt'] for c in api.calls if c[1].endswith('/generate')]
        self.assertEqual(['Garden','Beach'],prompts)

    def test_cancel_waits_for_remote_terminal_state(self):
        api=FakeTransport(status='running')
        pipeline=cli.Pipeline(api,self.root/'state')
        run=pipeline.run(self.request)
        requested=pipeline.cancel(run['run_id'])
        self.assertEqual('cancelling',requested['stage'])
        api.status='cancelled'
        self.assertEqual('cancelled',pipeline.status(run['run_id'])['stage'])

    def test_invalid_run_id_cannot_create_lock_outside_state(self):
        pipeline=cli.Pipeline(FakeTransport(),self.root/'state')
        with self.assertRaises(cli.PipelineError):pipeline.status('../escape')
        self.assertFalse((self.root/'escape.lock').exists())

    def test_resume_recovers_photo_committed_before_local_checkpoint(self):
        api = FakeTransport(fail_after_photo=True)
        pipeline = cli.Pipeline(api, self.root / "state")
        with self.assertRaisesRegex(OSError, "connection dropped"):
            pipeline.run(self.request)
        result = pipeline.run(self.request)
        self.assertEqual("succeeded", result["stage"])
        photo_posts = [c for c in api.calls if c[1].endswith("/photos") and c[0] == "POST"]
        self.assertEqual(5, len(photo_posts))
        self.assertEqual(5, len(api.datasets[0]["photos"]))
        self.assertIn(self.request["photos"][0]["image_hash"],
                      [photo["source_sha256"] for photo in api.datasets[0]["photos"]])

    def test_failed_training_is_not_restarted(self):
        api = FakeTransport(status="failed")
        pipeline = cli.Pipeline(api, self.root / "state")
        first = pipeline.run(self.request)
        second = pipeline.run(self.request)
        self.assertEqual("failed", first["stage"])
        self.assertEqual("failed", second["stage"])
        self.assertEqual(1, len([c for c in api.calls if c[1] == "/training/jobs" and c[0] == "POST"]))

    def test_missing_caption_is_structured_needs_input(self):
        (self.photos / "view0.txt").unlink()
        _, missing = cli.load_request(self.request_file)
        self.assertEqual("needs_input", missing["status"])
        self.assertEqual("missing_caption", missing["reason"])

    def test_too_few_photos_is_reported_before_remote_upload(self):
        for index in range(2, 5):
            (self.photos / f"view{index}.jpg").unlink()
            (self.photos / f"view{index}.txt").unlink()
        _, missing = cli.load_request(self.request_file)
        self.assertEqual("needs_input", missing["status"])
        self.assertEqual("insufficient_photos", missing["reason"])
        self.assertEqual(2, missing["photos_found"])

    def test_continue_reloads_fixed_caption_with_same_run_id(self):
        (self.photos / "view0.txt").unlink()
        api = FakeTransport()
        pipeline = cli.Pipeline(api, self.root / "state")
        pending = pipeline.record_needs_input(self.request_file, {
            "status": "needs_input", "reason": "missing_caption", "photo": "view0.jpg"})
        (self.photos / "view0.txt").write_text("person view 0", encoding="utf-8")
        result = pipeline.continue_run(pending["run_id"])
        self.assertEqual(pending["run_id"], result["run_id"])
        self.assertEqual("succeeded", result["stage"])

    def test_zip_path_traversal_is_rejected(self):
        archive = self.root / "bad.zip"
        import zipfile
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("../escape.txt", "bad")
        with self.assertRaisesRegex(cli.PipelineError, "unsafe path"):
            cli.extract_zip(archive, self.root / "unpacked")

    def test_zip_symbolic_link_is_rejected(self):
        import zipfile
        archive = self.root / "link.zip"
        info = zipfile.ZipInfo("photo.jpg")
        info.create_system = 3
        info.external_attr = (0o120777 << 16)
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr(info, "outside.jpg")
        with self.assertRaisesRegex(cli.PipelineError, "symbolic links"):
            cli.extract_zip(archive, self.root / "links")

    def test_replacing_zip_changes_content_identity(self):
        import zipfile
        archive = self.root / "photos.zip"
        request_file = self.root / "archive-request.json"
        def write_zip(prefix):
            with zipfile.ZipFile(archive, "w") as output:
                for index in range(5):
                    output.writestr(f"person{index}.jpg", f"{prefix} image {index}")
                    output.writestr(f"person{index}.txt", "person in a blue shirt")
        write_zip("first")
        request_file.write_text(json.dumps({"version": 1, "subject": "Mira", "photos": str(archive)}))
        first, _ = cli.load_request(request_file)
        write_zip("replacement")
        second, _ = cli.load_request(request_file)
        self.assertNotEqual(first["fingerprint"], second["fingerprint"])

    def test_adopt_video_is_repeatable_and_distinct_requests_get_distinct_runs(self):
        api = FakeTransport()
        pipeline = cli.Pipeline(api, self.root / "state")
        video = {"prompt": "person smiling", "seconds": 5, "strength": .8}
        first = pipeline.adopt("t-existing", video)
        repeated = pipeline.adopt("t-existing", video)
        changed = pipeline.adopt("t-existing", {"prompt": "person waves", "seconds": 5, "strength": .8})
        self.assertEqual(first["run_id"], repeated["run_id"])
        self.assertEqual("succeeded", repeated["stage"])
        self.assertNotEqual(first["run_id"], changed["run_id"])
        generations = [c for c in api.calls if c[0] == "POST" and c[1].endswith("/generate")]
        self.assertEqual(2, len(generations))


if __name__ == "__main__":
    unittest.main()
