#!/usr/bin/env python3
"""Resumable local CLI for training and using FreeVideo H3 LoRAs."""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
STATE = ROOT / "state" / "runs"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_ZIP_MEMBER = 25 * 1024 * 1024
MAX_ZIP_TOTAL = 500 * 1024 * 1024
ANGLES = ("front", "left-three-quarter", "right-three-quarter", "left-profile",
          "right-profile", "slightly-above", "slightly-below", "waist-up",
          "full-body", "back", "other")
_LOCAL_LOCK = threading.RLock()


@contextmanager
def run_lock(path):
    """Serialize checkpoints across processes; OS locks release after a crash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCAL_LOCK:
        with path.open("a+b") as stream:
            if os.name == "nt":
                import msvcrt
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class PipelineError(Exception):
    pass


def compact(value):
    print(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_zip(archive, destination):
    """Safely extract a small photo archive; reject links, traversal and bombs."""
    destination = Path(destination).resolve()
    total = 0
    with zipfile.ZipFile(archive) as source:
        planned = []
        seen = set()
        for item in source.infolist():
            name = item.filename.replace("\\", "/")
            pure = PurePosixPath(name)
            mode = item.external_attr >> 16
            invalid_windows = any(
                part in (".", "..") or part.endswith((".", " ")) or
                any(char in part for char in '<>:"|?*') or
                part.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *("COM" + str(n) for n in range(1, 10)), *("LPT" + str(n) for n in range(1, 10))}
                for part in pure.parts)
            if (pure.is_absolute() or ".." in pure.parts or
                    not pure.parts or invalid_windows):
                raise PipelineError("ZIP contains an unsafe path")
            if mode & 0o170000 == 0o120000:
                raise PipelineError("ZIP symbolic links are not accepted")
            if item.is_dir():
                continue
            folded = "/".join(pure.parts).casefold()
            if folded in seen:
                raise PipelineError("ZIP contains duplicate or case-colliding paths")
            seen.add(folded)
            if item.file_size > MAX_ZIP_MEMBER:
                raise PipelineError("ZIP photo exceeds 25 MB")
            total += item.file_size
            if total > MAX_ZIP_TOTAL:
                raise PipelineError("ZIP expands beyond 500 MB")
            target = (destination / Path(*pure.parts)).resolve()
            if target != destination and destination not in target.parents:
                raise PipelineError("ZIP contains an unsafe path")
            planned.append((item, target))
        if destination.exists():
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=destination.name + ".tmp-", dir=destination.parent))
        try:
            for item, target in planned:
                relative = target.relative_to(destination)
                output = temporary / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                with source.open(item) as src, output.open("wb") as dst:
                    while True:
                        block = src.read(1024 * 1024)
                        if not block:
                            break
                        dst.write(block)
            os.replace(temporary, destination)
        except Exception:
            import shutil
            shutil.rmtree(temporary, ignore_errors=True)
            raise


def photo_source(request, request_path):
    value = request.get("photos")
    if isinstance(value, dict):
        value = value.get("path")
    if not isinstance(value, str) or not value:
        raise PipelineError("photos must name a directory or ZIP archive")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = request_path.parent / candidate
    candidate = candidate.resolve()
    if candidate.is_file() and candidate.suffix.lower() == ".zip":
        if candidate.stat().st_size > MAX_ZIP_TOTAL:
            raise PipelineError("ZIP archive exceeds 500 MB")
        extraction = STATE / "prepared" / file_sha256(candidate)[:24]
        if not extraction.exists():
            extract_zip(candidate, extraction)
        candidate = extraction
    if not candidate.is_dir():
        raise PipelineError("photos path must be a directory or ZIP archive")
    rows = []
    seen_images = set()
    for image in sorted(candidate.rglob("*")):
        if image.is_file() and image.suffix.lower() in IMAGE_EXTS:
            if image.is_symlink():
                raise PipelineError("Photo directory contains a symbolic link")
            caption_file = image.with_suffix(".txt")
            if not caption_file.is_file() or caption_file.is_symlink():
                return [], {"status": "needs_input", "reason": "missing_caption", "photo": image.name,
                            "caption_file": caption_file.name}
            caption = caption_file.read_text(encoding="utf-8-sig").strip()
            if not caption:
                return [], {"status": "needs_input", "reason": "empty_caption", "photo": image.name,
                            "caption_file": caption_file.name}
            if image.stat().st_size > MAX_ZIP_MEMBER:
                raise PipelineError("Use photos up to 25 MB")
            image_hash = file_sha256(image)
            if image_hash in seen_images:
                raise PipelineError("Photo directory contains duplicate image content")
            seen_images.add(image_hash)
            rows.append({"path": str(image), "name": image.name,
                         "image_hash": image_hash, "caption": caption,
                         "caption_hash": sha256(caption.encode("utf-8"))})
    if not rows:
        raise PipelineError("No JPG, PNG or WebP photos found")
    return rows, None


def load_request(path):
    request_path = Path(path).expanduser().resolve()
    value = read_json(request_path)
    if not isinstance(value, dict) or value.get("version") != 1:
        raise PipelineError("request must be an object with version: 1")
    allowed = {"version", "subject", "photos", "profile", "trigger", "video", "epochs", "rank", "megapixels"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise PipelineError("unknown request field(s): " + ", ".join(unknown))
    subject = value.get("subject")
    if not isinstance(subject, str) or not subject.strip() or len(subject.strip()) > 80:
        raise PipelineError("request subject must contain 1 to 80 characters")
    profile = value.get("profile", "h3-person")
    if profile != "h3-person":
        raise PipelineError("supported profile is h3-person")
    config = {"epochs": value.get("epochs", 50), "rank": value.get("rank", 8),
              "megapixels": value.get("megapixels", .25)}
    if not isinstance(config["epochs"], int) or not 1 <= config["epochs"] <= 500:
        raise PipelineError("epochs must be an integer from 1 to 500")
    if config["rank"] not in (4, 8, 16):
        raise PipelineError("rank must be 4, 8 or 16")
    if not isinstance(config["megapixels"], (int, float)) or not .1 <= config["megapixels"] <= 1:
        raise PipelineError("megapixels must be between 0.1 and 1")
    rows, missing = photo_source(value, request_path)
    if missing:
        return None, missing
    if len(rows) < 5:
        return None, {"status": "needs_input", "reason": "insufficient_photos", "photos_found": len(rows),
                      "minimum_photos": 5, "recommended_photos": "20–30 distinct photos"}
    if len(rows) > 200:
        raise PipelineError("Use at most 200 photos")
    if any(len(row["caption"]) > 2000 for row in rows):
        raise PipelineError("Photo captions must be 2,000 characters or fewer")
    fingerprint_data = {"subject": subject.strip(), "photos": [
        {"image": row["image_hash"], "caption": row["caption_hash"]} for row in rows], "training": config}
    content_key = sha256(json.dumps({k: v for k, v in fingerprint_data.items() if k != "training"}, sort_keys=True, separators=(",", ":")).encode())
    trigger = value.get("trigger")
    if trigger is None:
        slug = "".join(c.lower() if c.isalnum() else "_" for c in subject).strip("_")
        trigger = "fv_" + slug[:20].rstrip("_") + "_" + content_key[:6]
    if not isinstance(trigger, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{2,39}", trigger):
        raise PipelineError("trigger must start with a letter and contain only letters, digits or underscores")
    fingerprint_data["trigger"] = trigger
    fingerprint_data["profile_version"] = "h3-person-v1"
    dataset_fingerprint = sha256(json.dumps({"subject": subject.strip(), "trigger": trigger,
        "photos": fingerprint_data["photos"]}, sort_keys=True, separators=(",", ":")).encode())
    fingerprint = sha256(json.dumps(fingerprint_data, sort_keys=True, separators=(",", ":")).encode())
    video = value.get("video")
    if video is not None:
        if not isinstance(video, dict) or not isinstance(video.get("prompt"), str) or not video["prompt"].strip():
            raise PipelineError("video must contain a non-empty prompt")
        if set(video) - {"prompt", "seconds", "strength"}:
            raise PipelineError("video contains an unknown field")
        video = {"prompt": video["prompt"], "seconds": video.get("seconds", 5), "strength": video.get("strength", .8)}
        if not isinstance(video["seconds"], (int, float)) or not 1.625 <= video["seconds"] <= 60:
            raise PipelineError("video seconds must be between 1.625 and 60")
        if not isinstance(video["strength"], (int, float)) or not -4 <= video["strength"] <= 4:
            raise PipelineError("video strength must be between -4 and 4")
    return {"subject": subject.strip(), "trigger": trigger, "profile": profile, "training": config,
            "photos": rows, "video": video, "fingerprint": fingerprint,
            "dataset_fingerprint": dataset_fingerprint, "request_path": str(request_path)}, None


class SSHTransport:
    """Calls the loopback API through SSH; the bearer token remains on Windows."""
    def __init__(self, config):
        self.config = config
        self.host_scope = config.get("host", "windows-user@windows-host")

    def request(self, method, path, body=None):
        host = self.config.get("host", "windows-user@windows-host")
        key = os.path.expanduser(self.config.get("key", "~/.ssh/windows_pc_ed25519"))
        token_file = self.config.get("token_file", "C:/Users/WINDOWS_USER/FreeVideo/FreeVideo-engine/agent/token")
        root = self.config.get("remote_root", "C:/Users/WINDOWS_USER/FreeVideo/FreeVideo-engine")
        remote_python = self.config.get("remote_python", root + "/python/cpython-3.12.3-windows-x86_64-none/python.exe")
        # Python stdin handles large photo JSON reliably under Windows OpenSSH;
        # PowerShell Console.ReadToEnd can stall on these payloads.
        settings = {"token_file": token_file, "method": method, "path": path, "has_body": body is not None}
        script = "import json,sys,urllib.request,urllib.error,pathlib\n"
        script += "s=" + repr(settings) + "\n"
        script += "token=pathlib.Path(s['token_file']).read_text(encoding='utf-8-sig').strip()\n"
        script += "data=sys.stdin.buffer.readline().rstrip(bytes([10])) if s['has_body'] else None\n"
        script += "r=urllib.request.Request('http://127.0.0.1:8188/freevideo/agent/v1'+s['path'],data=data,method=s['method'],headers={'Authorization':'Bearer '+token,'X-FreeVideo-Client':'Codex','Content-Type':'application/json'})\n"
        script += "try:\n with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(r,timeout=55) as response: print(response.read().decode('utf-8-sig'))\n"
        script += "except urllib.error.HTTPError as e:\n print(e.read(2048).decode('utf-8',errors='replace'));sys.exit(1)\n"
        script += "except urllib.error.URLError as e:\n print(json.dumps({'error':{'code':'server_unavailable','message':'FreeVideo loopback server is unavailable','detail':str(e.reason)[:160]}}));sys.exit(1)\n"
        encoded = base64.b64encode(script.encode()).decode()
        remote = subprocess.list2cmdline([remote_python, '-X', 'utf8', '-c',
                                         "import base64;exec(base64.b64decode('" + encoded + "'))"])
        command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "IdentitiesOnly=yes", "-i", key, host, remote]
        try:
            result = subprocess.run(command, input=json.dumps(body).encode()+b"\n" if body is not None else b"", capture_output=True, timeout=75)
        except subprocess.TimeoutExpired:
            raise PipelineError("SSH/API request timed out; reconnect with continue using the saved run ID") from None
        if result.returncode:
            raise PipelineError("FreeVideo API request failed: " + (result.stdout or result.stderr).decode(errors="replace")[:500])
        return json.loads(result.stdout.decode("utf-8-sig"))



class LocalTransport:
    """Call the Windows loopback API directly when the CLI runs on Windows."""
    def __init__(self, config):
        self.config = config
        self.host_scope = config.get("host", "windows-local")

    def request(self, method, path, body=None):
        token_path = Path(os.path.expandvars(os.path.expanduser(self.config.get(
            "token_file", r"C:\Users\WINDOWS_USER\FreeVideo\FreeVideo-engine\agent\token"))))
        token = token_path.read_text(encoding="utf-8-sig").strip()
        url = "http://127.0.0.1:8188/freevideo/agent/v1" + path
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(url, data=data, method=method,
            headers={"Authorization": "Bearer " + token, "X-FreeVideo-Client": "Codex",
                     **({"Content-Type": "application/json"} if data is not None else {})})
        try:
            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=55) as response:
                return json.loads(response.read().decode("utf-8-sig"))
        except urllib.error.HTTPError as error:
            detail = error.read(2048).decode("utf-8", errors="replace")
            raise PipelineError("FreeVideo API request failed: " + detail[:500]) from None
        except urllib.error.URLError as error:
            raise PipelineError("FreeVideo loopback API is unavailable: " + str(error.reason)[:300]) from None


def default_transport(config):
    kind = config.get("transport", "local" if os.name == "nt" else "ssh")
    if kind == "local":
        return LocalTransport(config)
    if kind == "ssh":
        return SSHTransport(config)
    raise PipelineError("config transport must be 'local' or 'ssh'")


class Pipeline:
    def __init__(self, transport, state_dir=STATE, wait=0):
        self.api, self.state_dir, self.wait = transport, Path(state_dir), wait
        self.host_scope = getattr(transport, "host_scope", "default")

    def save(self, state):
        write_state(self.state_dir / (state["run_id"] + ".json"), state)

    def get(self, run_id):
        if not re.fullmatch(r"(?:[0-9a-f]{20}|adopt[0-9a-f]{15})", run_id):
            raise PipelineError("Invalid run ID")
        path = self.state_dir / (run_id + ".json")
        if not path.is_file():
            raise PipelineError("Unknown local run: " + run_id)
        state = read_json(path)
        if state.get("host_scope", "default") != self.host_scope:
            raise PipelineError("Run belongs to a different configured host")
        return state

    def run(self, request, nonce=None):
        identity = sha256((request["fingerprint"] + self.host_scope + (nonce or "")).encode())
        video_key = json.dumps(request.get("video"), sort_keys=True, separators=(",", ":"))
        run_id = sha256((identity + video_key).encode())[:20]
        if nonce is None:
            for saved_path in self.state_dir.glob("*.json"):
                try:
                    saved = read_json(saved_path)
                except (OSError, ValueError):
                    continue
                if (saved.get("stage") == "needs_input" and saved.get("request_path") == request["request_path"]
                        and saved.get("host_scope", "default") == self.host_scope):
                    run_id = saved["run_id"]
                    break
        with run_lock(self.state_dir / (run_id + ".lock")):
            return self._run_locked(request, nonce, run_id, identity)

    def _run_locked(self, request, nonce, run_id, identity):
        state = self.get(run_id) if (self.state_dir / (run_id + ".json")).exists() else {
            "run_id": run_id, "version": 1, "profile_version": "h3-person-v1", "profile": request["profile"],
            "host_scope": self.host_scope,
            "fingerprint": request["fingerprint"], "training_ref": "fv_" + identity[:16], "request": request,
            "stage": "queued_upload", "uploads": {}, "created": time.time()}
        if state["stage"] == "needs_input":
            state.update(request=request, fingerprint=request["fingerprint"], stage="queued_upload", missing=None)
        self.save(state)
        try:
            self._advance(state)
        except Exception as error:
            # Preserve checkpoint and report a resumable failure without traceback.
            state["last_error"] = str(error)[:500]
            self.save(state)
            raise
        return self._status_locked(run_id, wait=self.wait)

    def _advance(self, state):
        req = state["request"]
        if state["stage"] == "queued_upload":
            capability = self.api.request("GET", "/training")
            if capability.get("client_source_sha256") is not True:
                raise PipelineError("Remote extension lacks client_source_sha256 idempotency; update it before uploading")
            if capability.get("ready") is not True:
                missing = capability.get("missing")
                raise PipelineError("Training backend is not ready" + (": " + ", ".join(missing) if isinstance(missing, list) else ""))
            dataset_key = sha256((req["dataset_fingerprint"] + self.host_scope).encode())
            name = "fv_" + dataset_key[:12]
            datasets = self.api.request("GET", "/training/datasets")
            dataset_rows = datasets if isinstance(datasets, list) else datasets.get("datasets", [])
            found = next((d for d in dataset_rows
                          if d.get("name") == name and d.get("trigger") == req["trigger"]), None)
            if found is None:
                found = next((d for d in dataset_rows if d.get("trigger", "").lower() == req["trigger"].lower()), None)
                if found:
                    if found.get("name") != name:
                        raise PipelineError("Trigger is already assigned to a different photo dataset; choose another trigger")
                    remote_hashes = {photo.get("source_sha256") for photo in found.get("photos", [])}
                    expected_hashes = {photo["image_hash"] for photo in req["photos"]}
                    if not remote_hashes.issubset(expected_hashes):
                        raise PipelineError("Trigger is already assigned to a different photo dataset; choose another trigger")
            if found:
                dataset = found
            else:
                dataset = self.api.request("POST", "/training/datasets", {"name": name, "trigger": req["trigger"], "subject": req["subject"]})
            state["dataset_id"] = dataset.get("id") or dataset.get("dataset", {}).get("id")
            if not state["dataset_id"]:
                raise PipelineError("Dataset creation returned no ID")
            self.save(state)
            rows = req["photos"]
            for row in rows:
                if row["image_hash"] in state["uploads"]:
                    continue
                # API duplicate photos are content-addressed; inspect dataset before retrying a crash gap.
                current = self.api.request("GET", "/training/datasets/" + state["dataset_id"])
                photos = current.get("photos", [])
                old = next((p for p in photos if p.get("source_sha256") == row["image_hash"]), None)
                if old:
                    remote_caption = old.get("caption", "")
                    raw_caption = row["caption"]
                    prefixed_caption = req["trigger"] + ", " + raw_caption
                    if old.get("angle") != "other" or remote_caption not in (raw_caption, prefixed_caption):
                        raise PipelineError("Remote photo with this source hash has different caption or angle")
                    state["uploads"][row["image_hash"]] = old["id"]
                    self.save(state)
                    continue
                data = Path(row["path"]).read_bytes()
                if sha256(data) != row["image_hash"]:
                    raise PipelineError("A source photo changed after the request was prepared; reload the request")
                media = self.api.request("POST", "/media", {"name": row["name"], "data_base64": base64.b64encode(data).decode()})
                file_ref = media.get("file") or (media.get("media") or {}).get("file")
                if not file_ref:
                    raise PipelineError("Media upload returned no file reference")
                photo = self.api.request("POST", "/training/datasets/" + state["dataset_id"] + "/photos",
                    {"file": file_ref, "angle": "other", "caption": row["caption"],
                     "source_sha256": row["image_hash"]})
                photo_row = next((p for p in photo.get("photos", []) if p.get("source_sha256") == row["image_hash"]), None)
                state["uploads"][row["image_hash"]] = (photo_row or {}).get("id", "uploaded")
                self.save(state)
            state["stage"] = "training"
            self.save(state)
        if state["stage"] == "training":
            # Find by deterministic ref if the response was lost after remote creation.
            ref = state.get("training_ref", "fv_" + state["run_id"][:16])
            jobs = self.api.request("GET", "/training/jobs")
            records = jobs if isinstance(jobs, list) else jobs.get("jobs", [])
            job = next((j for j in records if j.get("ref") == ref), None)
            if job is None:
                body = {"dataset": state["dataset_id"], "ref": ref, **req["training"]}
                job = self.api.request("POST", "/training/jobs", body)
            job_id = job.get("id") or job.get("job", {}).get("id")
            if not job_id:
                raise PipelineError("Training submission returned no ID")
            state["training_id"] = job_id
            self.save(state)

    def _poll_training(self, state):
        row = self.api.request("GET", "/training/jobs/" + state["training_id"] + "?wait=" + str(min(25, max(0, self.wait))))
        state.pop("last_error", None)
        state.pop("driver_error", None)
        status = row.get("status") or (row.get("job") or {}).get("status")
        state["training_status"] = status
        phase = row.get("phase") or (row.get("job") or {}).get("phase")
        if phase:
            state["phase"] = str(phase)[:80]
        progress = row.get("progress") or (row.get("job") or {}).get("progress")
        if isinstance(progress, (int, float)) and 0 <= progress <= 100:
            state["progress"] = progress
        if status in ("failed", "cancelled", "interrupted"):
            state["stage"] = "cancelled" if status == "cancelled" else "failed"
            error = str(row.get("error") or status)
            state["failure"] = error if len(error) <= 500 else error[:150] + " ... " + error[-330:]
            self.save(state)
            return
        if status == "succeeded":
            lora = row.get("lora") or (row.get("job") or {}).get("lora")
            adapter = row.get("adapter") or (row.get("job") or {}).get("adapter") or {}
            if (not isinstance(lora, str) or not lora.lower().endswith(".safetensors")
                    or adapter.get("verified") is not True):
                state["stage"] = "failed"
                state["failure"] = "Training did not report a verified LoRA metadata file"
                self.save(state)
                return
            state["lora"] = lora
            if state.get("cancel_requested"):
                state["stage"] = "cancelled"
            elif state["request"].get("video"):
                state["stage"] = "generating"
            else:
                state["stage"] = "succeeded"
            self.save(state)
        else:
            self.save(state)

    def _generate(self, state):
        video = state["request"]["video"]
        if not state.get("video_job_id"):
            result = self.api.request("POST", "/training/jobs/" + state["training_id"] + "/generate",
                {"prompt": video["prompt"], "seconds": video["seconds"], "strength": video["strength"],
                 "ref": "fv_" + state["run_id"][:16] + "_video"})
            state["video_job_id"] = result.get("id") or result.get("job", {}).get("id")
            if not state["video_job_id"]:
                raise PipelineError("Video generation returned no job ID")
            self.save(state)
        result = self.api.request("GET", "/jobs/" + state["video_job_id"] + "?wait=" + str(min(25, max(0, self.wait))))
        state.pop("last_error", None)
        state.pop("driver_error", None)
        status = result.get("status") or (result.get("job") or {}).get("status")
        state["video_status"] = status
        if status in ("succeeded", "complete", "completed"):
            state["stage"] = "succeeded"
            video_value = result.get("video") or result.get("output") or result.get("result")
            if isinstance(video_value, dict) and isinstance(video_value.get("video"), dict):
                state["library_id"] = video_value.get("library_id")
                video_value = video_value["video"]
            if isinstance(video_value, dict):
                keep = ("id", "path", "file", "filename", "url", "bytes", "seconds")
                video_value = {key: video_value[key] for key in keep if key in video_value}
            state["video"] = video_value
        elif status in ("failed", "cancelled", "interrupted", "error"):
            state["stage"] = "cancelled" if status == "cancelled" else "failed"
            state["failure"] = str(result.get("error") or status)[:500]
        self.save(state)

    def status(self, run_id, wait=0):
        self.get(run_id)
        with run_lock(self.state_dir / (run_id + ".lock")):
            try:
                return self._status_locked(run_id, wait)
            except Exception as error:
                self._remember_error(run_id, error)
                raise

    def _status_locked(self, run_id, wait=0):
        state = self.get(run_id)
        wait = min(60, max(0, wait))
        deadline = time.monotonic() + wait
        while state["stage"] in ("training", "generating", "cancelling"):
            remaining = max(0, deadline - time.monotonic())
            if wait and remaining < 1:
                break
            self.wait = min(25, int(remaining)) if wait else 0
            if state["stage"] in ("training", "cancelling") and state.get("training_id") and not state.get("video_job_id"):
                self._poll_training(state)
                state = self.get(run_id)
                if state["stage"] == "generating" and state.get("request", {}).get("video"):
                    self.wait = 0
                    self._generate(state)
            elif state["stage"] in ("generating", "cancelling") and state.get("video_job_id"):
                self._generate(state)
            state = self.get(run_id)
            if not wait:
                break
        result = self.public(state)
        driver_path = self.state_dir / (run_id + ".driver.json")
        if driver_path.is_file():
            meta = read_json(driver_path)
            result["driver"] = {k: meta[k] for k in ("status", "pid", "heartbeat", "last_error") if k in meta}
        if state["stage"] in ("training", "generating", "cancelling"):
            result["next_poll_seconds"] = 30
        return result

    @staticmethod
    def public(state):
        if state.get("stage") == "succeeded":
            state.pop("last_error", None)
        return {k: state[k] for k in ("run_id", "stage", "dataset_id", "training_id", "training_status",
                    "phase", "progress", "cancel_requested", "library_id", "video_job_id", "video_status", "video", "lora", "failure", "last_error", "driver_error") if k in state}

    def cancel(self, run_id):
        self.get(run_id)
        with run_lock(self.state_dir / (run_id + ".lock")):
            return self._cancel_locked(run_id)

    def _cancel_locked(self, run_id):
        state = self.get(run_id)
        if state["stage"] in ("succeeded", "failed", "cancelled"):
            return self.public(state)
        for key, endpoint in (("training_id", "/training/jobs/{}/cancel"), ("video_job_id", "/jobs/{}/cancel")):
            if state.get(key):
                self.api.request("POST", endpoint.format(state[key]), {})
        state["stage"] = "cancelling" if state.get("training_id") or state.get("video_job_id") else "cancelled"
        state["cancel_requested"] = True
        self.save(state)
        return self.public(state)

    def doctor(self):
        response = self.api.request("GET", "/training")
        keys = ("ready", "missing", "api_version", "client_source_sha256", "running", "hardware_note")
        result = {key: response[key] for key in keys if key in response}
        result.setdefault("client_source_sha256", False)
        result["upload_idempotency"] = "available" if result["client_source_sha256"] else "update extension to API 1.1"
        return result

    def record_needs_input(self, request_path, details):
        request_path = str(Path(request_path).expanduser().resolve())
        stable = sha256(("needs-input:" + self.host_scope + request_path).encode())[:20]
        state_path = self.state_dir / (stable + ".json")
        state = {"run_id": stable, "version": 1, "profile_version": "h3-person-v1", "host_scope": self.host_scope,
                 "request_path": request_path, "stage": "needs_input", "missing": details,
                 "created": time.time()}
        with run_lock(self.state_dir / (stable + ".lock")):
            write_state(state_path, state)
        return {"run_id": stable, **details}

    def continue_run(self, run_id, wait=0):
        self.get(run_id)
        with run_lock(self.state_dir / (run_id + ".lock")):
            try:
                return self._continue_locked(run_id, wait)
            except Exception as error:
                self._remember_error(run_id, error)
                raise

    def _remember_error(self, run_id, error):
        path = self.state_dir / (run_id + ".json")
        if path.is_file():
            try:
                state = read_json(path)
                state["last_error"] = str(error)[:500]
                self.save(state)
            except Exception:
                pass

    def _continue_locked(self, run_id, wait=0):
        state = self.get(run_id)
        if state.get("host_scope", "default") != self.host_scope:
            raise PipelineError("Run belongs to a different configured host")
        if state["stage"] == "needs_input":
            request, missing = load_request(state["request_path"])
            if missing:
                state["missing"] = missing
                self.save(state)
                return {"run_id": run_id, **missing}
            state.update(request=request, fingerprint=request["fingerprint"], stage="queued_upload", uploads={},
                         profile_version="h3-person-v1", profile=request["profile"], missing=None)
            self.save(state)
            self._advance(state)
        elif state["stage"] not in ("failed", "succeeded", "cancelled", "cancelling"):
            # Resume only the recorded checkpoint; never submit a new training job blindly.
            if state["stage"] == "training" and not state.get("training_id"):
                self._advance(state)  # resolves the deterministic remote ref first
            else:
                self._advance(state)
        return self._status_locked(run_id, wait)

    def adopt(self, training_id, video=None):
        remote = self.api.request("GET", "/training/jobs/" + training_id)
        if remote.get("status") != "succeeded":
            raise PipelineError("Only a successfully completed training job can be adopted")
        lora = remote.get("lora")
        adapter = remote.get("adapter") or {}
        if (not isinstance(lora, str) or not lora.lower().endswith(".safetensors")
                or adapter.get("verified") is not True):
            raise PipelineError("Training job has no validated LoRA metadata file")
        if video is not None:
            if not isinstance(video, dict) or not isinstance(video.get("prompt"), str) or not video["prompt"].strip():
                raise PipelineError("video must contain a non-empty prompt")
            video = {"prompt": video["prompt"], "seconds": video.get("seconds", 5), "strength": video.get("strength", .8)}
            if not isinstance(video["seconds"], (int, float)) or not 1.625 <= video["seconds"] <= 60:
                raise PipelineError("video seconds must be between 1.625 and 60")
            if not isinstance(video["strength"], (int, float)) or not -4 <= video["strength"] <= 4:
                raise PipelineError("video strength must be between -4 and 4")
        video_key = json.dumps(video, sort_keys=True, separators=(",", ":")) if video else ""
        run_id = "adopt" + sha256((training_id + self.host_scope + video_key).encode())[:15]
        state_path = self.state_dir / (run_id + ".json")
        with run_lock(self.state_dir / (run_id + ".lock")):
            if state_path.exists():
                state = self.get(run_id)
                if state.get("request", {}).get("video") != video:
                    raise PipelineError("Adopted video request changed; use a new run reference")
                if state["stage"] == "generating":
                    self._generate(state)
                return self.public(state)
            state = {"run_id": run_id, "version": 1, "profile_version": "h3-person-v1", "host_scope": self.host_scope,
                     "stage": "generating" if video else "succeeded",
                     "training_id": training_id, "training_status": "succeeded", "lora": lora,
                     "dataset_id": remote.get("dataset"), "adopted_metadata": {k: remote.get(k) for k in ("request", "epochs", "rank", "megapixels", "checkpoint_bytes", "adapter")},
                     "request": {"video": video}, "created": time.time(), "adopted": True}
            self.save(state)
            if video:
                self._generate(state)
            return self.public(state)


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--config", default=str(ROOT / "config.json"), help="SSH/API settings JSON")
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run"); run.add_argument("--request", required=True); run.add_argument("--wait", action="store_true")
    status = commands.add_parser("status"); status.add_argument("run_id"); status.add_argument("--wait", type=int, default=0)
    cont = commands.add_parser("continue"); cont.add_argument("run_id"); cont.add_argument("--wait", action="store_true")
    cancel = commands.add_parser("cancel"); cancel.add_argument("run_id")
    retrain = commands.add_parser("retrain"); retrain.add_argument("--request", required=True)
    adopt = commands.add_parser("adopt"); adopt.add_argument("training_id"); adopt.add_argument("--video", help="JSON file containing prompt and optional seconds/strength")
    commands.add_parser("doctor")
    return root


def main(argv=None, transport_factory=default_transport):
    args = parser().parse_args(argv)
    try:
        if not Path(args.config).is_file():
            raise PipelineError("Config missing; copy config.example.json to config.json and set the connection")
        config = read_json(args.config)
        pipeline = Pipeline(transport_factory(config), wait=25 if getattr(args, "wait", False) else 0)
        if args.command in ("run", "retrain"):
            request, missing = load_request(args.request)
            if missing:
                result = pipeline.record_needs_input(args.request, missing)
                compact(result); return 2
            result = pipeline.run(request, nonce=os.urandom(12).hex() if args.command == "retrain" else None)
        elif args.command == "status":
            result = pipeline.status(args.run_id, min(60, args.wait))
        elif args.command == "continue":
            result = pipeline.continue_run(args.run_id, min(60, 25 if args.wait else 0))
        elif args.command == "cancel":
            result = pipeline.cancel(args.run_id)
        elif args.command == "adopt":
            video = read_json(args.video) if args.video else None
            result = pipeline.adopt(args.training_id, video)
        else:
            result = pipeline.doctor()
        if (args.command in ("run", "retrain", "continue", "adopt")
                and result.get("stage") in ("queued_upload", "training", "generating", "cancelling")):
            try:
                from driver import start_driver
                result["driver"] = start_driver(args.config, result["run_id"])
            except Exception as error:
                result["driver"] = {"status": "error", "error": str(error)[:300],
                                    "next": "Run continue with the saved run ID to reconnect"}
        compact(result)
        return 0
    except Exception as error:
        print("freevideo-lora: " + str(error)[:500], file=sys.stderr)
        compact({"status": "error", "error": str(error)[:500]})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
