# FreeVideo character replacement POC

Backend: **Wan2.2-Animate via WanGP**, with SAM 2 tracking. This is a separate
experimental engine, accessed through the authenticated FreeVideo agent API.
It does not operate the ComfyUI UI or submit ComfyUI workflows.

Upload a short source video and a clear replacement-character picture using
FreeVideo's existing `/freevideo/agent/v1/media` upload endpoint. Supply their
returned input file identifiers and a normalized first-frame bounding box:

```json
{
  "ref": "my-first-replacement",
  "source": "uploaded-source.mp4",
  "character": "uploaded-character.png",
  "box": [0.2, 0.05, 0.8, 0.98],
  "prompt": "A person performing the original action with natural lighting."
}
```

`box` is `[left, top, right, bottom]`, each from 0 to 1. Select exactly one
person. First POC: one continuous shot, 1–5 seconds, constant frame rate,
at most 60 fps, no cuts. Prefer an unobstructed person and a similar body size.
Use a character image including the intended outfit and body, plus a clear face.
Our existing H3 identity LoRA cannot be loaded into Wan; its training photos
can supply the character reference instead. No retraining is required here.

API under `/freevideo/agent/v1`:

- `GET /replacement`: setup/readiness (experimental, never a quality guarantee).
- `POST /replacement/jobs`: create an idempotent job with the JSON above.
- `GET /replacement/jobs/{id}?wait=25`: compact status.
- `POST /replacement/jobs/{id}/cancel`: cancel this job.
- `GET /replacement/jobs/{id}/files/{name}`: stream a completed deliverable.
- `GET /replacement/jobs/{id}/preview`: local browser playback of a completed MP4.

Routes require the existing agent token and loopback access. The read-only
preview uses the same local browser access checks as FreeVideo's video player,
so it can play through the SSH tunnel without exposing a token. The token
remains on Windows. All source media, masks and outputs remain on Windows.
Job files and logs persist under `agent/replacement/runs/`.

The adjacent CLI reuses the existing LoRA pipeline's SSH/local transport:

```sh
python freevideo_replace.py doctor
python freevideo_replace.py run --request request.json
python freevideo_replace.py status r0123456789ab --wait 25
python freevideo_replace.py cancel r0123456789ab
```

Use `request.example.json` for local file uploads, or supply `source` and
`character` as existing FreeVideo upload IDs. On Windows pass
`--pipeline ../lora-pipeline` before the subcommand. Neither transport copies
credentials to the Mac. Polling uses compact JSON, without inference logs.

## Preservation contract

The source is decoded once into RGB frames. Only a dilated union of the source
person mask and generated-person mask is replaced. Output keeps the source
frame count, dimensions and constant frame rate. Original audio is copied to a
lossless FFV1/MKV master. A separate MP4 preview is lossy. Every master frame is
decoded and checked: **RGB pixels outside the saved edit region must match the
decoded source exactly**. The original source file is never overwritten.

This does not guarantee that pixels inside the edit region depict a convincing
replacement. Background newly revealed by a different silhouette is generated.
Shadows, reflections, hair and foreground occlusion need visual review and may
need corrected masks. Do not advertise this POC as solved for arbitrary footage.

## Installation

Run `setup_windows.py --root ENGINE_ROOT` with the existing trainer Python.
It prepares an isolated environment, pinned WanGP and SAM 2 source, then checks
runtime imports. `--models` additionally downloads quantized Animate and its
supporting weights. Setup is explicit and idempotent, not part of every job.
The replacement environment reads existing trainer packages through a `.pth`
file; any replacement-specific packages install locally. It never changes the
inference or trainer packages. Setup logs record the dependency snapshot.

Run `install_extension.py --root ENGINE_ROOT --source EXACT_SOURCE` using the
FreeVideo Python while the server is idle. It backs up and patches only known
integration anchors. Restart FreeVideo after installation. Unknown layouts
are refused. The existing LoRA installer must not overwrite these additional
integration anchors without reapplying this extension.

`replacement_worker.py` holds FreeVideo's existing GPU lock for the entire process. Tracking,
generation and compositing run sequentially. WanGP uses its failsafe offload
profile 5, SDPA, 480p-class working resolution, and 81-frame windows. The optional
smart memory pinning ring is disabled: Animate can otherwise reserve about 3.6 GB
of host RAM for faster transfers. This trades transfer speed for available RAM;
the video settings and resource guards stay unchanged. Actual
8 GB VRAM / 16 GB RAM suitability still needs a real input benchmark.

On Windows, an optional backend setting `windows_working_set_mib` caps only the
generation child's resident RAM through `SetProcessWorkingSetSizeEx`. The default
is disabled (0). A value such as 6144 allows Windows to page model weights more
aggressively while leaving system headroom. This can be much slower and does not
cap virtual memory or guarantee model fit. The child verifies the hard limit,
and existing RAM/temperature guards remain active. Changing this setting changes
the execution signature and requires a new deliberate request reference.
Microsoft API: https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-setprocessworkingsetsizeex


Dependencies retain their upstream licenses: WanGP Community License, Wan
model license and SAM 2 Apache 2.0. See `provenance.json` for pinned sources.

## Verification

`python -m unittest discover -s tests -v` checks idempotency, validation,
mask-boundary preservation and the real FFV1 encode/decode round trip when
NumPy, Pillow and imageio-ffmpeg are available. A synthetic compositor test is
not evidence of a successful AI character replacement. Read `verification.json`
for the actual deployment and end-to-end status.

`prefetch.py --root ENGINE_ROOT` can stage the large weights without loading
the GPU. Wait for `agent/replacement/prefetch.json` to say `succeeded` before
using `setup_windows.py --models`; do not run two downloaders on the same files.
Alternatively, `finish_setup.py --root ENGINE_ROOT` waits for that prefetch and
any existing runtime probe, then runs the final runtime/assets check itself.
Its state is in `finish.json`; it stops on errors instead of repeatedly retrying.
Use `setup_windows.py --skip-install` to recheck an already prepared environment
without changing dependencies. `ready: true` is required before submitting a job.

## Recovery and resource controls (0.2)

Submit once with a stable `ref`. The CLI caches uploaded file IDs and the job ID
by host and input hashes. If the connection drops after submission, repeat the
same command: it reuses uploads and the server's idempotent request. Changed
inputs with the same ref are rejected. The client requires API 0.2.

An exited worker becomes `interrupted`; a surviving owned child becomes
`orphaned`. Neither starts another render automatically. Inspect compact status,
then use:

```sh
python freevideo_replace.py resume r0123456789ab
```

This calls `POST /replacement/jobs/{id}/resume`. Resume keeps the ID and verifies
stage SHA256 receipts. Unchanged completed stages are skipped; invalid stages
and their dependent stages are recomputed. A completed generation receipt can
be adopted after a crash before journal commit. If generation never produced
that receipt, explicitly resuming may rerun generation. Pipeline/configuration, dependency snapshot
or core weight size/timestamp changes require a new ref. Cancel an `orphaned`
job before resuming; PID creation times guard against stopping unrelated reused
PIDs. Windows Job Objects kill runtime descendants if their worker exits.

Source inputs are frozen and hashed before work. Output is published by directory
rename only after frame counts, frame rate, lossless protected pixels, audio
codec preservation and preview decoding pass. Lossy preview pixels have no exact
preservation guarantee. Logs, masks, stage receipts and resource peaks remain
on Windows for diagnosis.

`agent/replacement/backend.json` controls limits. Defaults: at least 8 GB free
disk and 2 GB available RAM at start, plus source-size disk estimates. The worker
checks available RAM and NVIDIA telemetry every 10 seconds, stopping after
15 seconds below 512 MiB available RAM or 30 seconds at/above 87°C. These are
software stop thresholds, not a thermal safety guarantee. Unavailable GPU
telemetry is reported explicitly. Stage limits default to 15 minutes, or two
hours for generation; override per stage with `phase_timeout_seconds`.

Setup routes WanGP explicit asset downloads through our resumable downloader.
Folder snapshots retain the pinned upstream resolver with HTTP timeouts.
Downloads use exclusive locks, validate HTTP resume ranges and declared lengths,
and validate checkpoint structure before promotion. SHA256 receipts detect later
changes; they are not independent upstream authenticity proofs. Interrupted
`.part` files remain resumable. A checksum failure requires investigation;
do not automatically delete and redownload large files.

For a server update, stage this package, run its tests in the Windows replacement
environment, then copy it to `agent/replacement-pipeline` while all replacement,
setup and prefetch processes are idle. Preserve a backup of the previous package.
The installer validates `source-sha256.json`, serializes against submissions,
backs up integration files, and restores them if a patch fails. Restart the idle
server and require the versioned doctor to pass. Roll back both the pipeline
package and the backed-up integration files together if validation fails.

Local CLI state is in `state/`; it contains IDs and hashes, never bearer tokens.
If server uploads were manually removed, retain run records and remove only the
corresponding `upload-*.json` entries before retrying an unsubmitted request.
Never remove a recorded job ID merely to restart a render.
