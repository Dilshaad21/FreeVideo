---
name: freevideo-lora
description: Train a person-specific LoRA and optionally generate a video through the configured FreeVideo Windows installation, using the project CLI and persisted run IDs.
---

# FreeVideo LoRA workflow

Use this skill when asked to train a person LoRA or generate a video with one through this project. Run the CLI from the project directory with `python freevideo_lora.py`; check `--help` if the installed command differs. The default config is `config.json`. A safe template is in `config.example.json`.

Prepare a version 1 JSON request with `subject`, `photos`, and `profile: "h3-person"` (the profile defaults to this if omitted). `photos` can name a ZIP archive or a directory of JPG, PNG, or WebP images with adjacent `.txt` captions. If a caption is missing, the CLI returns `needs_input`; provide the missing information and continue the same run. Add `trigger` only when the user supplies or requests a specific trigger phrase. Add `video` only when video generation is requested; it contains a prompt and optional duration in seconds and strength. Optional `epochs`, `rank`, and `megapixels` fields can override training settings when specifically requested. See [examples/request.json](../../examples/request.json), [request.schema.json](../../request.schema.json), and the [photo guide](references/photo-guide.md).

Submit with `python freevideo_lora.py run --request REQUEST.json`. Use `--wait` when it is useful to keep the current invocation attached until the run completes or reports a meaningful state change. Otherwise, record the returned run ID and check progress with `status RUN_ID --wait SECONDS`; use bounded waits and report meaningful state changes rather than repeatedly polling. If a previous invocation ended or the session was interrupted, use `continue RUN_ID` to reattach to the persisted run. Do not submit the same training request again just because the calling session ended; first check the existing run ID. Use `cancel RUN_ID` only when cancellation is requested. Use `retrain --request REQUEST.json` only when a fresh training run is explicitly wanted. `doctor` checks the configured setup. To use an already-finished successful training, `adopt TRAINING_ID` can record its provenance; add `--video video.json` only when the user also requested generation from it.

Do not read implementation code or remote logs for routine progress. Use the CLI's concise status and error information. Do not put secrets in request files or print tokens. The FreeVideo agent token remains on the Windows host at the configured token file; SSH is transport only. Training and generation go through the FreeVideo agent API. Adapters and generated videos remain on Windows; do not download or transfer them unless the user explicitly asks.

The `h3-person` profile has been tested at 50 epochs, rank 8, and 0.25 megapixels. Treat those values as a tested setup, not a compatibility guarantee. For image selection, consult the [photo guide](references/photo-guide.md).

If `doctor` reports absent or incompatible training routes after an update, read [maintenance](references/maintenance.md). Use `transport: local` when running on the Windows FreeVideo PC, or `transport: ssh` from a connected machine. Run records belong to the CLI installation that created them; use its returned IDs and the same configuration.

An active submission starts a background driver automatically. It handles the training-to-video handoff without LLM polling; keep the driver computer awake and connected. Record the run ID and report the submission. Check status on user request or when verifying completion; do not maintain a rapid agent polling loop. `continue` restarts a stopped driver while preserving remote job IDs. A driver error after repeated connectivity failures needs a reconnect, not a new training request. Changing only the requested scene reuses the prior training.
