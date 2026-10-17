# Reusable Windows training and character replacement

This fork adds two FreeVideo agent workflows:

- [LoRA training](../agent/lora-pipeline/README.md): captioned photos or a ZIP,
  persistent training jobs and an optional requested video. The pipeline includes
  a reusable skill and a photo-angle guide.
- [Character replacement](../agent/replacement-pipeline/README.md): a short source
  video, character reference picture and normalized first-frame person box.
  Wan2.2-Animate via WanGP generates the replacement; SAM 2 masks and the
  compositor preserve decoded source RGB outside the saved edit region in the
  lossless master. The MP4 preview is lossy.

The server modules are integrated into `freevideo_engine/agent/routes.py`. Runtime
setup remains explicit. Copy the pipeline directories to the Windows engine's
`agent/` directory, or point the installers to the active source checkout. Do not
install dependencies or download model weights for each ordinary job.

For LoRA setup, copy `lora-pipeline/config.example.json` to `config.json` and set
real connection paths. `lora-pipeline/extension/backend.example.json` is a template
for `ENGINE_ROOT/agent/training/backend.json`; replace every `WINDOWS_USER` path.
Extract the bundled pinned trainer source into `ENGINE_ROOT/agent/training/Fizgig`
before the explicit training setup script. Existing configured hosts should
reuse their installed backend and environment. Consult the maintenance skill
before changing an installation.

Configure replacement using its Windows setup script after the training runtime
is available. Its environment reads the tested trainer torch stack without
modifying it. Require the agent doctor to report ready before submission.

## Validation record

The deployed Windows implementation passed 34 replacement tests, including native
Windows worker-crash descendant cleanup, concurrent state writes, stage checksum
recovery, download range/truncation checks, idempotent client replay, FFV1 pixel
preservation, copied audio and odd-dimension preview decoding. Its API 0.2 reported
ready, and the separate LoRA service remained ready. This fork reapplies those
modules to the public upstream main route layout using the compatibility-checked
integration anchors. Public source integration is not an additional GPU benchmark.

Publication checks passed 29 LoRA orchestration/installer tests, 16 backend tests,
and 34 replacement tests (the native Windows cleanup test is skipped on macOS).
The deployed Windows replacement suite passed all 34 tests. The standalone
disk-mapping probe needs torch and is run separately, not through unittest discovery.
A Linux/Windows GitHub Actions workflow runs the portable suites.
Real AI character replacement quality and VRAM use on new footage still require
an actual user-selected clip and character image. No private media, live run
records, machine diagnostics or credentials are included in this fork.

The bundled Fizgig archive is pinned to commit
`eb7492cd582bcb6de0094aa7b89c65c2ea5b4024` from
[shootthesound/Fizgig](https://github.com/shootthesound/Fizgig), with tested Windows
disk-streaming changes. Its LICENSE, NOTICE and third-party notices remain in the
archive. WanGP, SAM 2 and downloaded models retain their upstream licenses; see
replacement provenance and the respective upstream projects.
