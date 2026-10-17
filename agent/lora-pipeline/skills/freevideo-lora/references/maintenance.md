# Maintenance on the Windows host

Read this only when `doctor` reports missing training routes, an unsupported API, or missing prerequisites. A FreeVideo update can switch the active source directory without carrying this custom extension forward.

The persistent pipeline lives at `FreeVideo-engine/agent/lora-pipeline`. Training datasets, checkpoints and the isolated trainer remain at `FreeVideo-engine/agent/training` across source updates. Do not rebuild dependencies or redownload models for an ordinary source update.

Identify the exact active FreeVideo source directory. Check that integration first, on Windows:

```powershell
python install_windows.py --root ENGINE_ROOT --source ACTIVE_SOURCE --check
```

If compatible, reapply and restart with:

```powershell
python install_windows.py --root ENGINE_ROOT --source ACTIVE_SOURCE --restart
```

Use a FreeVideo Python environment that has psutil installed. The installer refuses active generation or training, validates every integration anchor, backs up replaced modules, and verifies API version 1.1 after restart. Unknown or partial integrations require a code review of the new routes; do not force a patch or replace the whole host routes file. Backups are under `agent/training/backups`.

`extension/setup_training.py` is an explicit environment/model setup operation for a new or broken trainer, not a step for each run. The bundled `vendor/fizgig-eb7492cd.tar.gz` contains the pinned trainer source and tested Windows disk-streaming changes, including upstream LICENSE and NOTICE. The installed `agent/training/backend.json` is the authoritative machine-specific configuration. Keep the text encoder disk streaming enabled on this 16 GB RAM machine. The adapter family is MiniMax H3; do not load it into a different model family without a separate compatibility implementation.

Checkpoint files exist, but automatic optimizer resume is not implemented. An interrupted training job is reported rather than silently retrained. Use an explicit new training request only when requested; `continue` reconnects to pipeline orchestration and live jobs.

A cold server start can outlast the initial readiness check. If the startup log is still advancing, wait and run `doctor` again instead of restarting the server repeatedly.
