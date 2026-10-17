# FreeVideo LoRA pipeline

This project lets an agent submit a person-specific LoRA training request to the FreeVideo installation on the configured Windows host, then check or continue that run and generate a video. The CLI owns remote orchestration and persisted run state, so agents can work from the request and run ID without reading implementation files or training logs.

## Use it from another agent

Install or copy this project directory into a location available to the target agent, then point that agent to `skills/freevideo-lora/SKILL.md` as a project-local skill. The target agent can read that file directly or register the directory as one of its local skills, depending on its skill-loading setup. Keep the project folder available for the duration of a run so the CLI can find its configuration and persisted state.

Start with `python freevideo_lora.py --help` from this directory. The default configuration path is `config.json`; copy `config.example.json` to that name and fill in the connection details for the Windows host. `remote_root` is optional and defaults to the example path. Pass `--config PATH` before the subcommand to use another configuration file. Never put the FreeVideo agent token in this project configuration; the token remains on the Windows host at the configured token file.

See [the request example](examples/request.json), [the request schema](request.schema.json), and [the photo guide](skills/freevideo-lora/references/photo-guide.md) when preparing a training request. Images need adjacent `.txt` captions; supported formats are JPG, PNG, and WebP. A missing caption returns `needs_input` so the same run can continue after supplying it.

## Operational boundaries

Training and video generation use the FreeVideo agent API on the Windows host. SSH is only the transport used by this pipeline, and the agent token stays on Windows. Trained adapters and generated videos remain on that host; the pipeline does not download them automatically.

The default training profile `h3-person` has been exercised with 50 epochs, rank 8, and a 0.25 megapixel setting. That is a tested configuration, not a claim of universal compatibility across machines, subjects, or FreeVideo versions.

## Updating the FreeVideo integration

Run `doctor` before a new run. If a FreeVideo update removes the training routes, follow [maintenance instructions](skills/freevideo-lora/references/maintenance.md). `install_windows.py` checks compatibility, refuses active GPU work, preserves backups, and can restart the idle server. The trainer environment and downloaded model files are reused.

On the Windows PC set `"transport": "local"` in `config.json` to call the loopback API directly. Agents on this Mac use `"transport": "ssh"`. Neither mode exposes the bearer token. State is local to the machine running the CLI; copy the run records when deliberately migrating orchestration to another machine, and keep the same connection configuration.

Run the portable CLI/installer tests with `python -m unittest discover -s tests -v`. Extension tests also require the actual FreeVideo source and its dependencies: set `FREEVIDEO_SOURCE` to that source before running `extension/test_training.py`.

## One submission, no LLM polling loop

`run`, `retrain`, `continue`, and `adopt --video` start or reuse a background driver when work remains. It polls the API in code and automatically submits the requested video after training succeeds. You can close the conversation; the computer running the driver must remain awake and connected. Status replies include the run ID and driver health. The driver stops on completion, cancellation, a training failure, or three consecutive connection errors; `continue RUN_ID` reconnects without duplicating training. It does not automatically restart a failed trainer.

Changing only the video prompt reuses the same dataset and training job and creates a separate video run. `retrain` explicitly requests a new training job. For an existing adapter use `adopt TRAINING_ID --video VIDEO.json`.
