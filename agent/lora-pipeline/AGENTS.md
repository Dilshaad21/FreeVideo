# Using the FreeVideo LoRA pipeline

For a person LoRA training or generated video request, use the project-local skill at [`skills/freevideo-lora/SKILL.md`](skills/freevideo-lora/SKILL.md) and the `freevideo_lora.py` CLI. Give the CLI a request file and use its returned run ID to check or continue persisted work. Do not inspect source code or remote training logs for routine use.

Use only the FreeVideo agent API for training and generation. SSH transports requests to the configured Windows host; its agent token remains in the Windows token file. Do not copy tokens into requests, config, messages, or output. Adapters and generated videos stay on Windows unless the user explicitly requests a separate transfer.
