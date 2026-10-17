# FreeVideo agent workflows

For person LoRA training, read `agent/lora-pipeline/skills/freevideo-lora/SKILL.md`.
Use that pipeline's CLI and persistent run IDs. For character replacement, read
`agent/replacement-pipeline/AGENTS.md` and its README, then use its CLI.

Use only the authenticated FreeVideo agent API for training and generation.
Do not drive the ComfyUI UI or submit replacement workflows directly to it.
Keep credentials, reference photos, source videos, datasets, trained adapters,
model weights and run state out of Git. Use the example configurations; machine
configuration and agent tokens stay on the Windows host.

Do not regenerate or retrain after a connection interruption. Reconnect using
saved IDs; replacement resume is explicit and verifies completed stage receipts.
A passing readiness check does not prove AI quality or VRAM feasibility on new
footage. The H3 identity LoRA cannot be loaded into Wan; replacement uses a
reference picture and needs a source clip and target-person first-frame box.

Run the affected pipeline's tests when changing its code. Preserve the pinned
trainer source notices. Deployment/setup are explicit maintenance operations,
never routine steps in an ordinary training or replacement request.
