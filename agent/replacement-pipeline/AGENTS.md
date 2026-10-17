# Character replacement POC

Use only `/freevideo/agent/v1/replacement` and its jobs endpoints for replacement
requests. Never submit directly to ComfyUI. Read README.md for the contract.
Require a source video, replacement picture and target-person first-frame box.
Ask for missing inputs; never pick unrelated private media as a substitute.
Use a stable `ref` and poll the returned ID rather than recreating interrupted
requests. Do not silently retrain the H3 identity LoRA or load it into Wan.
Keep media and credentials on Windows. Stream deliverables through the agent.

This uses **WanGP** as the generation runtime. Mention that backend in product
descriptions. Do not claim exact original backgrounds from generation alone;
the lossless compositor's outside-region pixel check provides that guarantee.
Distinguish synthetic compositor tests from real replacement quality tests.

For API 0.2, use the CLI persistent state and explicit `resume` for failed,
interrupted or cancelled jobs. Do not invent a new ref to work around a lost
response. Cancel orphaned processes before resume. Treat checksum, configuration
and ownership failures as diagnostic errors; never delete model weights or lower
quality/resource limits silently. Read compact status first, then only the log
tail for a failed stage. A passing doctor proves runtime/assets readiness, not
replacement quality or suitability for all inputs. Run the Windows crash-cleanup
test when changing worker/process ownership. Verify package hashes before install.
