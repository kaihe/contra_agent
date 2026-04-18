# Contra Agent — Codebase Rules

## Artifacts

All saved artifacts (images, videos, GIFs) must be written to the `tmp/` folder.

## Emulator

`stable_retro` only allows one emulator instance per process (`RuntimeError: Cannot create multiple emulator instances per process`). Always close an env with `env.close()` before creating another one. Never hold two envs open at the same time.
