---
description: Python environment setup and execution rules
---

# Python Environment Rules

1. The default Python environment for this project is a conda environment named `vllm-env`.
2. When executing Python scripts or commands in the terminal, ALWAYS use `conda run -n vllm-env python` or activate the environment first. Do not use the system `python` command directly unless the environment is already activated.
3. If you encounter a `ModuleNotFoundError` for packages like `stable_retro` or `stable_baselines3`, it means you forgot to use the `vllm-env` conda environment.
