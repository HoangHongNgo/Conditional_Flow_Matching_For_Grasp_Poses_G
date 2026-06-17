---
trigger: model_decision
description: When need to create temporary scripts, debug code, run experiments, or store intermediate data files
---

# Scratch Directory Usage Rule
When you (the AI Agent) need to create temporary scripts, debug code, run experiments, or store intermediate data files, you **MUST** strictly follow these rules:
1. **Mandatory Location:** Always create and execute temporary files inside the `scratch/` directory located at the root of the project. If the directory does not exist, create it first.
2. **Zero Clutter:** NEVER create temporary files (e.g., `test.py`, `temp.json`, `debug_log.txt`) in the project root or intermingled with the main source code directories.
3. **Descriptive Naming:** Give meaningful names to your scratch files that reflect their purpose (e.g., `scratch/test_cfm_inference.py` instead of `scratch/temp.py`).
4. **Self-Contained:** Treat the `scratch/` directory as your isolated sandbox. Ensure that running scripts from inside `scratch/` does not unintentionally mutate or delete the main project's structural files.