---
trigger: model_decision
description: When writing a command
---

Whenever running Python scripts, training loops, or tests in this project, **always use the project's specific virtual environment**.

Command to use:
Use the python binary directly from the `py310` directory:
```bash
./py310/bin/python <script_name.py> [args...]
```

Do not use plain `python`, system Python, or a separately activated environment for project scripts unless the user explicitly asks.
