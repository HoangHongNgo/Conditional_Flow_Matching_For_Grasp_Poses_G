# AGENTS.md

## Repository Context

- This is a Python 3.10 / CUDA 12+ deep learning repository for EconomicGrasp.
- Core entry points include `train.py`, `test.py`, `inference.py`, `train.sh`, `eval.sh`, and the `flow/` CFM pipeline.
- On the `cfm` branch, the primary working area is the `flow/` directory. Prefer inspecting and editing `flow/` first unless the user explicitly asks to work elsewhere.
- Large local assets such as datasets, checkpoints, generated results, caches, and compiled extensions should be treated as environment state unless the user explicitly asks to change them.

## Git Workflow

- Start every code-changing task by checking `git status --short`.
- Assume existing uncommitted changes belong to the user. Do not overwrite, revert, reformat, stage, or commit unrelated changes.
- Work on the current branch unless the user explicitly asks to create or switch branches.
- Never run destructive Git commands such as `git reset --hard`, `git clean`, `git checkout --`, `git restore`, or force-push unless the user explicitly requests that exact action.
- For this project, when there are code changes to save, prefer an automatic `git add` -> `git commit` -> `git push hoanghongngo cfm` flow unless the user explicitly asks for a different Git target or workflow.
- Keep diffs focused on the requested task. Avoid broad formatting, dependency churn, or generated-file updates unless needed for the task.
- Before finalizing, inspect the diff for accidental changes and report files changed plus verification performed.

## Branch And Commit Conventions

- Use short branch names when asked to create one, such as `fix/<topic>`, `feat/<topic>`, `exp/<topic>`, or `docs/<topic>`.
- Use concise commit subjects in imperative mood when asked to commit.
- Prefer one logical change per commit. If a task produces unrelated edits, ask before grouping them.

## Verification

- Prefer the narrowest verification that covers the change.
- Whenever running Python scripts, training loops, or tests in this project, always use the project virtual environment by invoking the Python binary directly:

  ```bash
  ./py310/bin/python <script_name.py> [args...]
  ```

- Do not use plain `python`, system Python, or a separately activated environment for project scripts unless the user explicitly asks.
- For Python-only edits, at minimum run syntax checks on touched Python files when full tests are not practical:

  ```bash
  ./py310/bin/python -m py_compile path/to/file.py
  ```

- For model, dataset, training, or inference changes, use a focused smoke command where possible before full GPU runs.
- Full training and GraspNet evaluation are expensive and dataset-dependent; do not start long GPU jobs unless the user asks or the task clearly requires them.
- If verification cannot run because dependencies, CUDA, checkpoints, or datasets are unavailable, state the exact blocker.

## Review Workflow

- When asked for a review, prioritize bugs, regressions, data corruption risks, numerical issues, shape/device mismatches, and missing tests.
- Present findings first with file and line references.
- Do not edit files during a review unless the user asks for fixes.

## Codex Operating Rules

- Prefer `rg` and `rg --files` for searching.
- Use existing project patterns before introducing new abstractions.
- Add comments only when they clarify non-obvious behavior.
- If the user's request is unclear, ask specific follow-up questions instead of guessing the intent, context, or expected outcome.
- Write an English docstring for every Python function that is added or modified.
- Write all source-code comments and annotations in English.
- For key tensors and variables, proactively annotate shapes in nearby comments, for example `# [B, 1024, 3]` or `# Shape: list of length B of tensors with shape [N, 3]`.
- Treat files under `scratch/` as exploratory unless the user asks to productionize them.
- Avoid editing vendored or third-party code under `libs/`, `MinkowskiEngine/`, or `codex-desktop-linux/` unless the task is specifically about those directories.
