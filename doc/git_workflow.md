# Git Workflow For This Repository

This repository has large datasets, checkpoints, compiled extensions, generated outputs, and active research experiments. Use a conservative Git workflow that protects local state and keeps diffs reviewable.

## Daily Flow

1. Inspect the tree before starting:

   ```bash
   git status --short
   ```

2. Create a focused branch when starting new work:

   ```bash
   git checkout main
   git pull
   git checkout -b feat/<topic>
   ```

3. Keep each task scoped to one logical change.

4. Review before staging:

   ```bash
   git diff
   git status --short
   ```

5. Stage only relevant files:

   ```bash
   git add path/to/file.py doc/git_workflow.md
   ```

6. Commit with a concise imperative subject:

   ```bash
   git commit -m "Add focused grasp refinement smoke test"
   ```

7. Push only when ready to share:

   ```bash
   git push -u origin feat/<topic>
   ```

## Codex Prompts

Use implementation prompts that force the agent to preserve local work:

```text
Check git status first. Implement the smallest correct change for <task>.
Do not modify unrelated files. Do not commit. Run the narrowest relevant
verification and summarize changed files.
```

Use review prompts when you only want feedback:

```text
Review the uncommitted changes like a PR. List bugs and risks first with
file and line references. Do not edit files.
```

Use commit-prep prompts when you want help staging:

```text
Inspect the diff, identify which files belong to this task, and propose a
commit message. Do not stage or commit until I confirm.
```

## Protected Actions

Do not run these unless explicitly requested:

```bash
git reset --hard
git clean
git checkout --
git restore
git push --force
```

These commands can destroy uncommitted experiments, generated labels, or local environment files.

## Verification Policy

Use the smallest check that proves the change:

- Documentation-only changes: inspect the rendered Markdown or review the diff.
- Python syntax-only changes: `python -m py_compile <touched files>`.
- Dataset or model pipeline changes: run a small smoke test before expensive training.
- Training or evaluation changes: confirm dataset, checkpoint, CUDA device, and expected runtime before launching long jobs.

Full training and GraspNet evaluation depend on local datasets and GPU availability, so they should be intentional rather than automatic.

## Files To Treat Carefully

- `checkpoints/`, `results/`, logs, and generated caches are local artifacts.
- `MinkowskiEngine/`, `libs/`, and compiled operator directories may contain third-party or build output.
- `scratch/` is for experiments; move stable scripts into the main tree only after review.
- `codex-desktop-linux/` appears to be unrelated local tooling and should not be edited for EconomicGrasp tasks.
