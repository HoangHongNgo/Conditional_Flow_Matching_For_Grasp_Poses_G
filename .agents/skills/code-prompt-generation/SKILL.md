---
name: code-prompt-generation
description: Build a complete prompt for code generation after reading the user's request and the current workspace context. Use when Codex must inspect the repository, identify constraints, ask concise follow-up questions for unresolved ambiguity, and then deliver a high-signal prompt that another coding agent can execute.
---

# Code Prompt Generation

## Overview

Turn an underspecified coding request into an execution-ready prompt. Read both the user's request and the repository context before drafting anything final.

## Workflow

### 1. Read The Request Precisely

Extract the following before touching the repo:

- The requested outcome
- The expected form of the answer: code change, design, refactor, bug fix, test, review, migration, script, prompt only
- Any explicit constraints such as language, framework, performance, style, file scope, or "do not change" boundaries
- Any implied success criteria

Restate the task in one or two sentences for yourself. Do not invent requirements.

### 2. Inspect Workspace Context

Read only enough local context to ground the prompt. Prefer fast discovery over deep reading.

Default inspection sequence:

1. Read repo-local instruction files such as `AGENTS.md` when present.
2. Run `git status --short` to detect unrelated user changes and avoid overwriting them.
3. Inspect the most relevant files and directories with `rg`, `rg --files`, and targeted file reads.
4. Capture the stack, architecture, naming patterns, and existing conventions that the generated code should follow.

Summarize only the constraints that matter for the prompt, for example:

- Relevant directories and entry points
- Existing APIs or abstractions to reuse
- Validation commands to run
- Branch-specific or workspace-specific rules
- Files already modified by the user that should be left alone

If the workspace includes explicit operating rules, prefer them over generic best practices.

### 3. Identify Uncertainty

Separate uncertainty into two buckets:

- Blocking uncertainty: missing details that would materially change the solution
- Non-blocking uncertainty: details that can be handled with a reasonable assumption

Ask follow-up questions only for blocking uncertainty. Keep questions concise and decision-oriented.

Good follow-up questions:

- Ask for a choice when multiple implementations have different consequences.
- Ask for a missing target file or subsystem when the workspace does not make it discoverable.
- Ask for acceptance criteria when "done" is otherwise ambiguous.

Avoid unnecessary questions when:

- The answer is already implied by repo conventions or nearby code
- A safe default exists and can be stated explicitly as an assumption
- The uncertainty does not affect the generated prompt in a meaningful way

Use at most three short questions in one turn. If nothing important is unclear, skip questions.

### 4. Build The Final Prompt

Produce one prompt that another coding agent can execute without extra interpretation. The prompt must be specific, concise, and grounded in the inspected workspace.

Include these sections when relevant:

1. Objective
2. Workspace context
3. Constraints and non-goals
4. Relevant files or subsystems
5. Expected implementation approach
6. Verification requirements
7. Output expectations

Prefer concrete details over general advice. Mention exact paths, commands, and acceptance criteria when known.

### 5. Output Contract

If blocking questions remain, ask them first and wait.

If only non-blocking uncertainty remains, state the assumptions briefly and then provide the final prompt.

Return the final prompt in a fenced code block so the user can reuse it directly.

## Prompt Template

Use this structure and adapt it to the task:

```text
You are working in the repository at <workspace-path>.

Objective:
<What needs to be built, fixed, changed, or analyzed>

Workspace context:
- <Key repo facts learned from inspection>
- <Relevant architecture or conventions>
- <User changes or boundaries to respect>

Constraints:
- <Language/framework/version constraints>
- <Files or directories to prioritize or avoid>
- <Performance, style, or compatibility requirements>

Relevant files:
- <Path>: <Why it matters>
- <Path>: <Why it matters>

Implementation guidance:
- Reuse existing patterns from <path or subsystem>.
- Keep the diff focused on <scope>.
- Do not modify <path or area> unless required.

Verification:
- Run <command>
- If full validation is expensive, run <targeted smoke check>

Deliverable:
- Make the necessary code changes.
- Briefly summarize what changed and any assumptions.
- Report verification results and blockers, if any.
```

## Style Rules

- Prefer exact repository facts over generic coding instructions.
- Prefer short, high-signal prose over long background explanations.
- Make assumptions explicit instead of hiding them.
- Do not quote large chunks of local files into the prompt.
- Do not ask the follow-up questions and provide a "final" prompt in the same response if those questions are truly blocking.
