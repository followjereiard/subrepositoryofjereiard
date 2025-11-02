# AGENTS.md

Problem definition → small, safe change → change review → refactor — repeat the loop.

## Project Specification

- Refer to README.md

## Mandatory Rules

- Before changing anything, read the relevant files end to end, including all call/reference paths.
- Keep tasks, commits, and PRs small.
- If you make assumptions, record them in the Issue/PR/ADR.
- Never commit or log secrets; validate all inputs and encode/normalize outputs.
- Avoid premature abstraction and use intention-revealing names.
- Compare at least two options before deciding.

## Mindset

- Think like a senior engineer.
- Don’t jump in on guesses or rush to conclusions.
- Always evaluate multiple approaches; write one line each for pros/cons/risks, then choose the simplest solution.

## Code & File Reference Rules

- Read files thoroughly from start to finish (no partial reads).
- Before changing code, locate and read definitions, references, call sites, related tests, docs/config/flags.
- Do not change code without having read the entire file.
- Before modifying a symbol, run a global search to understand pre/postconditions and leave a 1–3 line impact note.

## Required Coding Rules

- Before coding, write a Problem 1-Pager: Context / Problem / Goal / Non-Goals / Constraints.
- Enforce limits: file ≤ 300 LOC, function ≤ 50 LOC, parameters ≤ 5, cyclomatic complexity ≤ 10. If exceeded, split/refactor.
- Prefer explicit code; no hidden “magic.”
- Follow DRY, but avoid premature abstraction.
- Isolate side effects (I/O, network, global state) at the boundary layer.
- Catch only specific exceptions and present clear user-facing messages.
- Use structured logging and do not log sensitive data (propagate request/correlation IDs when possible).
- Account for time zones and DST.

## Testing Rules

- New code requires new tests; bug fixes must include a regression test (write it to fail first).
- Tests must be deterministic and independent; replace external systems with fakes/contract tests.
- Include ≥1 happy path and ≥1 failure path in e2e tests.
- Proactively assess risks from concurrency/locks/retries (duplication, deadlocks, etc.).

## Security Rules

- Never leave secrets in code/logs/tickets.
- Validate, normalize, and encode inputs; use parameterized operations.
- Apply the Principle of Least Privilege.

## Clean Code Rules

- Use intention-revealing names.
- Each function should do one thing.
- Keep side effects at the boundary.
- Prefer guard clauses first.
- Symbolize constants (no hardcoding).
- Structure code as Input → Process → Return.
- Report failures with specific errors/messages.
- Make tests serve as usage examples; include boundary and failure cases.

## Anti-Pattern Rules

- Don’t modify code without reading the whole context.
- Don’t expose secrets.
- Don’t ignore failures or warnings.
- Don’t introduce unjustified optimization or abstraction.
- Don’t overuse broad exceptions.

## Additional Consideration

- After writing or editing the code, verify thoroughly that every requirement supplied by the user has been satisfied.
- Implement every component; do not omit any feature or edge case.
- Aim for production-quality output rather than a proof of concept.
- Follow the idiomatic style guide of the chosen language (for example PEP 8 for Python or effective-go for Go).
- Include clear inline comments where the intention may not be obvious, and provide concise function-level docstrings or documentation blocks.
- Validate all external inputs and handle errors gracefully, returning informative messages instead of stack traces.
- Ensure thread-safety or asynchronous correctness if concurrency is involved.
- Optimise for maintainability first, then for performance; profile any non-trivial optimisations.
- Write a minimal yet meaningful automated test suite that covers critical paths and failure cases; include instructions on running these tests.
- Avoid hard-coding configuration values; expose them through environment variables or a configuration file.
- Adhere to secure-coding best practices: sanitise inputs, avoid secrets in source control, and use prepared statements for database access.
- Provide logging at appropriate levels (info, warning, error) with enough context to aid debugging in production.
- Supply a brief README explaining setup, deployment steps, dependencies, and usage examples.

Follow these principles unless the user explicitly states otherwise.