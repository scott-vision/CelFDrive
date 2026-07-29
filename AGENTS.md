# Engineering guidance

Use this repository as a research-software project intended for reuse. Make changes that are understandable, testable, and maintainable by a scientist or developer who did not write the original code.

## Working principles

- Prefer the smallest coherent change that solves the stated problem. Do not refactor adjacent code merely for stylistic uniformity.
- Read the relevant call sites, configuration, and documentation before changing a public API or workflow.
- Preserve domain terms and existing conventions unless a change is intentional and documented. Do not invent abstractions, configuration options, or fallback behaviour without a demonstrated need.
- Give functions precise names, narrow responsibilities, explicit inputs/outputs, and useful docstrings for public or non-obvious behaviour. Avoid long positional-argument interfaces.
- Validate assumptions at boundaries (image dimensionality, units, configuration values, paths, model outputs) and fail with actionable error messages. Do not silently guess or swallow errors.
- Keep configuration defaults, README examples, and runtime behaviour aligned. Document units, coordinate conventions, image-array expectations, and hardware-specific assumptions.
- Add or update focused tests for every bug fix and meaningful behavior change. Test normal cases, boundary cases, and the documented error condition.
- Use comments to explain domain rationale or non-obvious constraints, not to narrate the code line by line. Delete stale comments when changing code.
- Keep changes reviewable: avoid large formatting-only diffs, speculative compatibility layers, blanket exception handling, and unrelated dependency churn.
- Before handing off a change, run the relevant checks and state what was verified and what could not be verified.

## Common low-quality automated-coding patterns to avoid

- Generic names (`data`, `result`, `helper`, `manager`) where a domain-specific name would clarify intent.
- Over-engineered wrappers, factories, inheritance, or new files for a one-use operation.
- Broad `try/except Exception` blocks, silent fallbacks, or returning placeholder values that conceal a failure.
- Repeating the same validation, conversion, or configuration logic in several call sites instead of placing it at a clear boundary.
- Comments/docstrings that restate syntax, contain unverified claims, or drift from the implementation.
- Adding many knobs without validation, units, clear defaults, or a concrete user need.
- Large, uniform rewrites that erase useful local conventions and make scientific review difficult.
- Cosmetic type hints, logging, or error handling that give an appearance of rigor without supporting a real workflow.
- Magic defaults chosen without provenance, especially for thresholds, pixel sizes, coordinate units, or hardware behaviour.
- Tests that only check implementation details, happy paths, or mocked behaviour rather than user-visible outcomes.
