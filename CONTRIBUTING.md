# Contributing

## Scope

This repository tracks the final ARC-Fuse codebase. Changes should preserve
that scope and avoid reintroducing historical experiment trees, generated
artifacts, or large result dumps unless they are required for reproducibility
and explicitly documented.

## Before You Open A PR

- Keep changes focused and reviewable.
- Update `README.md` and `docs/SOURCE_MAP.md` when you move or rename files.
- Do not commit API keys, secrets, dataset dumps, or local output directories.
- Prefer the existing `src/arc_fuse/` and `research_backend/arc_fuse_digimon/`
  layout for new code.

## Local Checks

Run the simplest relevant checks before pushing:

```bash
python -m pip install -e .
python run_demo.py --help
python research_backend/evaluate.py --help
```

If you modify the paper backend, also verify the smoke path:

```bash
bash research_backend/scripts/run_real_smoke.sh
```

## Style

- Keep files ASCII unless a file already uses non-ASCII text.
- Match the surrounding formatting and naming conventions.
- Prefer clear module boundaries over one-off scripts.

## Pull Requests

- Summarize what changed and why.
- Mention any file moves or deletions.
- Call out anything that affects reproducibility or historical outputs.
