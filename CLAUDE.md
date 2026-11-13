# CLAUDE.md

Simulation code from a BSc dissertation comparing three arbitration controllers
(continuous blend, utility switch, PAG-inspired FSM) in a Gymnasium environment.

## Commands
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python scripts/verify_fsm.py                        # FSM and delay-buffer checks
python -m pag_arbitration.experiments.evaluate      # full factorial sweep (~15 s on 12 cores)

ruff check . && ruff format --check .               # lint and format
mypy src                                            # type check
pytest -q                                           # tests (suite added in Phase 4)
```

## Conventions
- British English in docs, comments and commit messages.
- Conventional Commits (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`, `ci:`).
- Simulation behaviour must not change: per-seed results must reproduce the
  dissertation exactly. Compare outputs against a baseline run before merging
  any change to `environment.py`, `controller.py` or the episode loop.
- Never commit generated CSVs or figures; run experiments from a scratch directory
  (or `results/`, which is gitignored).
