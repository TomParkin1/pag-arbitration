# CLAUDE.md

Simulation code from a BSc dissertation comparing three arbitration controllers
(continuous blend, utility switch, PAG-inspired FSM) in a Gymnasium environment.

## Commands
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m pag_arbitration.experiments.factorial     # full factorial sweep -> results/

ruff check . && ruff format --check .               # lint and format
mypy                                                # strict type check (src, tests)
pytest                                              # fast, deterministic tests
```

## Conventions
- British English in docs, comments and commit messages.
- Conventional Commits (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`, `ci:`).
- Simulation behaviour must not change: per-seed results must reproduce the
  dissertation exactly. Compare outputs against a baseline run before merging
  any change to `environment.py`, `controller.py` or the episode loop.
- Experiments write to `results/` (gitignored); never commit generated CSVs.

## Layout
- `environment.py` (Gymnasium env), `controller.py` (Blend/Switch/PAG),
  `simulation.py` (episode runner, parallel sweep), `stats.py`, `plotting.py`.
- `experiments/`: one module per dissertation experiment, each with `run(out_dir, ...)`.
