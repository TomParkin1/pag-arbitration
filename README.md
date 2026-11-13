# pag-arbitration

Simulation study comparing PAG-inspired discrete mode switching with continuous
command blending for a foraging agent under predator threat.

BSc Computer Science dissertation project, University of Sheffield (2026).

## Quickstart

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .
pag-arbitration all --config smoke   # every experiment, 3 episodes per condition (~10 s)
pag-arbitration all                  # full dissertation run: 100 episodes per condition, seeds 1000-1099
pag-arbitration --help               # all commands and options
```

Outputs are written to `results/`. Full documentation to follow.
