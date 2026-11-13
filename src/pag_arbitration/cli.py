"""Command-line interface: ``pag-arbitration <command> [options]``.

Examples::

    pag-arbitration all --config smoke     # every experiment, 3 episodes per cell (< 1 min)
    pag-arbitration all                    # full dissertation run, 100 episodes per cell
    pag-arbitration factorial --episodes 20 --out-dir results/quick
    pag-arbitration demo --controller PAG --seed 42
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd

from pag_arbitration import analysis
from pag_arbitration.experiments import (
    ablation_delay,
    ablation_delay_noise,
    ablation_hysteresis,
    factorial,
    recovery_decomposition,
    sensitivity_beta,
)
from pag_arbitration.simulation import CONTROLLER_NAMES, SEED_BASE

ExperimentRunner = Callable[..., pd.DataFrame]

# name -> (runner, help text)
EXPERIMENTS: dict[str, tuple[ExperimentRunner, str]] = {
    "factorial": (factorial.run, "Controller x threat x noise x delay sweep (headline results)."),
    "ablation-hysteresis": (ablation_hysteresis.run, "PAG with and without its deadband."),
    "ablation-delay": (ablation_delay.run, "BLEND vs PAG across sensorimotor delays 0-5."),
    "ablation-delay-noise": (ablation_delay_noise.run, "Delay sweep at two noise levels."),
    "recovery": (recovery_decomposition.run, "Split deficits into true and terminal recoveries."),
    "sensitivity-beta": (sensitivity_beta.run, "BLEND across softmax temperatures."),
}

# Episodes per condition. "full" reproduces the dissertation; "smoke" is a quick
# end-to-end check of every code path that finishes in well under a minute.
CONFIG_EPISODES = {"full": 100, "smoke": 3}


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        choices=sorted(CONFIG_EPISODES),
        default="full",
        help="'full' reproduces the dissertation (100 episodes per condition); "
        "'smoke' runs 3 per condition as a quick check. Default: full.",
    )
    parser.add_argument("--episodes", type=int, help="Episodes per condition (overrides --config).")
    parser.add_argument(
        "--seed-base",
        type=int,
        default=SEED_BASE,
        help=f"Episode i of every condition uses seed SEED_BASE + i. Default: {SEED_BASE}.",
    )
    parser.add_argument(
        "--workers", type=int, help="Worker processes. Default: CPU count minus one."
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("results"), help="Output directory. Default: results/."
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="pag-arbitration",
        description="Reproduce the PAG vs blending arbitration experiments.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for name, (_runner, help_text) in EXPERIMENTS.items():
        _add_run_options(sub.add_parser(name, help=help_text))

    p_all = sub.add_parser("all", help="Run every experiment, then the analysis report.")
    _add_run_options(p_all)

    p_an = sub.add_parser(
        "analyse", help="Build summary tables and extra figures from existing CSVs."
    )
    p_an.add_argument(
        "--out-dir", type=Path, default=Path("results"), help="Directory holding the CSVs."
    )

    p_demo = sub.add_parser("demo", help="Watch a controller in a pygame window.")
    p_demo.add_argument("--controller", choices=CONTROLLER_NAMES, default="PAG")
    p_demo.add_argument("--seed", type=int, default=42)
    p_demo.add_argument("--episodes", type=int, default=1)
    p_demo.add_argument("--threat", choices=["low", "medium", "high"], default="medium")
    p_demo.add_argument(
        "--no-render", action="store_true", help="Run headless and print metrics only."
    )
    return parser


def _run_experiments(names: Sequence[str], args: argparse.Namespace) -> None:
    episodes = args.episodes or CONFIG_EPISODES[args.config]
    if episodes < 1:
        raise SystemExit("--episodes must be at least 1.")
    print(f"Writing results to {args.out_dir}/ ({episodes} episodes per condition)")
    for name in names:
        start = time.perf_counter()
        runner, _help = EXPERIMENTS[name]
        runner(args.out_dir, episodes=episodes, seed_base=args.seed_base, workers=args.workers)
        print(f"[{name}] finished in {time.perf_counter() - start:.1f} s\n")


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the ``pag-arbitration`` command."""
    args = build_parser().parse_args(argv)

    if args.command == "demo":
        from pag_arbitration.demo import run_demo

        df = run_demo(
            controller=args.controller,
            seed=args.seed,
            episodes=args.episodes,
            render=not args.no_render,
            threat=args.threat,
        )
        print(df.to_string(index=False))
    elif args.command == "analyse":
        analysis.run(args.out_dir)
    elif args.command == "all":
        _run_experiments(list(EXPERIMENTS), args)
        analysis.run(args.out_dir)
    else:
        _run_experiments([args.command], args)


if __name__ == "__main__":
    main()
