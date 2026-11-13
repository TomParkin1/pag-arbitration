"""End-to-end: the CLI runs an experiment and the analysis into a directory."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pag_arbitration.cli import main


def test_factorial_and_analysis_end_to_end(tmp_path: Path) -> None:
    main(["factorial", "--episodes", "1", "--workers", "1", "--out-dir", str(tmp_path)])
    df = pd.read_csv(tmp_path / "factorial_master_data.csv")
    assert len(df) == 3 * 2 * 2 * 3  # threat x noise x delay x controller
    assert (tmp_path / "factorial_results.png").exists()

    main(["analyse", "--out-dir", str(tmp_path)])
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "Survival steps" in summary


def test_demo_headless(capsys: pytest.CaptureFixture[str]) -> None:
    main(["demo", "--no-render", "--controller", "Switch", "--seed", "1000"])
    assert "survival_steps" in capsys.readouterr().out


@pytest.mark.parametrize("bad", [["--episodes", "0"], ["--workers", "0"], ["--workers", "-2"]])
def test_rejects_invalid_run_options(tmp_path: Path, bad: list[str]) -> None:
    with pytest.raises(SystemExit, match="must be at least 1"):
        main(["ablation-hysteresis", *bad, "--out-dir", str(tmp_path)])


def test_analyse_fails_without_data(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="No experiment CSVs"):
        main(["analyse", "--out-dir", str(tmp_path / "empty")])
    assert not (tmp_path / "empty").exists()
