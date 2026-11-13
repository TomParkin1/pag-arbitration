"""Experiments that regenerate the dissertation's results.

Each module exposes ``run(out_dir, episodes, seed_base, workers)``, which runs
the sweep, writes its CSV and figures into ``out_dir`` and returns the raw
per-episode DataFrame. Rows are written in grid order, so output files are
identical across runs.
"""
