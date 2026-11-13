"""Render the README demo GIF: BLEND, SWITCH and PAG side by side on one seed.

Uses the environment's own pygame renderer with a headless display, so the GIF
is reproducible. Requires Pillow (installed with matplotlib).

    python scripts/make_demo_gif.py --seed 1046 --steps 700 --out docs/assets/demo.gif
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame
from PIL import Image, ImageDraw, ImageFont

from pag_arbitration.environment import PAGArbitrationEnv
from pag_arbitration.simulation import CONTROLLER_NAMES, make_controller

PANEL = 260  # pixels per panel in the GIF
LABEL_H = 30
FONT = ImageFont.load_default(size=17)


def render_frames(controller: str, seed: int, steps: int, every: int) -> list[Image.Image]:
    """Run one episode, capturing every ``every``-th frame (frozen after death)."""
    env = PAGArbitrationEnv(config={"threat_level": "medium", "render_mode": "human"})
    agent = make_controller(controller)
    obs, _ = env.reset(seed=seed)
    frames: list[Image.Image] = []
    mode, alive, survived = "", True, 0
    for t in range(steps):
        if alive:
            obs, _r, terminated, _trunc, _info = env.step(agent.act(obs))
            mode, survived = agent.mode or "", t + 1
            alive = not terminated
        if t % every == 0:
            _draw(env)
            arr = np.transpose(pygame.surfarray.array3d(env.window), (1, 0, 2))
            img = Image.fromarray(arr).resize((PANEL, PANEL), Image.Resampling.LANCZOS)
            frames.append(_label(img, controller, mode, survived, alive))
    env.close()
    return frames


def _draw(env: PAGArbitrationEnv) -> None:
    """Like env.render() with larger markers (readable when scaled down), no clock tick."""
    canvas = pygame.Surface((env.window_size, env.window_size))
    canvas.fill((255, 255, 255))
    for pos, colour, radius in (
        (env.food_pos, (0, 170, 0), 14),
        (env.predator_pos, (220, 0, 0), 18),
        (env.agent_pos, (0, 0, 230), 18),
    ):
        pygame.draw.circle(
            canvas, colour, (int(pos[0] * env.scale), int(pos[1] * env.scale)), radius
        )
    bar = int((max(0.0, float(env.energy)) / 100) * 150)
    pygame.draw.rect(canvas, (0, 0, 0), (10, 10, 150, 20), 2)
    pygame.draw.rect(canvas, (0, 200, 0), (10, 10, bar, 20))
    env.window.blit(canvas, (0, 0))


def _label(img: Image.Image, controller: str, mode: str, steps: int, alive: bool) -> Image.Image:
    out = Image.new("RGB", (PANEL, PANEL + LABEL_H), "white")
    out.paste(img, (0, LABEL_H))
    status = mode if alive else f"died at step {steps}"
    ImageDraw.Draw(out).text((8, 5), f"{controller}: {status}", fill=(20, 20, 20), font=FONT)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1046)
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--every", type=int, default=3, help="Capture every n-th step.")
    parser.add_argument("--out", type=Path, default=Path("docs/assets/demo.gif"))
    args = parser.parse_args()

    panels = [render_frames(c, args.seed, args.steps, args.every) for c in CONTROLLER_NAMES]
    frames = []
    for row in zip(*panels, strict=True):
        frame = Image.new("RGB", (PANEL * 3 + 8, PANEL + LABEL_H), (200, 200, 200))
        for i, panel in enumerate(row):
            frame.paste(panel, (i * (PANEL + 4), 0))
        frames.append(frame.quantize(colors=32))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=40, loop=0)
    print(f"Saved {args.out} ({args.out.stat().st_size / 1e6:.1f} MB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
