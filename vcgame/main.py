"""vcgame — entry point."""

from __future__ import annotations

import argparse

from src.display import run_display_demo


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="vcgame",
        description="Navigate a simplicial fan on S².",
    )
    p.add_argument(
        "--levy",
        action="store_true",
        help="Let a Lévy-walk agent navigate.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=1.5,
        metavar="α",
        help="Lévy exponent for the agent (default: 1.5).",
    )
    p.add_argument(
        "--step",
        type=float,
        default=0.04,
        metavar="s",
        help="Agent step size in radians (default: 0.04).",
    )
    p.add_argument(
        "--shape",
        choices=["cube", "trunc_oct", "random"],
        default="cube",
        help="Vector configuration shape (default: cube).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=1102,
        metavar="N",
        help="RNG seed for --shape random (default: 1102).",
    )
    p.add_argument(
        "-d",
        action="store_true",
        dest="deletion",
        help="Enable deletion mode at startup (default: off).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.shape == "trunc_oct":
        from src.generate_trunc_oct import trunc_oct_fan, trunc_oct_vc
        fan = trunc_oct_fan()
        vc  = trunc_oct_vc()
    elif args.shape == "random":
        from src.generate_random import random_fan, random_vc
        fan = random_fan(seed=args.seed)
        vc  = random_vc(seed=args.seed)
    else:
        from src.generate_cube import cube_fan, cube_vc
        fan = cube_fan(3)
        vc  = cube_vc(3)
    agent = None
    if args.levy:
        from agents.random_agent import RandomAgent
        from src.player import Player
        player = Player([1.0, 0.2, 0.1], [0.0, 1.0, 0.0])
        agent  = RandomAgent(player, alpha=args.alpha, step=args.step)
    run_display_demo(fan, vc, agent=agent, allow_deletion=args.deletion)


if __name__ == "__main__":
    main()
