"""vcgame — entry point."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from src.display import run_display_demo


def _parse_vec_arg(s: str) -> np.ndarray:
    """Parse a comma-separated string like '1.0,2.0,3.0' into a float array."""
    return np.array([float(x) for x in s.split(",")], dtype=float)


def _fix_negative_args() -> None:
    """Join --pos/--heading with their value using '=' so argparse doesn't
    mistake a leading '-' in the value for a flag.
    """
    _vec_flags = {"--pos", "--heading"}
    argv = sys.argv
    fixed = [argv[0]]
    i = 1
    while i < len(argv):
        if argv[i] in _vec_flags and i + 1 < len(argv):
            fixed.append(f"{argv[i]}={argv[i + 1]}")
            i += 2
        else:
            fixed.append(argv[i])
            i += 1
    sys.argv = fixed


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
        choices=["cube", "trunc_oct", "random", "reflexive"],
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
        "--polytope",
        type=int,
        default=0,
        metavar="ID",
        help="Reflexive polytope id 0–4318 for --shape reflexive (default: 0).",
    )
    p.add_argument(
        "-d",
        action="store_true",
        dest="deletion",
        help="Enable deletion mode at startup (default: off).",
    )
    p.add_argument(
        "--pos",
        type=str,
        default=None,
        metavar="x,y,z",
        help="Initial player position as 'x,y,z' (direction, will be normalised).",
    )
    p.add_argument(
        "--heading",
        type=str,
        default=None,
        metavar="x,y,z",
        help="Initial player heading as 'x,y,z'.",
    )
    p.add_argument(
        "--color",
        type=int,
        default=0,
        metavar="N",
        help="Initial color mode: 0=sun, 1=radius, 2=wireframe (default: 0).",
    )
    p.add_argument(
        "--flashlight",
        action="store_true",
        help="Start with flashlight on.",
    )
    return p.parse_args()


def main() -> None:
    _fix_negative_args()
    args = _parse_args()
    if args.shape == "trunc_oct":
        from src.generate_trunc_oct import trunc_oct_fan, trunc_oct_vc
        fan = trunc_oct_fan()
        vc  = trunc_oct_vc()
    elif args.shape == "random":
        from src.generate_random import random_fan, random_vc
        fan = random_fan(seed=args.seed)
        vc  = random_vc(seed=args.seed)
    elif args.shape == "reflexive":
        from src.generate_reflexive import reflexive_fan, reflexive_vc
        fan = reflexive_fan(polytope_id=args.polytope)
        vc  = reflexive_vc(polytope_id=args.polytope)
    else:
        from src.generate_cube import cube_fan, cube_vc
        fan = cube_fan(3)
        vc  = cube_vc(3)

    initial_pos       = _parse_vec_arg(args.pos)     if args.pos     else None
    initial_heading   = _parse_vec_arg(args.heading) if args.heading else None

    agent = None
    if args.levy:
        from agents.random_agent import RandomAgent
        from src.player import Player
        pos0 = initial_pos   if initial_pos   is not None else [1.0, 0.2, 0.1]
        hdg0 = initial_heading if initial_heading is not None else [0.0, 1.0, 0.0]
        player = Player(pos0, hdg0)
        agent  = RandomAgent(player, alpha=args.alpha, step=args.step)

    run_display_demo(
        fan, vc,
        agent=agent,
        allow_deletion=args.deletion,
        initial_pos=initial_pos,
        initial_heading=initial_heading,
        initial_color=args.color,
        initial_flashlight=args.flashlight,
    )


if __name__ == "__main__":
    main()
