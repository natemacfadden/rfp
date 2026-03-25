"""
Game loop and debug utilities for the curses-based fan display.
"""

from __future__ import annotations

import curses
import math
import os
import sys
import time
import threading
from typing import TYPE_CHECKING

try:
    from pynput import keyboard as _pynput_kb
    _HAS_PYNPUT = True
except ImportError:
    _HAS_PYNPUT = False

import numpy as np

from .player import Player
from renderer.renderer import (
    Renderer,
    _cone_edge_map,
    _project,
    _orient_normal,
    _M3_HEIGHT,
    _M3_THETA_MAX,
    _HUD_ROWS,
    _ray_intersects_triangle,
)

if TYPE_CHECKING:
    from regfans import Fan
    from _curses import _CursesWindow


def _debug_dump(
    player,
    fan,
    stdscr: "_CursesWindow",
    view_scale: float,
    vectors: list | None = None,
    cli_cmd: str = "",
) -> list[str]:
    """Compute and return geometry debug info as a list of lines."""
    p      = np.asarray(player.direction, float)   # unit direction (geometry)
    p_cart = np.asarray(player.cartesian,  float)  # actual 3D position
    e1     = np.asarray(player.heading,    float)
    cone   = tuple(sorted(player.current_cone(fan)))

    rows, cols = stdscr.getmaxyx()
    cy, cx = rows // 2, cols // 2
    scale  = float(min(rows, cols // 2) // 2 - 2) * 0.75 * view_scale

    view_dir = p
    e1_new   = e1
    e2_new   = np.cross(p, e1)

    def ray(l: int) -> np.ndarray:
        return np.asarray(fan.vectors(which=(l,))[0], float)

    def scr_of(w: np.ndarray):
        coord = _project(w, view_dir, e1_new, e2_new)
        if coord is None:
            return None, None
        return (cy - int(round(coord[1] * scale)),
                cx + int(round(coord[0] * scale * 2)))

    # ---- front cones --------------------------------------------------------
    front: set = set()
    for ct_raw in fan.cones():
        ct = tuple(sorted(ct_raw))
        vv = [ray(l) for l in ct]
        n  = np.cross(vv[1] - vv[0], vv[2] - vv[0])
        nn = float(np.linalg.norm(n))
        if nn > 1e-12:
            n = n / nn
        n  = _orient_normal(n, vv[0])
        if float(np.dot(n, view_dir)) > 0:
            front.add(ct)

    # ---- current face -------------------------------------------------------
    curr_vv = [ray(l) for l in cone]
    curr_nf  = np.cross(curr_vv[1] - curr_vv[0], curr_vv[2] - curr_vv[0])
    curr_nn  = float(np.linalg.norm(curr_nf))
    if curr_nn > 1e-12:
        curr_nf = curr_nf / curr_nn
    curr_nf = _orient_normal(curr_nf, curr_vv[0])
    curr_c = (curr_vv[0] + curr_vv[1] + curr_vv[2]) / 3.0

    plane_d = float(np.dot(curr_c, curr_nf))
    denom   = float(np.dot(p, curr_nf))
    t_face  = plane_d / denom if abs(denom) > 1e-12 else float(np.linalg.norm(curr_c))
    p_src   = p * t_face + _M3_HEIGHT * curr_nf

    h_raw  = e1 - float(np.dot(e1, curr_nf)) * curr_nf
    h_norm = float(np.linalg.norm(h_raw))
    h_proj = h_raw / h_norm if h_norm > 1e-12 else e1

    cos_tmax = math.cos(math.radians(_M3_THETA_MAX))

    # 2D beam axis: straight up on screen (matches draw logic — heading direction)
    h_scr_y, h_scr_x, h_scr_len = 1.0, 0.0, 1.0

    # ---- spherical coords ---------------------------------------------------
    r_xy   = math.sqrt(float(p[0])**2 + float(p[1])**2)
    az_deg = math.degrees(math.atan2(float(p[1]), float(p[0])))
    el_deg = math.degrees(math.atan2(float(p[2]), r_xy))

    # ---- build m3_faces (same logic as draw) --------------------------------
    m3: dict = {}
    for ct in front:
        vv = [ray(l) for l in ct]
        c  = (vv[0] + vv[1] + vv[2]) / 3.0
        nf = np.cross(vv[1] - vv[0], vv[2] - vv[0])
        nn = float(np.linalg.norm(nf))
        if nn > 1e-12:
            nf = nf / nn
        nf = _orient_normal(nf, vv[0])
        m3[ct] = (vv, c, nf)

    # ---- per-face occlusion (mirrors draw logic) ----------------------------
    EPS = 1e-3
    occ: dict = {}
    for ct0, (vv0, c0, nf0) in m3.items():
        dv    = c0 - p_src
        dist0 = float(np.linalg.norm(dv))
        if dist0 < 1e-12:
            occ[ct0] = True
            continue
        dir0 = dv / dist0
        ok   = True
        for ct1, (vv1, c1, nf1) in m3.items():
            if ct1 == ct0:
                continue
            if ct1 != cone and float(np.dot(nf1, curr_nf)) > 0.99:
                continue
            t1 = _ray_intersects_triangle(p_src, dir0, vv1[0], vv1[1], vv1[2])
            if t1 is not None and EPS < t1 < dist0 - 1e-3:
                ok = False
                break
        occ[ct0] = not ok

    # ---- write report -------------------------------------------------------
    L: list[str] = []
    L.append("=" * 70)
    L.append(f"FLASHLIGHT DEBUG DUMP  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    L.append("=" * 70)
    if cli_cmd:
        L.append(f"cli     : {cli_cmd}")
    if vectors is not None:
        L.append(f"vectors : {vectors}")
    L.append(f"pos_3d  : ({p_cart[0]:+.4f}, {p_cart[1]:+.4f}, {p_cart[2]:+.4f})"
             f"  r={float(np.linalg.norm(p_cart)):.4f}")
    L.append(f"sph     : az={az_deg:+.2f}°  el={el_deg:+.2f}°")
    L.append(f"heading : ({e1[0]:+.4f}, {e1[1]:+.4f}, {e1[2]:+.4f})")
    L.append(f"cone    : {cone}")
    L.append(f"screen  : {rows}×{cols}  center=(r={cy},c={cx})  scale={scale:.2f}")
    L.append(f"p_src   : ({p_src[0]:+.4f}, {p_src[1]:+.4f}, {p_src[2]:+.4f})")
    L.append(f"cos_max : {cos_tmax:.4f}  (half-angle={_M3_THETA_MAX}°)")
    L.append("")

    hdr = (f"{'face':<22} {'r':>4} {'c':>5}  {'dr':>4} {'dc':>5}"
           f"  {'scr_cos':>7}  {'dist3d':>6}  {'cos3d':>6}  {'occ':>3}  {'hemi':>4}  note")
    L.append(hdr)
    L.append("-" * len(hdr))

    sorted_front = sorted(front, key=lambda ct: float(
        np.dot(np.mean([ray(l) for l in ct], axis=0), view_dir)))

    for ct in sorted_front:
        _, c3, nf3 = m3[ct]
        scr_r, scr_c = scr_of(c3)
        dv   = c3 - p_src
        dist = float(np.linalg.norm(dv))
        cos3d = float(np.dot(dv / dist, h_proj)) if dist > 1e-12 else 0.0
        in_hemi = cos3d > 0.0
        occluded = occ.get(ct, False)

        if scr_r is not None:
            dr = cy - scr_r
            dc = (scr_c - cx) * 0.5
            dlen = math.sqrt(dr * dr + dc * dc)
            scr_cos = ((dr * h_scr_y + dc * h_scr_x) / (dlen * h_scr_len)
                       if dlen > 0.5 else 1.0)
            in_scr = scr_cos > cos_tmax
        else:
            dr, dc, scr_cos, in_scr = 0, 0, 0.0, False

        note = ""
        if ct == cone:
            note = "← CURRENT FACE"
        elif occluded:
            note = "occluded"
        elif not in_hemi:
            note = "behind h_proj"
        elif in_scr and not occluded and in_hemi:
            note = "★ ILLUMINATED"

        scr_r_s = str(scr_r) if scr_r is not None else "?"
        scr_c_s = str(scr_c) if scr_c is not None else "?"
        L.append(
            f"{str(ct):<22} {scr_r_s:>4} {scr_c_s:>5}  {dr:>4} {dc:>5.1f}"
            f"  {scr_cos:>7.3f}  {dist:>6.3f}  {cos3d:>6.3f}"
            f"  {'Y' if occluded else 'n':>3}  {'Y' if in_hemi else 'n':>4}  {note}"
        )

    return L


def run_display_demo(
    fan: Fan,
    vc: object,
    agent: object = None,
    allow_deletion: bool = False,
    initial_pos: np.ndarray | None = None,
    initial_heading: np.ndarray | None = None,
    initial_color: int = 0,
    initial_flashlight: bool = False,
    vectors: list | None = None,
    cli_cmd: str = "",
    max_frames: int | None = None,
) -> None:
    """Launch a curses demo: render the fan with a player; quit on 'q'.

    Parameters
    ----------
    fan : regfans.Fan
        The fan to display.
    vc : regfans.VectorConfiguration
        The vector configuration used for circuit queries.
    agent : object or None, optional
        Optional agent with ``.player`` and ``.advance(fan)`` methods.
        When provided the agent drives movement; arrow keys instead
        adjust agent speed or nudge heading.
    allow_deletion : bool, optional
        Enable deletion mode at startup.
    initial_pos : np.ndarray or None, optional
        Starting position 3-vector (normalised to a direction).
    initial_heading : np.ndarray or None, optional
        Starting heading 3-vector.
    initial_color : int, optional
        Color mode at startup — 0 wireframe, 1 radius, 2 sun.
    initial_flashlight : bool, optional
        Start with the flashlight on.
    """
    TURN       = 0.04   # min turn rate (radians per frame at 60 fps)
    MAX_TURN   = 0.10   # max turn rate
    TURN_ACCEL = 0.006  # turn rate gain per frame
    MAX_SPEED  = 0.11   # top speed (arc-length per frame at 60 fps)
    MIN_SPEED  = 0.003
    ACCEL      = 0.004  # speed gain per frame
    DECEL      = 0.78   # speed multiplier applied when over-turning
    LAT_ACCEL  = 0.002  # max lateral "acceleration"; critical speed = LAT_ACCEL/TURN
    _FRAME_DT    = 1.0 / 60.0  # target frame duration (seconds)
    # Normalise display so vectors of max norm project to a fixed visual radius.
    _TARGET_NORM    = 1.9
    _max_norm       = float(np.linalg.norm(fan.vectors(), axis=1).max()) or 1.0
    _view_scale     = _TARGET_NORM / _max_norm
    _allow_deletion  = allow_deletion   # capture before _main shadows the name
    _debug_log_path  = "/tmp/vcgame_debug.txt"

    def _snapshot(f: object) -> str:
        """Return a human-readable summary of vectors and simplices."""
        cones  = list(f.cones())      # list of label tuples
        # Collect all labels to enumerate vectors in label order.
        all_labels = sorted({l for cone in cones for l in cone})
        def _fmt_vec(v: np.ndarray) -> str:
            return "[" + ", ".join(str(int(x)) for x in v) + "]"

        vecs_str  = "[" + ", ".join(
            _fmt_vec(f.vectors(which=(l,))[0]) for l in all_labels
        ) + "]"
        labels_str = "[" + ", ".join(str(l) for l in all_labels) + "]"
        simplices_str = "[" + ", ".join(
            "[" + ", ".join(str(l) for l in cone) + "]"
            for cone in sorted(cones)
        ) + "]"
        return (
            f"vectors   = {vecs_str}\n"
            f"labels    = {labels_str}\n"
            f"simplices = {simplices_str}"
        )

    def _main(stdscr: _CursesWindow) -> None:
        curses.curs_set(0)
        stdscr.keypad(True)
        curses.mousemask(0)
        stdscr.nodelay(True)  # non-blocking; timing via time.sleep
        _pos0 = initial_pos     if initial_pos     is not None else [1.0, 0.2, 0.1]
        _hdg0 = initial_heading if initial_heading is not None else [0.0, 1.0, 0.0]
        if agent is None:
            player = Player(_pos0, _hdg0)
            from game.agents.random_agent import RandomAgent as _RA
            _agent_obj = _RA(player)
        else:
            player     = agent.player
            _agent_obj = agent
        renderer       = Renderer(fan, stdscr)
        allow_deletion = _allow_deletion
        locked         = False
        sphere_mode    = False
        color_mode     = initial_color
        flashlight_on  = initial_flashlight
        symbol_mode    = 0
        agent_active   = agent is not None
        _speed         = LAT_ACCEL / TURN  # start at the critical speed
        _turn_rate     = TURN
        _sun_angle     = 0.0
        _SUN_ROT_RATE  = 0.005 / 3        # radians per frame at 60 fps

        # OS-level key press/release tracking via pynput.
        # _pressed is updated from a background thread; read each frame.
        _pressed: set[int] = set()
        _kb_listener = None
        _use_pynput  = False
        _lock        = threading.Lock()
        if _HAS_PYNPUT:
            try:
                _KEY_MAP = {
                    _pynput_kb.Key.up:    curses.KEY_UP,
                    _pynput_kb.Key.down:  curses.KEY_DOWN,
                    _pynput_kb.Key.left:  curses.KEY_LEFT,
                    _pynput_kb.Key.right: curses.KEY_RIGHT,
                }
                def _on_press(k):
                    c = _KEY_MAP.get(k)
                    if c is not None:
                        with _lock: _pressed.add(c)
                def _on_release(k):
                    c = _KEY_MAP.get(k)
                    if c is not None:
                        with _lock: _pressed.discard(c)
                _kb_listener = _pynput_kb.Listener(
                    on_press=_on_press, on_release=_on_release)
                _kb_listener.start()
                _use_pynput = True
            except Exception:
                _kb_listener = None
        # TTL fallback state (used when pynput unavailable/failed)
        _KEY_TTL  = 0.12
        _last_seen: dict[int, float] = {}

        nonlocal_fan  = [fan]
        _irregularity = [not fan.is_regular()]   # updated on every flip

        def _try_move(step: float) -> None:
            f = nonlocal_fan[0]
            old_cone = player.current_cone(f)
            crossed  = player.move(step, f)
            if crossed is None or locked:
                return
            new_cone = player.current_cone(f)
            circ     = player.find_circuit_for_crossing(old_cone, new_cone, f)
            if circ is None:
                return
            if min(circ.signature) == 1 and not allow_deletion:
                return
            new_fan = f.flip(circ)
            if new_fan is None:
                return
            nonlocal_fan[0]    = new_fan
            renderer._fan      = new_fan
            renderer._edge_map = _cone_edge_map(new_fan)
            _irregularity[0]   = not new_fan.is_regular()

        def _agent_step() -> None:
            f        = nonlocal_fan[0]
            old_cone = player.current_cone(f)
            _agent_obj.advance(f)
            if locked:
                return
            new_cone = player.current_cone(f)
            if new_cone == old_cone:
                return
            circ = player.find_circuit_for_crossing(
                old_cone, new_cone, f,
            )
            if circ is None:
                return
            if min(circ.signature) == 1 and not allow_deletion:
                return
            new_fan = f.flip(circ)
            if new_fan is None:
                return
            nonlocal_fan[0]    = new_fan
            renderer._fan      = new_fan
            renderer._edge_map = _cone_edge_map(new_fan)
            _irregularity[0]   = not new_fan.is_regular()

        _agent_rate = 1.0   # steps per frame (can be fractional)
        _agent_acc  = 0.0   # fractional accumulator
        _debug_on       = False  # whether debug overlay is visible
        _edge_thickness = 1
        _frame_count = 0

        try:
            while True:
                if agent_active:
                    _agent_acc += _agent_rate
                    while _agent_acc >= 1.0:
                        _agent_step()
                        _agent_acc -= 1.0
                f       = nonlocal_fan[0]
                cone    = player.current_cone(f)
                facet   = player.pointed_facet(f)
                # Compute flippability for each edge of the current cone.
                _flip_status: dict[tuple[int, int], bool] = {}
                _cl = list(cone)
                for _i in range(len(_cl)):
                    _ea = _cl[_i];  _eb = _cl[(_i + 1) % len(_cl)]
                    _ek = (min(_ea, _eb), max(_ea, _eb))
                    _adjs = renderer._edge_map.get(_ek, set()) - {cone}
                    _ok = False
                    for _adj in _adjs:
                        _c = player.find_circuit_for_crossing(cone, _adj, f)
                        if _c is not None and f.flip(_c) is not None:
                            _ok = True
                            break
                    _flip_status[_ek] = _ok
                _pdbg = renderer.draw(player.direction, player.heading, cone,
                                      facet, locked, allow_deletion, color_mode,
                                      _view_scale, _flip_status, _irregularity[0],
                                      sphere_mode, agent_active, _sun_angle,
                                      flashlight=flashlight_on,
                                      symbol_mode=symbol_mode,
                                      pixel_debug=_debug_on,
                                      edge_thickness=_edge_thickness)
                _sun_angle += _SUN_ROT_RATE

                if _debug_on:
                    _dbg_lines = _debug_dump(
                        player, nonlocal_fan[0], stdscr, _view_scale,
                        vectors=vectors, cli_cmd=cli_cmd,
                    )
                    _snap_lines = _snapshot(nonlocal_fan[0]).splitlines()
                    _dbg_lines = _dbg_lines + [""] + _snap_lines
                    with open(_debug_log_path, "w") as _df:
                        _df.write("\n".join(_dbg_lines) + "\n")
                        if _pdbg:
                            _df.write("\n")
                            _df.write("\n".join(_pdbg) + "\n")
                    _dbg_rows, _dbg_cols = stdscr.getmaxyx()
                    _dbg_attr = curses.A_REVERSE
                    for _di, _dl in enumerate(_dbg_lines):
                        if _di >= _dbg_rows - _HUD_ROWS:
                            break
                        _txt = _dl[: _dbg_cols - 1].ljust(_dbg_cols - 1)
                        try:
                            stdscr.addstr(_di, 0, _txt, _dbg_attr)
                        except curses.error:
                            pass

                stdscr.refresh()
                _frame_count += 1
                if max_frames is not None and _frame_count >= max_frames:
                    stdscr.nodelay(False)
                    stdscr.getch()
                    break

                # Drain ALL pending input this frame so terminal key-repeat
                # buffering cannot cause movement to continue after key release.
                _move_keys: set[int] = set()
                _MOVE_SET = {curses.KEY_UP, curses.KEY_DOWN,
                             curses.KEY_LEFT, curses.KEY_RIGHT}
                _quit = False
                while True:
                    key = stdscr.getch()
                    if key == -1:
                        break
                    if key in _MOVE_SET:
                        _move_keys.add(key)
                    elif key == ord("q"):
                        _quit = True
                    elif key == ord("a"):
                        agent_active = not agent_active
                    elif key == ord("s"):
                        sphere_mode = not sphere_mode
                    elif key == ord("d"):
                        allow_deletion = not allow_deletion
                    elif key == ord("f"):
                        flashlight_on = not flashlight_on
                    elif key == ord("l"):
                        locked = not locked
                    elif key == ord("p"):
                        _debug_on = not _debug_on
                    elif key == ord("1"):
                        color_mode = 1 if color_mode != 1 else 0
                    elif key == ord("2"):
                        color_mode = 2 if color_mode != 2 else 0
                    elif key == ord("6"):
                        symbol_mode = 0
                    elif key == ord("7"):
                        symbol_mode = 1
                    elif key == ord("8"):
                        symbol_mode = 2
                    elif key == ord("9"):
                        symbol_mode = 3
                    elif key == ord("0"):
                        symbol_mode = 4
                    elif key == ord("-"):
                        symbol_mode = 5
                    elif key == ord("t"):
                        _edge_thickness = 2 if _edge_thickness == 1 else 1
                if _quit:
                    break

                # Determine which movement keys are currently active.
                if _use_pynput:
                    with _lock:
                        _active = _pressed.copy()
                else:
                    _now = time.monotonic()
                    for _k in _move_keys:
                        _last_seen[_k] = _now
                    _active = {_k for _k, _t in _last_seen.items()
                               if _now - _t <= _KEY_TTL}

                if not agent_active:
                    _fwd   = curses.KEY_UP    in _active
                    _back  = curses.KEY_DOWN  in _active
                    _left  = curses.KEY_LEFT  in _active
                    _right = curses.KEY_RIGHT in _active

                    if _fwd or _back:
                        _try_move(_speed if _fwd else -_speed)
                        _speed = min(MAX_SPEED, _speed + ACCEL)
                    else:
                        _speed = max(MIN_SPEED, _speed * 0.85)

                    if _left or _right:
                        _turn_rate = min(MAX_TURN, _turn_rate + TURN_ACCEL)
                        if _left:
                            player.turn(-_turn_rate)
                        else:
                            player.turn(_turn_rate)
                        if _turn_rate * _speed > LAT_ACCEL:
                            _speed = max(MIN_SPEED, _speed * DECEL)
                    else:
                        _turn_rate = TURN

                elif agent_active:
                    _RATE_FACTOR = 1.5
                    _RATE_MIN    = 0.05
                    _RATE_MAX    = 8.0
                    _NUDGE       = 0.20
                    if   curses.KEY_UP    in _move_keys:
                        _agent_rate = min(_RATE_MAX,
                                          _agent_rate * _RATE_FACTOR)
                    elif curses.KEY_DOWN  in _move_keys:
                        _agent_rate = max(_RATE_MIN,
                                          _agent_rate / _RATE_FACTOR)
                    if   curses.KEY_LEFT  in _move_keys:
                        _agent_obj.player.turn(-_NUDGE)
                    elif curses.KEY_RIGHT in _move_keys:
                        _agent_obj.player.turn(_NUDGE)

                time.sleep(_FRAME_DT)

        finally:
            if _kb_listener is not None:
                _kb_listener.stop()

    # Redirect fd 2 to /dev/null for the entire curses session so that
    # C-level stderr output (e.g. macOS pynput accessibility warning)
    # does not bleed into the display.
    _stderr_fd   = sys.stderr.fileno()
    _saved_fd    = os.dup(_stderr_fd)
    _devnull_fd  = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull_fd, _stderr_fd)
    os.close(_devnull_fd)
    try:
        curses.wrapper(_main)
    finally:
        os.dup2(_saved_fd, _stderr_fd)
        os.close(_saved_fd)

