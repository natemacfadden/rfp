"""
Curses-based ASCII renderer for the fan and player position on S².
"""

from __future__ import annotations

import curses
from typing import TYPE_CHECKING

import numpy as np

from .player import Player

if TYPE_CHECKING:
    from regfans import Fan
    from _curses import _CursesWindow

_RADIUS_PAIR_START = 6

# (r, g, b) in 0–1000 range for curses
_VIRIDIS_KEYS: list[tuple[int, int, int]] = [
    (267,   4, 329),
    (231, 322, 545),
    (129, 569, 549),
    (173, 694, 478),
    (369, 788, 384),
    (675, 863, 204),
    (867, 902, 114),
    (992, 906, 145),
]


def _viridis_rgb(t: float) -> tuple[int, int, int]:
    t  = max(0.0, min(1.0, t))
    s  = t * (len(_VIRIDIS_KEYS) - 1)
    lo = int(s)
    hi = min(lo + 1, len(_VIRIDIS_KEYS) - 1)
    f  = s - lo
    r0, g0, b0 = _VIRIDIS_KEYS[lo]
    r1, g1, b1 = _VIRIDIS_KEYS[hi]
    return int(r0 + f*(r1-r0)), int(g0 + f*(g1-g0)), int(b0 + f*(b1-b0))


def _project(
    v: np.ndarray,
    p: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
) -> tuple[float, float] | None:
    """
    **Description:**
    Project a 3D vector onto the tangent plane at `p`, returning 2D
    screen coords.
    Returns `None` if the vector is nearly antipodal to `p`.

    **Arguments:**
    - `v`: Vector to project (need not be unit).
    - `p`: Player position (unit vector), normal to tangent plane.
    - `e1`: Tangent basis vector pointing "up" on screen.
    - `e2`: Tangent basis vector pointing "right" on screen.

    **Returns:**
    `(x_screen, y_screen)` or `None` if clipped.
    """
    n = np.linalg.norm(v)
    if n < 1e-12:
        return None
    vn = v / n
    if np.dot(vn, p) < -0.95:
        return None
    v_proj = v - np.dot(v, p) * p
    return float(np.dot(v_proj, e2)), float(np.dot(v_proj, e1))


def _cone_edge_map(fan: Fan) -> dict[tuple[int, int], set[tuple[int, ...]]]:
    """
    **Description:**
    Return a mapping from each edge (sorted ray-label pair) to the set of cones
    (label tuples) that contain it.

    **Arguments:**
    - `fan`: A `regfans.Fan`.

    **Returns:**
    Dict of `(min_label, max_label)` → set of cone label tuples.
    """
    edge_map: dict[tuple[int, int], set[tuple[int, ...]]] = {}
    for cone in fan.cones():
        labels = list(cone)
        ct = tuple(sorted(labels))
        for i in range(len(labels)):
            a, b = labels[i], labels[(i + 1) % len(labels)]
            key = (min(a, b), max(a, b))
            edge_map.setdefault(key, set()).add(ct)
    return edge_map


def _draw_line(
    scr: _CursesWindow,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    ch: str,
    attr: int,
) -> None:
    """
    **Description:**
    Draw a line between two screen positions using Bresenham's algorithm.

    **Arguments:**
    - `scr`: Curses window.
    - `r0`, `c0`: Start row and column.
    - `r1`, `c1`: End row and column.
    - `ch`: Character to draw.
    - `attr`: Curses attribute.

    **Returns:**
    Nothing.
    """
    rows, cols = scr.getmaxyx()

    def put(r: int, c: int) -> None:
        if 0 <= r < rows - 1 and 0 <= c < cols - 1:
            try:
                scr.addstr(r, c, ch, attr)
            except curses.error:
                pass

    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1
    r, c = r0, c0

    if dc >= dr:
        err = dc // 2
        while c != c1:
            put(r, c)
            err -= dr
            if err < 0:
                r += sr
                err += dc
            c += sc
    else:
        err = dr // 2
        while r != r1:
            put(r, c)
            err -= dc
            if err < 0:
                c += sc
                err += dr
            r += sr
    put(r1, c1)


def _fill_triangle(
    scr: _CursesWindow,
    pts: list[tuple[int, int]],
    ch: str,
    attr: int,
) -> None:
    rows, cols = scr.getmaxyx()
    pts_sorted = sorted(pts, key=lambda rc: rc[0])
    (r0, c0), (r1, c1), (r2, c2) = pts_sorted
    if r0 == r2:
        return

    def interp(ra: int, ca: int, rb: int, cb: int, r: int) -> float:
        if ra == rb:
            return float(ca)
        return ca + (cb - ca) * (r - ra) / (rb - ra)

    for r in range(max(0, r0), min(rows - 1, r2 + 1)):
        if r <= r1:
            cl = interp(r0, c0, r1, c1, r)
            cr = interp(r0, c0, r2, c2, r)
        else:
            cl = interp(r1, c1, r2, c2, r)
            cr = interp(r0, c0, r2, c2, r)
        left  = int(round(min(cl, cr)))
        right = int(round(max(cl, cr)))
        for c in range(max(0, left), min(cols - 1, right + 1)):
            try:
                scr.addstr(r, c, ch, attr)
            except curses.error:
                pass


def _fill_triangle_radius(
    scr: _CursesWindow,
    pts: list[tuple[int, int]],
    v3d: list[np.ndarray],
    r_min: float,
    r_max: float,
    n_pairs: int,
    pair_start: int,
) -> None:
    rows, cols = scr.getmaxyx()
    order = sorted(range(3), key=lambda i: pts[i][0])
    (r0, c0), (r1, c1), (r2, c2) = [pts[o] for o in order]
    rn0, rn1, rn2 = [float(np.linalg.norm(v3d[o])) for o in order]

    if r0 == r2:
        return

    def lerp(a: float, b: float, ra: int, rb: int, r: int) -> float:
        return a if ra == rb else a + (b - a) * (r - ra) / (rb - ra)

    for r in range(max(0, r0), min(rows - 1, r2 + 1)):
        if r <= r1:
            cl, cr   = lerp(c0, c1, r0, r1, r),  lerp(c0, c2, r0, r2, r)
            rnl, rnr = lerp(rn0, rn1, r0, r1, r), lerp(rn0, rn2, r0, r2, r)
        else:
            cl, cr   = lerp(c1, c2, r1, r2, r),  lerp(c0, c2, r0, r2, r)
            rnl, rnr = lerp(rn1, rn2, r1, r2, r), lerp(rn0, rn2, r0, r2, r)
        if cl > cr:
            cl, cr, rnl, rnr = cr, cl, rnr, rnl
        left, right = int(round(cl)), int(round(cr))
        for c in range(max(0, left), min(cols - 1, right + 1)):
            tc     = (c - left) / (right - left) if right > left else 0.0
            radius = rnl + tc * (rnr - rnl)
            t      = max(0.0, min(1.0, (radius - r_min) / (r_max - r_min)))
            pair   = pair_start + round(t * (n_pairs - 1))
            try:
                attr = curses.color_pair(pair) | curses.A_BOLD
                scr.addstr(r, c, "\u2592", attr)
            except curses.error:
                pass


class Renderer:
    """
    **Description:**
    Curses-based renderer for a fan and player position on S². Projects 3D cone
    edges as flat line segments onto the tangent plane at the player's position.

    **Arguments:**
    - `fan`: A `regfans.Fan` whose cones will be drawn.
    - `stdscr`: A curses window (full screen).
    """

    def __init__(self, fan: Fan, stdscr: _CursesWindow) -> None:
        """
        **Description:**
        Initialise the renderer with a fan and curses screen.

        **Arguments:**
        - `fan`: A `regfans.Fan`.
        - `stdscr`: A curses window.

        **Returns:**
        Nothing.
        """
        self._fan      = fan
        self._stdscr   = stdscr
        self._edge_map = _cone_edge_map(fan)
        self._init_colors()

    def _init_colors(self) -> None:
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN,    -1)
        curses.init_pair(2, curses.COLOR_YELLOW,  -1)
        curses.init_pair(3, curses.COLOR_GREEN,   -1)
        curses.init_pair(4, curses.COLOR_WHITE,   -1)
        curses.init_pair(5, curses.COLOR_WHITE,   -1)
        if curses.can_change_color() and curses.COLORS >= 32:
            n_avail = curses.COLOR_PAIRS - _RADIUS_PAIR_START
            n = max(2, min(32, curses.COLORS - 16, n_avail))
            for i in range(n):
                r, g, b = _viridis_rgb(i / (n - 1))
                curses.init_color(16 + i, r, g, b)
                curses.init_pair(_RADIUS_PAIR_START + i, 16 + i, -1)
            self._n_radius = n
        else:
            for i, fg in enumerate([curses.COLOR_BLUE, curses.COLOR_CYAN,
                                     curses.COLOR_GREEN, curses.COLOR_YELLOW,
                                     curses.COLOR_RED]):
                curses.init_pair(_RADIUS_PAIR_START + i, fg, -1)
            self._n_radius = 5

    def draw(
        self,
        player_pos: np.ndarray,
        player_heading: np.ndarray,
        current_cone: tuple[int, ...],
        pointed_facet: tuple[int, int] | None = None,
        locked: bool = False,
        allow_deletion: bool = True,
    ) -> None:
        """
        **Description:**
        Render one frame: all cone edges as flat projected line segments,
        the active cone highlighted, the pointed-at facet highlighted,
        and the player marker.

        **Arguments:**
        - `player_pos`: Unit vector in R³ giving the player's position on S².
        - `player_heading`: Unit tangent vector at `player_pos` pointing "up".
        - `current_cone`: Label tuple of the cone containing the player.
        - `pointed_facet`: Sorted label pair of the facet the player is
          aiming at, or `None`.
        - `locked`: Whether movement is locked.
        - `allow_deletion`: Whether deletion mode is active.

        **Returns:**
        Nothing.
        """
        scr = self._stdscr
        scr.erase()
        rows, cols = scr.getmaxyx()
        cy, cx = rows // 2, cols // 2
        scale  = float(min(rows, cols // 2) // 2 - 2) * 0.75

        p        = player_pos
        e1       = player_heading
        view_dir = p
        e1_new   = e1
        e2_new   = np.cross(p, e1)

        fan    = self._fan
        labels = list(current_cone)
        active_edge_set: set[tuple[int, int]] = set()
        for i in range(len(labels)):
            a, b = labels[i], labels[(i + 1) % len(labels)]
            active_edge_set.add((min(a, b), max(a, b)))

        ray_cache: dict[int, np.ndarray] = {}

        def ray(label: int) -> np.ndarray:
            if label not in ray_cache:
                ray_cache[label] = fan.vectors(which=(label,))[0]
            return ray_cache[label]

        front_cones: set[tuple[int, ...]] = set()
        cone_normals: dict[tuple[int, ...], np.ndarray] = {}
        for cone in fan.cones():
            clabels = list(cone)
            ct = tuple(sorted(clabels))
            vs = [ray(l) for l in clabels]
            n  = np.cross(vs[1] - vs[0], vs[2] - vs[0])
            if np.dot(n, vs[0]) < 0:
                n = -n
            nn = np.linalg.norm(n)
            if nn > 1e-12:
                n = n / nn
            cone_normals[ct] = n
            if np.dot(n, view_dir) > 0:
                front_cones.add(ct)

        def screen_pt(label: int) -> tuple[int, int] | None:
            coord = _project(ray(label), view_dir, e1_new, e2_new)
            if coord is None:
                return None
            col = cx + int(round(coord[0] * scale * 2))
            row = cy - int(round(coord[1] * scale))
            return (row, col)

        def _draw_edge(a: int, b: int, ch: str, attr: int) -> None:
            ca = _project(ray(a), view_dir, e1_new, e2_new)
            cb = _project(ray(b), view_dir, e1_new, e2_new)
            if ca is None or cb is None:
                return
            c0 = cx + int(round(ca[0] * scale * 2))
            r0 = cy - int(round(ca[1] * scale))
            c1 = cx + int(round(cb[0] * scale * 2))
            r1 = cy - int(round(cb[1] * scale))
            _draw_line(scr, r0,     c0, r1,     c1, ch, attr)
            _draw_line(scr, r0 + 1, c0, r1 + 1, c1, ch, attr)

        all_norms = np.linalg.norm(fan.vectors(), axis=1)
        r_min = float(all_norms.min())
        r_max = float(all_norms.max())
        if r_max <= r_min:
            r_max = r_min + 1.0

        sorted_front = sorted(
            front_cones,
            key=lambda ct: float(
                np.dot(np.mean([ray(l) for l in ct], axis=0), view_dir)
            ),
        )

        for ct in sorted_front:
            clabels = list(ct)
            pts = [screen_pt(l) for l in clabels]
            if any(pt is None for pt in pts):
                continue
            _fill_triangle_radius(  # type: ignore[arg-type]
                scr, pts,
                [ray(l) for l in clabels],
                r_min, r_max,
                self._n_radius, _RADIUS_PAIR_START,
            )
            for i in range(len(clabels)):
                a, b = clabels[i], clabels[(i + 1) % len(clabels)]
                edge = (min(a, b), max(a, b))
                if pointed_facet and edge == pointed_facet:
                    continue
                is_active = edge in active_edge_set
                ch_e   = "+" if is_active else "."
                attr_e = (curses.color_pair(2) | curses.A_BOLD if is_active
                          else curses.color_pair(1))
                _draw_edge(a, b, ch_e, attr_e)

        if pointed_facet:
            a, b = pointed_facet
            if self._edge_map.get(pointed_facet, set()) & front_cones:
                _draw_edge(a, b, "*", curses.color_pair(5) | curses.A_BOLD)

        coord = _project(p, view_dir, e1_new, e2_new)
        if coord is not None:
            col = cx + int(round(coord[0] * scale * 2))
            row = cy - int(round(coord[1] * scale))
            attr = curses.color_pair(3) | curses.A_BOLD
            for dr, s in ((-1, "^^"), (0, "||")):
                r, c = row + dr, col
                if 0 <= r < rows - 1 and 0 <= c + 1 < cols - 1:
                    try:
                        scr.addstr(r, c, s, attr)
                    except curses.error:
                        pass

        facet_str = str(pointed_facet) if pointed_facet else "none"
        hud_base = (
            f" pos=({p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f})"
            f"  cone={current_cone}"
            f"  facet={facet_str}"
            f"  "
        )
        stay_str  = "[S]tay:ON" if locked else "[S]tay:off"
        stay_attr = (curses.color_pair(2) | curses.A_BOLD
                     if locked else curses.color_pair(4))
        del_str   = "  [D]el:ON" if allow_deletion else "  [D]el:off"
        del_attr  = (curses.color_pair(2) | curses.A_BOLD
                     if allow_deletion else curses.color_pair(4))
        tail = "  [q]uit"
        col = 0
        try:
            scr.addstr(rows - 1, col,
                       hud_base[: cols - 1], curses.color_pair(4))
            col += len(hud_base)
            if col < cols - 1:
                scr.addstr(rows - 1, col,
                           stay_str[: cols - 1 - col], stay_attr)
                col += len(stay_str)
            if col < cols - 1:
                scr.addstr(rows - 1, col, del_str[: cols - 1 - col], del_attr)
                col += len(del_str)
            if col < cols - 1:
                scr.addstr(rows - 1, col,
                           tail[: cols - 1 - col], curses.color_pair(4))
        except curses.error:
            pass


def run_display_demo(
    fan: Fan, vc: object, agent: object = None,
) -> None:
    """
    **Description:**
    Launch a curses demo: renders the fan with a player and waits
    for 'q' to quit.

    **Arguments:**
    - `fan`: A `regfans.Fan` to display.
    - `vc`: A `regfans.VectorConfiguration` for circuit queries.
    - `agent`: Optional agent with `.player` and `.advance(fan)`.
      When provided the agent drives movement; arrow keys are
      disabled and the loop runs at ~20 fps.

    **Returns:**
    Nothing.
    """
    STEP = 0.08
    TURN = 0.12

    def _main(stdscr: _CursesWindow) -> None:
        curses.curs_set(0)
        stdscr.keypad(True)
        if agent is None:
            stdscr.nodelay(False)
            player = Player([1.0, 0.2, 0.1], [0.0, 1.0, 0.0])
        else:
            stdscr.timeout(50)
            player = agent.player
        renderer       = Renderer(fan, stdscr)
        allow_deletion = True
        locked         = False

        nonlocal_fan = [fan]

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

        def _agent_step() -> None:
            f        = nonlocal_fan[0]
            old_cone = player.current_cone(f)
            agent.advance(f)
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

        while True:
            if agent is not None:
                _agent_step()
            f       = nonlocal_fan[0]
            cone    = player.current_cone(f)
            facet   = player.pointed_facet(f)
            renderer.draw(player.position, player.heading, cone,
                          facet, locked, allow_deletion)
            stdscr.refresh()
            key = stdscr.getch()
            if   key == ord("q"):  break
            elif key == ord("s"):  locked = not locked
            elif key == ord("d"):  allow_deletion = not allow_deletion
            elif agent is None:
                if   key == curses.KEY_UP:    _try_move(STEP)
                elif key == curses.KEY_DOWN:  _try_move(-STEP)
                elif key == curses.KEY_LEFT:
                    player.turn(-TURN)
                    _try_move(STEP)
                elif key == curses.KEY_RIGHT:
                    player.turn(TURN)
                    _try_move(STEP)

    curses.wrapper(_main)
