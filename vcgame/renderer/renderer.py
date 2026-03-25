"""
Curses-based ASCII renderer for the fan and player position on S².

Rendering uses a backward (ray-casting) approach: for each screen pixel a
line is fired from the screen plane inward along -p, and the first cone face
it hits is shaded.  Edges are drawn on top via Bresenham (flat) or SLERP
arcs (sphere).
"""

from __future__ import annotations

import curses
import math
from typing import TYPE_CHECKING

import numba
import numpy as np

from .colors import (
    _init_colors,
    _RADIUS_PAIR_START,
    _EDGE_PAIR_BASE,
    _IREG_BG_PAIR,
    _FILL_PAIR,
)

if TYPE_CHECKING:
    from regfans import Fan
    from _curses import _CursesWindow

_SLERP_STEP   = 0.04   # arc-length step for spherical arc sampling
_SUN_DISTANCE = 20.0
_SUN_REF      = np.array([1.0, 1.0, 1.0])

_COLOR_LABELS  = ("off", "radius", "sun")
# Symbol styles: (label, ramp_string).  Brightness t∈[0,1] indexes the ramp.
_SYMBOL_STYLES: tuple = (
    ("block",   "\u2591\u2592\u2593\u2588"),   # ░▒▓█  — block shading
    ("digits",  "123456789"),                   # numeric brightness 1–9
    ("ascii",   " .:-=+*#%@"),                  # classic ASCII density ramp
)
_M3_HEIGHT     = 0.003  # player elevation above current face (flashlight mode)
_M3_THETA_MAX  = 55.0   # flashlight cone half-angle from heading, degrees

# Point light for "sun" mode.
# Placed diagonally so all cube axes shade differently.
# _SUN_BRIGHTNESS normalises intensity so the closest expected surface
# (at ~1 unit from origin, ~19 units from the sun) maps to roughly 1.
_SUN_POS: np.ndarray = np.array([1.0, 2.0, 3.0])
_SUN_POS = _SUN_POS / float(np.linalg.norm(_SUN_POS)) * _SUN_DISTANCE
_SUN_BRIGHTNESS   = float(np.dot(_SUN_POS - _SUN_REF, _SUN_POS - _SUN_REF))
_SUN_AMBIENT      = 0.12   # base illumination on all surfaces, including shadowed ones
_SUN_MAX          = 0.72   # cap on sun brightness (prevents over-saturation at peak)
_DIM_LEVEL        = 0.45   # default brightness when color fill is off
_FL_BOOST         = 1.55   # max flashlight brightness increase above _DIM_LEVEL

_HUD_ROWS = 2  # number of rows reserved at screen bottom for HUD

# Distance from player surface position to screen plane, in scene units.
# Acts as the FOV parameter: larger = narrower field of view.
_FOV_DIST: float = 1.0


# ---------------------------------------------------------------------------
# Curses helpers
# ---------------------------------------------------------------------------

def _addstr(scr, r: int, c: int, text: str, attr: int = 0) -> None:
    """Write text to the curses screen, silently ignoring out-of-bounds errors."""
    try:
        scr.addstr(r, c, text, attr)
    except curses.error:
        pass


def _orient_normal(n: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Return n flipped so it points toward ref (dot(n, ref) > 0)."""
    return n if np.dot(n, ref) >= 0.0 else -n


def _edge_attrs(
    edge: tuple,
    is_active: bool,
    flip_status: dict | None,
) -> tuple[str, int]:
    """Return (character, curses attr) for rendering an edge."""
    if is_active and flip_status is not None:
        flippable = flip_status.get(edge, False)
        return "+", curses.color_pair(
            _EDGE_PAIR_BASE + (2 if flippable else 3)
        )
    if is_active:
        return "+", curses.color_pair(2) | curses.A_BOLD
    return ".", curses.color_pair(1)


def _project(
    v: np.ndarray,
    p: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
) -> tuple[float, float] | None:
    """Project a 3D vector onto the tangent plane at ``p``.

    Returns 2D screen coordinates, or ``None`` if the vector is nearly
    antipodal to ``p``.

    Parameters
    ----------
    v : np.ndarray
        Vector to project (need not be unit).
    p : np.ndarray
        Player position (unit vector), normal to tangent plane.
    e1 : np.ndarray
        Tangent basis vector pointing "up" on screen.
    e2 : np.ndarray
        Tangent basis vector pointing "right" on screen.

    Returns
    -------
    tuple[float, float] or None
        ``(x_screen, y_screen)``, or ``None`` if clipped.
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
    """Return a mapping from each edge to the set of cones containing it."""
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
    """Draw a line between two screen positions using Bresenham's algorithm."""
    rows, cols = scr.getmaxyx()

    def put(r: int, c: int) -> None:
        if 0 <= r < rows - _HUD_ROWS and 0 <= c < cols - 1:
            _addstr(scr, r, c, ch, attr)

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


def _ray_intersects_triangle(
    orig: np.ndarray,
    d: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
) -> float | None:
    """Möller–Trumbore ray–triangle intersection.

    Parameters
    ----------
    orig : np.ndarray
        Ray origin.
    d : np.ndarray
        Ray direction (need not be normalised).
    v0, v1, v2 : np.ndarray
        Triangle vertices.

    Returns
    -------
    float or None
        ``t > 1e-6`` such that ``orig + t*d`` lies on the triangle, or
        ``None`` if there is no intersection.
    """
    e1 = v1 - v0
    e2 = v2 - v0
    h  = np.cross(d, e2)
    a  = float(np.dot(e1, h))
    if abs(a) < 1e-8:
        return None
    f  = 1.0 / a
    s  = orig - v0
    u  = float(f * np.dot(s, h))
    if u < 0.0 or u > 1.0:
        return None
    q  = np.cross(s, e1)
    vv = float(f * np.dot(d, q))
    if vv < 0.0 or u + vv > 1.0:
        return None
    t  = float(f * np.dot(e2, q))
    return t if t > 1e-6 else None


# ---------------------------------------------------------------------------
# Backward rendering helpers
# ---------------------------------------------------------------------------

def _compute_p_surface(
    p: np.ndarray,
    v0: np.ndarray,
    face_normal: np.ndarray,
) -> np.ndarray:
    """Return the point where ray p intersects the face plane.

    The face plane passes through ``v0`` with outward normal ``face_normal``.
    Returns ``p`` (unit vector fallback) if the ray is nearly parallel to the
    plane.

    Parameters
    ----------
    p : np.ndarray
        Unit viewing direction (player position on S²).
    v0 : np.ndarray
        Any vertex of the current cone.
    face_normal : np.ndarray
        Outward unit normal of the current cone's face plane.

    Returns
    -------
    np.ndarray
        3D point on the face plane: ``λ * p`` where
        ``λ = dot(v0, face_normal) / dot(p, face_normal)``.
    """
    denom = float(np.dot(p, face_normal))
    if abs(denom) < 1e-12:
        return p.copy()
    lam = float(np.dot(v0, face_normal)) / denom
    return lam * p


def _pixel_row_positions(
    r: int,
    c_arr: np.ndarray,
    screen_center: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    scale: float,
    cx: int,
    cy: int,
) -> np.ndarray:
    """Return 3D positions for every pixel in screen row ``r``.

    The screen plane is centred at ``screen_center = p_surface + FOV_DIST*p``.
    Each pixel at column ``c`` is offset by scene-unit distances along ``e1``
    (up) and ``e2`` (right).  Columns are scaled by 2× relative to rows to
    compensate for terminal character cells being ~2× taller than wide,
    preserving circular aspect ratio.

    Parameters
    ----------
    r : int
        Screen row index.
    c_arr : np.ndarray
        1-D array of column indices to evaluate, shape ``(N,)``.
    screen_center : np.ndarray
        3D centre of the screen plane, shape ``(3,)``.
    e1 : np.ndarray
        Screen "up" basis vector, shape ``(3,)``.
    e2 : np.ndarray
        Screen "right" basis vector, shape ``(3,)``.
    scale : float
        Pixels per scene unit (vertical).
    cx, cy : int
        Screen centre in pixel coordinates.

    Returns
    -------
    np.ndarray
        Shape ``(N, 3)`` — one 3D position per column.
    """
    s     = (cy - r) / scale                   # scene units along e1
    u_arr = (c_arr - cx) / (scale * 2.0)       # scene units along e2
    row_center = screen_center + s * e1        # (3,)
    return row_center[np.newaxis, :] + u_arr[:, np.newaxis] * e2[np.newaxis, :]



def _sphere_row_hits(
    pixel_row: np.ndarray,
    p: np.ndarray,
) -> np.ndarray:
    """Solve ``|pixel_row[i] - t*p|² = 1`` for each pixel.

    Finds where the line from ``pixel_row[i]`` in direction ``-p`` hits the
    unit sphere.  Takes the smallest positive root.

    Parameters
    ----------
    pixel_row : np.ndarray
        Screen pixel positions, shape ``(N, 3)``.
    p : np.ndarray
        Unit viewing direction (ray direction is ``-p``), shape ``(3,)``.

    Returns
    -------
    np.ndarray
        Shape ``(N,)`` — intersection parameter ``t > 0``, or ``inf`` if the
        line misses the unit sphere.
    """
    # Expanding |pixel - t*p|² = 1 with |p|=1:
    # t² - 2·dot(pixel,p)·t + (|pixel|²-1) = 0
    b      = pixel_row @ p                           # (N,)
    c_coef = np.sum(pixel_row ** 2, axis=1) - 1.0   # (N,)
    disc   = b * b - c_coef                          # (N,) discriminant
    result = np.full(len(pixel_row), np.inf)
    valid  = disc >= 0.0
    if not np.any(valid):
        return result
    sq     = np.sqrt(np.maximum(0.0, disc[valid]))
    t1     = b[valid] - sq
    t2     = b[valid] + sq
    # Take smallest positive root; fall back to larger root if t1 ≤ 0.
    t_best = np.where(t1 > 1e-6, t1, np.where(t2 > 1e-6, t2, np.inf))
    result[valid] = t_best
    return result


def _shadow_blocked(
    hit_pos: np.ndarray,
    target: np.ndarray,
    v0s: np.ndarray,
    v1s: np.ndarray,
    v2s: np.ndarray,
    skip_idx: int = -1,
    eps: float = 1e-3,
) -> bool:
    """Return True if the line segment from ``hit_pos`` to ``target`` is
    blocked by any cone triangle.

    Parameters
    ----------
    hit_pos : np.ndarray
        Start of the shadow ray.
    target : np.ndarray
        End point (sun position or flashlight source).
    v0s, v1s, v2s : np.ndarray
        Stacked triangle vertices, shape ``(T, 3)`` each.
    skip_idx : int
        Row index to skip in the triangle arrays (avoids self-intersection).
        Pass -1 to skip nothing.
    eps : float
        Nudge distance along ray to avoid self-intersection.
    """
    if len(v0s) == 0:
        return False
    to_target = target - hit_pos
    dist      = float(np.linalg.norm(to_target))
    if dist < 1e-12:
        return False
    d    = to_target / dist
    orig = hit_pos + eps * d

    edge1 = v1s - v0s                               # (T, 3)
    edge2 = v2s - v0s                               # (T, 3)
    h     = np.cross(d, edge2)                      # (T, 3)
    a     = np.einsum('ti,ti->t', edge1, h)         # (T,)

    mask = np.abs(a) >= 1e-8
    if 0 <= skip_idx < len(mask):
        mask[skip_idx] = False
    if not np.any(mask):
        return False

    f  = np.where(mask, 1.0 / np.where(mask, a, 1.0), 0.0)
    s  = orig - v0s                                 # (T, 3)
    u  = f * np.einsum('ti,ti->t', s, h)            # (T,)
    mask &= (u >= 0.0) & (u <= 1.0)
    if not np.any(mask):
        return False

    q  = np.cross(s, edge1)                         # (T, 3)
    vv = f * (q @ d)                                # (T,)
    mask &= (vv >= 0.0) & (u + vv <= 1.0)
    if not np.any(mask):
        return False

    t = f * np.einsum('ti,ti->t', q, edge2)         # (T,)
    mask &= (t > 1e-6) & (t < dist - eps)
    return bool(np.any(mask))


def _fl_brightness_pixel(
    hit_pos: np.ndarray,
    p_src: np.ndarray,
    h_proj: np.ndarray,
    cos_tmax: float,
    v0s: np.ndarray,
    v1s: np.ndarray,
    v2s: np.ndarray,
    curr_idx: int,
) -> float:
    """Per-pixel flashlight brightness at ``hit_pos``.

    Returns a value in ``[0, 1]`` based on cone angle from ``h_proj`` and
    inverse-square distance falloff.  Returns 0 if occluded or outside the
    cone.

    Parameters
    ----------
    hit_pos : np.ndarray
        3D position of the hit pixel.
    p_src : np.ndarray
        Flashlight source position (above current face).
    h_proj : np.ndarray
        Flashlight heading direction (unit vector, face-plane projection).
    cos_tmax : float
        Cosine of the flashlight cone half-angle.
    v0s, v1s, v2s : np.ndarray
        Stacked triangle vertices, shape ``(T, 3)`` each.
    curr_idx : int
        Index of current cone in stacked arrays (skipped in occlusion test).
    """
    dv   = hit_pos - p_src
    dist = float(np.linalg.norm(dv))
    if dist < 1e-12:
        return 1.0
    cos_a = float(np.dot(dv / dist, h_proj))
    if cos_a <= cos_tmax:
        return 0.0
    if _shadow_blocked(p_src, hit_pos, v0s, v1s, v2s):
        return 0.0
    fl = (cos_a - cos_tmax) / (1.0 - cos_tmax)
    return fl * fl * fl / (1.0 + 20.0 * dist * dist)


def _compute_brightness(
    hit_pos: np.ndarray,
    face_normal: np.ndarray,
    hit_idx: int,
    curr_idx: int,
    color_mode: int,
    r_max: float,
    sun_pos: np.ndarray | None,
    sphere_mode: bool,
    flashlight: bool,
    p_src: np.ndarray | None,
    h_proj: np.ndarray | None,
    cos_tmax: float,
    v0s: np.ndarray,
    v1s: np.ndarray,
    v2s: np.ndarray,
    fl_v0s: np.ndarray,
    fl_v1s: np.ndarray,
    fl_v2s: np.ndarray,
) -> float:
    """Compute pixel brightness in [0, 1] given hit geometry and lighting mode.

    Parameters
    ----------
    hit_pos : np.ndarray
        3D position of the hit point.
    face_normal : np.ndarray
        Outward unit normal of the hit cone's face.
    hit_idx : int
        Index of the hit cone in stacked arrays (skipped in sun shadow test).
    curr_idx : int
        Index of current cone in stacked arrays (skipped in flashlight test).
    color_mode : int
        0 = wireframe (not called), 1 = radius, 2 = sun.
    r_max : float
        Maximum vector magnitude (used for radius normalisation).
    sun_pos : np.ndarray or None
        Sun position in scene coordinates.
    sphere_mode : bool
        If True, skip sun occlusion (all faces visible on a convex sphere).
    flashlight : bool
        Whether the flashlight is active.
    p_src : np.ndarray or None
        Flashlight source position.
    h_proj : np.ndarray or None
        Flashlight heading direction.
    cos_tmax : float
        Cosine of flashlight cone half-angle.
    v0s, v1s, v2s : np.ndarray
        Sun shadow test triangle arrays (sun-facing subset), shape ``(T, 3)``.
    fl_v0s, fl_v1s, fl_v2s : np.ndarray
        Flashlight shadow test triangle arrays (all triangles), shape ``(T, 3)``.

    Returns
    -------
    float
        Brightness value in [0, 1].
    """
    if color_mode == 1:
        brt = max(0.0, min(1.0,
                           float(np.linalg.norm(hit_pos)) / r_max * _DIM_LEVEL))
    elif color_mode == 2 and sun_pos is not None:
        to_sun = sun_pos - hit_pos
        dist   = float(np.linalg.norm(to_sun))
        if dist < 1e-12:
            brt = _SUN_AMBIENT
        else:
            lam = max(0.0, float(np.dot(face_normal, to_sun / dist)))
            if not sphere_mode and _shadow_blocked(
                    hit_pos, sun_pos, v0s, v1s, v2s, skip_idx=hit_idx):
                brt = _SUN_AMBIENT
            else:
                brt = min(_SUN_MAX,
                          _SUN_AMBIENT + (1.0 - _SUN_AMBIENT)
                          * lam * _SUN_BRIGHTNESS / (dist * dist))
    else:
        brt = _DIM_LEVEL

    if flashlight and p_src is not None and h_proj is not None:
        fl_b = _fl_brightness_pixel(
            hit_pos, p_src, h_proj, cos_tmax, fl_v0s, fl_v1s, fl_v2s, curr_idx,
        )
        if color_mode == 2:
            brt = max(0.0, min(1.0, brt + fl_b * 0.55))
        else:
            brt = max(0.0, min(1.0, _DIM_LEVEL + fl_b * _FL_BOOST))

    return brt


# ---------------------------------------------------------------------------
# JIT shadow kernel
# ---------------------------------------------------------------------------

@numba.njit(parallel=True, cache=True)
def _shadow_blocked_all(
    hit_pos:  np.ndarray,
    target:   np.ndarray,
    v0s:      np.ndarray,
    v1s:      np.ndarray,
    v2s:      np.ndarray,
    skip_idx: np.ndarray,
    eps:      float = 1e-3,
) -> np.ndarray:
    """Shadow test for all hit pixels at once via Möller–Trumbore.

    Parameters
    ----------
    hit_pos  : (N, 3)  surface hit positions
    target   : (3,)    light source position (sun or flashlight)
    v0s, v1s, v2s : (T, 3)  triangle vertices
    skip_idx : (N,)  int32  triangle index to skip per pixel (-1 = skip none)
    eps      : float  nudge along shadow ray to avoid self-intersection

    Returns
    -------
    shadowed : (N,) bool  — True if the path to the light is blocked
    """
    N = hit_pos.shape[0]
    T = v0s.shape[0]
    shadowed = np.zeros(N, numba.boolean)

    for i in numba.prange(N):
        dx = target[0] - hit_pos[i, 0]
        dy = target[1] - hit_pos[i, 1]
        dz = target[2] - hit_pos[i, 2]
        dist = (dx*dx + dy*dy + dz*dz) ** 0.5
        if dist < 1e-12:
            continue
        inv = 1.0 / dist
        dx *= inv; dy *= inv; dz *= inv

        ox = hit_pos[i, 0] + eps * dx
        oy = hit_pos[i, 1] + eps * dy
        oz = hit_pos[i, 2] + eps * dz

        for k in range(T):
            if k == skip_idx[i]:
                continue

            e1x = v1s[k,0] - v0s[k,0]
            e1y = v1s[k,1] - v0s[k,1]
            e1z = v1s[k,2] - v0s[k,2]
            e2x = v2s[k,0] - v0s[k,0]
            e2y = v2s[k,1] - v0s[k,1]
            e2z = v2s[k,2] - v0s[k,2]

            hx = dy*e2z - dz*e2y
            hy = dz*e2x - dx*e2z
            hz = dx*e2y - dy*e2x
            a  = e1x*hx + e1y*hy + e1z*hz
            if abs(a) < 1e-8:
                continue

            f  = 1.0 / a
            sx = ox - v0s[k,0]
            sy = oy - v0s[k,1]
            sz = oz - v0s[k,2]
            u  = f * (sx*hx + sy*hy + sz*hz)
            if u < 0.0 or u > 1.0:
                continue

            qx = sy*e1z - sz*e1y
            qy = sz*e1x - sx*e1z
            qz = sx*e1y - sy*e1x
            vv = f * (dx*qx + dy*qy + dz*qz)
            if vv < 0.0 or u + vv > 1.0:
                continue

            t = f * (e2x*qx + e2y*qy + e2z*qz)
            if t > 1e-6 and t < dist - eps:
                shadowed[i] = True
                break

    return shadowed


# ---------------------------------------------------------------------------
# JIT pixel-fill kernel
# ---------------------------------------------------------------------------

@numba.njit(parallel=True, cache=True)
def _hit_pixels_numba(
    all_pix: np.ndarray,
    N_mat:   np.ndarray,
    c_vec:   np.ndarray,
    H_mat:   np.ndarray,
    d:       np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the closest cone hit for every pixel ray.

    Parameters
    ----------
    all_pix : (N, 3)   ray origins (screen pixel positions)
    N_mat   : (K, 3)   facet plane normals (front-facing cones only)
    c_vec   : (K,)     plane offsets  dot(n_k, v0_k)
    H_mat   : (K, 3, 3) halfspace matrices — rows are oriented edge normals
    d       : (3,)     shared ray direction (-p)

    Returns
    -------
    best_t  : (N,) float64  — closest hit distance, inf if no hit
    hit_idx : (N,) int32    — index into the K front-facing cones, -1 if no hit
    """
    N = all_pix.shape[0]
    K = N_mat.shape[0]

    # Per-cone constants (shared across all pixels)
    ndotd    = np.empty(K)
    d_scores = np.empty((K, 3))
    for k in range(K):
        s = 0.0
        for j in range(3):
            s += N_mat[k, j] * d[j]
        ndotd[k] = s
        for h in range(3):
            s2 = 0.0
            for j in range(3):
                s2 += H_mat[k, h, j] * d[j]
            d_scores[k, h] = s2

    best_t  = np.full(N, np.inf)
    hit_idx = np.full(N, -1, numba.int32)

    for i in numba.prange(N):
        for k in range(K):
            if abs(ndotd[k]) < 1e-12:
                continue
            orig_dot_n = 0.0
            for j in range(3):
                orig_dot_n += N_mat[k, j] * all_pix[i, j]
            t = (c_vec[k] - orig_dot_n) / ndotd[k]
            if t <= 1e-6:
                continue
            inside = True
            for h in range(3):
                score = d_scores[k, h] * t
                for j in range(3):
                    score += H_mat[k, h, j] * all_pix[i, j]
                if score < -1e-9:
                    inside = False
                    break
            if inside and t < best_t[i]:
                best_t[i]  = t
                hit_idx[i] = k

    return best_t, hit_idx


# ---------------------------------------------------------------------------
# Renderer class
# ---------------------------------------------------------------------------

class Renderer:
    """Curses-based backward renderer for a fan and player position on S².

    For each screen pixel a line is fired inward along -p; the first cone face
    hit is shaded and written to the terminal.  Edges are drawn on top via
    Bresenham lines (flat mode) or SLERP great-circle arcs (sphere mode).

    Parameters
    ----------
    fan : regfans.Fan
        The fan whose cones will be drawn.
    stdscr : _CursesWindow
        A curses window (full screen).
    """

    def __init__(self, fan: Fan, stdscr: _CursesWindow) -> None:
        self._fan      = fan
        self._stdscr   = stdscr
        self._edge_map = _cone_edge_map(fan)
        _init_colors(self)

    def draw(
        self,
        player_pos: np.ndarray,
        player_heading: np.ndarray,
        current_cone: tuple[int, ...],
        pointed_facet: tuple[int, int] | None = None,
        locked: bool = False,
        allow_deletion: bool = True,
        color_mode:   int   = 0,
        view_scale:   float = 1.0,
        flip_status:  dict  | None = None,
        is_irregular: bool  = False,
        sphere_mode:  bool  = False,
        agent_active: bool  = False,
        sun_angle:    float = 0.0,
        flashlight:   bool  = False,
        symbol_mode:  int   = 0,
        pixel_debug:  bool  = False,
    ) -> list[str] | None:
        """Render one frame.

        Returns a list of debug lines if ``pixel_debug=True`` and flat mode is
        active, otherwise None.
        """

        # ── setup ────────────────────────────────────────────────────────────
        scr = self._stdscr
        scr.bkgd(' ', curses.color_pair(_IREG_BG_PAIR) if is_irregular else 0)
        scr.erase()
        rows, cols = scr.getmaxyx()
        cy, cx = rows // 2, cols // 2
        scale  = float(min(rows, cols // 2) // 2 - 2) * 0.75 * view_scale
        if sphere_mode:
            scale = float(max(1, rows // 2 - _HUD_ROWS - 1))

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

        # ── cone data ────────────────────────────────────────────────────────
        ray_cache: dict[int, np.ndarray] = {}

        def ray(label: int) -> np.ndarray:
            if label not in ray_cache:
                v = fan.vectors(which=(label,))[0]
                if sphere_mode:
                    n = float(np.linalg.norm(v))
                    if n > 1e-12:
                        v = v / n
                ray_cache[label] = v
            return ray_cache[label]

        all_cones_list: list[tuple[int, ...]] = []
        cone_normals:   dict[tuple, np.ndarray] = {}
        cone_verts:     dict[tuple, list]       = {}   # {ct: [v0, v1, v2]}
        front_cones:    set[tuple[int, ...]]    = set()

        for cone in fan.cones():
            clabels = list(cone)
            ct      = tuple(sorted(clabels))
            all_cones_list.append(ct)
            vs = [ray(l) for l in clabels]
            cone_verts[ct] = vs
            n  = np.cross(vs[1] - vs[0], vs[2] - vs[0])
            n  = _orient_normal(n, vs[0])
            nn = float(np.linalg.norm(n))
            if nn > 1e-12:
                n = n / nn
            cone_normals[ct] = n
            if float(np.dot(n, view_dir)) > 0:
                front_cones.add(ct)

        # Precompute stacked triangle vertex arrays for vectorised shadow tests.
        _ct_to_idx: dict[tuple, int] = {ct: i for i, ct in enumerate(all_cones_list)}
        _all_v0s = np.array([cone_verts[ct][0] for ct in all_cones_list])
        _all_v1s = np.array([cone_verts[ct][1] for ct in all_cones_list])
        _all_v2s = np.array([cone_verts[ct][2] for ct in all_cones_list])

        # Precompute front-facing cone arrays for the numba pixel-fill kernel.
        _front_full_idx = np.array(
            [i for i, ct in enumerate(all_cones_list)
             if float(np.dot(cone_normals[ct], p)) > 0.0],
            dtype=np.int32,
        )
        _fl_N_mat = np.array([cone_normals[all_cones_list[i]] for i in _front_full_idx])
        _fl_c_vec = np.array([
            float(np.dot(cone_normals[all_cones_list[i]],
                         cone_verts[all_cones_list[i]][0]))
            for i in _front_full_idx
        ])
        _fl_H_list = []
        for i in _front_full_idx:
            r0, r1, r2 = cone_verts[all_cones_list[i]]
            h01 = np.cross(r0, r1); h01 = h01 if np.dot(h01, r2) >= 0 else -h01
            h12 = np.cross(r1, r2); h12 = h12 if np.dot(h12, r0) >= 0 else -h12
            h20 = np.cross(r2, r0); h20 = h20 if np.dot(h20, r1) >= 0 else -h20
            _fl_H_list.append(np.stack([h01, h12, h20]))
        _fl_H_mat = np.array(_fl_H_list) if _fl_H_list else np.zeros((0, 3, 3))

        # ── camera: player on face surface ───────────────────────────────────
        curr_ct     = tuple(sorted(current_cone))
        curr_vs     = cone_verts.get(curr_ct, [np.array([1.,0.,0.])]*3)
        curr_normal = cone_normals.get(curr_ct, p)
        p_surface   = _compute_p_surface(p, curr_vs[0], curr_normal)
        _max_vec_norm = float(np.linalg.norm(fan.vectors(), axis=1).max()) or 1.0
        screen_center = (_max_vec_norm + 1e-6) * p

        # ── mode precompute ───────────────────────────────────────────────────
        r_max: float = 1.0
        if color_mode == 1:
            r_max = float(np.linalg.norm(fan.vectors(), axis=1).max()) or 1.0

        sun_pos_cur: np.ndarray | None = None
        # Sun-facing triangle subset (used for sun shadow tests).
        # Default to full arrays; overridden below when color_mode==2.
        _sun_v0s, _sun_v1s, _sun_v2s = _all_v0s, _all_v1s, _all_v2s
        _sun_sub_idx: dict[int, int] = {}  # full_idx -> sun_subset_idx
        if color_mode == 2:
            _sc = float(np.cos(sun_angle))
            _ss = float(np.sin(sun_angle))
            sun_pos_cur = np.array([
                _sc * _SUN_POS[0] - _ss * _SUN_POS[1],
                _ss * _SUN_POS[0] + _sc * _SUN_POS[1],
                _SUN_POS[2],
            ])
            _sun_dir = sun_pos_cur / (float(np.linalg.norm(sun_pos_cur)) or 1.0)
            _cone_normals_arr = np.array(
                [cone_normals[ct] for ct in all_cones_list]
            )
            _sun_mask = (_cone_normals_arr @ _sun_dir) > 0
            _sun_v0s  = _all_v0s[_sun_mask]
            _sun_v1s  = _all_v1s[_sun_mask]
            _sun_v2s  = _all_v2s[_sun_mask]
            _sun_sub_idx = {
                int(full_i): sub_i
                for sub_i, full_i in enumerate(np.where(_sun_mask)[0])
            }

        p_src:    np.ndarray | None = None
        h_proj:   np.ndarray | None = None
        cos_tmax: float             = 0.0
        if flashlight:
            p_src    = p_surface + _M3_HEIGHT * curr_normal
            cos_tmax = float(np.cos(np.radians(_M3_THETA_MAX)))
            # Project heading onto face plane for the 3D cone gate.
            h_face_raw  = e1_new - float(np.dot(e1_new, curr_normal)) * curr_normal
            h_face_norm = float(np.linalg.norm(h_face_raw))
            h_proj      = h_face_raw / h_face_norm if h_face_norm > 1e-12 else e1_new

        # ── per-pixel fill pass ───────────────────────────────────────────────
        _pdbg_lines: list[str] | None = None
        if color_mode != 0:
            _sym_ramp = _SYMBOL_STYLES[symbol_mode % len(_SYMBOL_STYLES)][1]
            _n_r      = self._n_radius
            dir_vec   = -p   # ray direction for every pixel

            if sphere_mode:
                # Precompute oriented great-circle edge normals per cone.
                c_arr: np.ndarray = np.arange(cols - 1, dtype=float)
                cone_list: list[tuple] = []
                nAB_list, nBC_list, nCA_list = [], [], []
                for ct in all_cones_list:
                    vs   = cone_verts[ct]
                    nAB  = np.cross(vs[0], vs[1])
                    nBC  = np.cross(vs[1], vs[2])
                    nCA  = np.cross(vs[2], vs[0])
                    if float(np.dot(nAB, vs[0] + vs[1] + vs[2])) < 0.0:
                        nAB, nBC, nCA = -nAB, -nBC, -nCA
                    cone_list.append(ct)
                    nAB_list.append(nAB)
                    nBC_list.append(nBC)
                    nCA_list.append(nCA)
                nAB_mat = np.array(nAB_list)
                nBC_mat = np.array(nBC_list)
                nCA_mat = np.array(nCA_list)

                for r in range(rows - _HUD_ROWS):
                    pixel_row = _pixel_row_positions(
                        r, c_arr, screen_center, e1_new, e2_new, scale, cx, cy,
                    )
                    t_vals = _sphere_row_hits(pixel_row, p)
                    valid  = np.isfinite(t_vals)
                    if not np.any(valid):
                        continue
                    hit_pts = pixel_row - t_vals[:, np.newaxis] * p[np.newaxis, :]
                    norms   = np.linalg.norm(hit_pts, axis=1, keepdims=True)
                    hit_pts = np.where(norms > 1e-12, hit_pts / norms, hit_pts)
                    tol = -1e-9
                    in_AB  = (hit_pts @ nAB_mat.T) >= tol
                    in_BC  = (hit_pts @ nBC_mat.T) >= tol
                    in_CA  = (hit_pts @ nCA_mat.T) >= tol
                    inside = in_AB & in_BC & in_CA
                    inside[~valid] = False
                    has_hit  = np.any(inside, axis=1)
                    cone_idx = np.where(has_hit, np.argmax(inside, axis=1), -1)
                    for i in np.where(has_hit)[0]:
                        ct      = cone_list[cone_idx[i]]
                        hit_pos = hit_pts[i]
                        face_n  = cone_normals[ct]
                        brt = _compute_brightness(
                            hit_pos, face_n,
                            _sun_sub_idx.get(_ct_to_idx[ct], -1),
                            _ct_to_idx[curr_ct],
                            color_mode, r_max, sun_pos_cur,
                            True,
                            flashlight, p_src, h_proj, cos_tmax,
                            _sun_v0s, _sun_v1s, _sun_v2s,
                            _all_v0s, _all_v1s, _all_v2s,
                        )
                        t_n   = max(0.0, min(1.0, brt))
                        s_idx = round(t_n * (len(_sym_ramp) - 1))
                        ch    = _sym_ramp[max(0, min(len(_sym_ramp) - 1, s_idx))]
                        pair  = _RADIUS_PAIR_START + round(t_n * (_n_r - 1))
                        _addstr(scr, r, int(c_arr[i]),
                                ch, curses.color_pair(pair) | curses.A_BOLD)

            else:
                # Flat mode: build all pixel positions at once, then one cone
                # loop over all R*C pixels (replaces 38 row iterations × N cone
                # iterations with N cone iterations over all pixels together).
                _R = rows - _HUD_ROWS
                _C = cols - 1
                _r_arr = np.arange(_R, dtype=float)
                _c_arr = np.arange(_C, dtype=float)
                _s_arr = (cy - _r_arr) / scale
                _u_arr = (_c_arr - cx) / (scale * 2.0)
                # (R, 3) row centres, then broadcast to (R, C, 3) → (R*C, 3)
                _row_centers = screen_center + _s_arr[:, None] * e1_new[None, :]
                _all_pix = (
                    _row_centers[:, None, :]
                    + _u_arr[None, :, None] * e2_new[None, None, :]
                ).reshape(-1, 3)

                _best_t, _hit_front = _hit_pixels_numba(
                    _all_pix, _fl_N_mat, _fl_c_vec, _fl_H_mat, dir_vec,
                )
                _neg_hits = np.isfinite(_best_t) & (_best_t < 0)
                if np.any(_neg_hits):
                    raise RuntimeError(
                        f"_hit_pixels_numba returned {int(_neg_hits.sum())} negative-t "
                        f"hits (min={float(_best_t[_neg_hits].min()):.4f}). "
                        "Screen plane is inside the polytope — increase screen distance."
                    )
                # map front-facing indices back to all_cones_list indices
                _hit_idx = np.full(_R * _C, -1, dtype=np.int32)
                _valid_hits = _hit_front >= 0
                _hit_idx[_valid_hits] = _front_full_idx[_hit_front[_valid_hits]]

                # ── pixel debug ──────────────────────────────────────────────
                if pixel_debug:
                    _PCHARS = (
                        "0123456789"
                        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                        "abcdefghijklmnopqrstuvwxyz"
                        "+!@#$%^&*~"
                    )
                    _hm = _hit_idx.reshape(_R, _C)
                    _pdbg_lines = []
                    _pdbg_lines.append(
                        "── PIXEL HIT MAP (char=face, ·=miss) ──"
                    )
                    for _pr in range(0, _R, 2):
                        _pdbg_lines.append("".join(
                            "·" if _hm[_pr, _pc] < 0
                            else _PCHARS[_hm[_pr, _pc] % len(_PCHARS)]
                            for _pc in range(_C)
                        ))
                    _n_miss = int((_hm < 0).sum())
                    _n_hit  = _R * _C - _n_miss
                    _pdbg_lines.append(f"")
                    _pdbg_lines.append(
                        f"Hit: {_n_hit}/{_R*_C}  Miss: {_n_miss}"
                        f"  front-facing cones: {len(_front_full_idx)}"
                    )
                    # Interior miss pixels (within bounding box of hits).
                    if _n_miss > 0 and _n_hit > 0:
                        _hr = np.where(np.any(_hm >= 0, axis=1))[0]
                        _hc = np.where(np.any(_hm >= 0, axis=0))[0]
                        if len(_hr) and len(_hc):
                            _rlo, _rhi = int(_hr[0]),  int(_hr[-1])
                            _clo, _chi = int(_hc[0]),  int(_hc[-1])
                            _ri = np.arange(_R)[:, None]
                            _ci = np.arange(_C)[None, :]
                            _int_miss = np.argwhere(
                                (_hm < 0)
                                & (_ri >= _rlo) & (_ri <= _rhi)
                                & (_ci >= _clo) & (_ci <= _chi)
                            )
                            _pdbg_lines.append(
                                f"Interior miss "
                                f"(bbox r={_rlo}-{_rhi} c={_clo}-{_chi}): "
                                f"{len(_int_miss)}"
                            )
                            # Near-miss details for up to 10 interior misses.
                            _Kf = len(_front_full_idx)
                            for _mr, _mc in _int_miss[:10]:
                                _mi   = int(_mr) * _C + int(_mc)
                                _orig = _all_pix[_mi]
                                _near: list = []
                                for _fk in range(_Kf):
                                    _nd = float(
                                        _fl_N_mat[_fk, 0] * dir_vec[0]
                                        + _fl_N_mat[_fk, 1] * dir_vec[1]
                                        + _fl_N_mat[_fk, 2] * dir_vec[2]
                                    )
                                    if abs(_nd) < 1e-12:
                                        continue
                                    _od = float(
                                        _fl_N_mat[_fk, 0] * _orig[0]
                                        + _fl_N_mat[_fk, 1] * _orig[1]
                                        + _fl_N_mat[_fk, 2] * _orig[2]
                                    )
                                    _tt = (float(_fl_c_vec[_fk]) - _od) / _nd
                                    if _tt <= 1e-6:
                                        continue
                                    _hp = _orig + _tt * dir_vec
                                    _sc = [
                                        float(
                                            _fl_H_mat[_fk, _h, 0] * _hp[0]
                                            + _fl_H_mat[_fk, _h, 1] * _hp[1]
                                            + _fl_H_mat[_fk, _h, 2] * _hp[2]
                                        )
                                        for _h in range(3)
                                    ]
                                    _near.append((
                                        min(_sc), _tt,
                                        all_cones_list[int(_front_full_idx[_fk])],
                                        _sc,
                                    ))
                                _near.sort(key=lambda x: -x[0])
                                _pdbg_lines.append(
                                    f"  Miss r={_mr} c={_mc}:"
                                )
                                for _ms, _tt2, _fct, _scs in _near[:5]:
                                    _pdbg_lines.append(
                                        f"    {str(_fct):<22} t={_tt2:.4f}"
                                        f"  min_score={_ms:.3e}"
                                        f"  [{_scs[0]:.3e},{_scs[1]:.3e},{_scs[2]:.3e}]"
                                    )

                _hit_flat = np.where(_hit_idx >= 0)[0]
                if len(_hit_flat):
                    _hit_pix  = _all_pix[_hit_flat]
                    _hit_pos  = _hit_pix + _best_t[_hit_flat, None] * dir_vec
                    _hit_ci   = _hit_idx[_hit_flat]
                    _rows_out = (_hit_flat // _C).astype(int)
                    _cols_out = (_hit_flat  % _C).astype(int)
                    _n_ramp   = len(_sym_ramp)

                    if color_mode == 1 and not flashlight:
                        # Radius mode, no lighting: vectorise brightness fully.
                        _brts  = np.clip(
                            np.linalg.norm(_hit_pos, axis=1) / r_max * _DIM_LEVEL,
                            0.0, 1.0,
                        )
                        _sidxs = np.clip(
                            np.round(_brts * (_n_ramp - 1)).astype(int),
                            0, _n_ramp - 1,
                        )
                        _pairs = (
                            _RADIUS_PAIR_START
                            + np.round(_brts * (_n_r - 1)).astype(int)
                        )
                        for k in range(len(_hit_flat)):
                            _addstr(scr, _rows_out[k], _cols_out[k],
                                    _sym_ramp[_sidxs[k]],
                                    curses.color_pair(int(_pairs[k])) | curses.A_BOLD)
                    elif color_mode == 2 and sun_pos_cur is not None and not flashlight:
                        # Sun mode, no flashlight: batch shadow test then
                        # vectorise brightness over all lit pixels.
                        _NP = len(_hit_flat)
                        _skip = np.array(
                            [_sun_sub_idx.get(int(_hit_ci[k]), -1)
                             for k in range(_NP)],
                            dtype=np.int32,
                        )
                        _shadowed = _shadow_blocked_all(
                            _hit_pos, sun_pos_cur,
                            _sun_v0s, _sun_v1s, _sun_v2s, _skip,
                        )
                        _face_ns = np.array(
                            [cone_normals[all_cones_list[_hit_ci[k]]]
                             for k in range(_NP)]
                        )
                        _to_sun  = sun_pos_cur - _hit_pos          # (NP, 3)
                        _dists   = np.linalg.norm(_to_sun, axis=1) # (NP,)
                        _to_sun_u = _to_sun / np.maximum(_dists[:, None], 1e-12)
                        _lam = np.maximum(0.0,
                            np.einsum('ni,ni->n', _face_ns, _to_sun_u))
                        _lit_brt = np.minimum(
                            _SUN_MAX,
                            _SUN_AMBIENT + (1.0 - _SUN_AMBIENT)
                            * _lam * _SUN_BRIGHTNESS / np.maximum(_dists**2, 1e-12)
                        )
                        _brts = np.where(_shadowed, _SUN_AMBIENT, _lit_brt)
                        _brts = np.clip(_brts, 0.0, 1.0)
                        _sidxs = np.clip(
                            np.round(_brts * (_n_ramp - 1)).astype(int),
                            0, _n_ramp - 1,
                        )
                        _pairs = (
                            _RADIUS_PAIR_START
                            + np.round(_brts * (_n_r - 1)).astype(int)
                        )
                        for k in range(_NP):
                            _addstr(scr, _rows_out[k], _cols_out[k],
                                    _sym_ramp[_sidxs[k]],
                                    curses.color_pair(int(_pairs[k])) | curses.A_BOLD)
                    else:
                        # Remaining cases: flashlight on, and/or color_mode==1
                        # with flashlight.  Vectorise flashlight brightness then
                        # combine with base (radius or sun).
                        _NP = len(_hit_flat)

                        # ── base brightness ───────────────────────────────────
                        if color_mode == 1:
                            _brts = np.clip(
                                np.linalg.norm(_hit_pos, axis=1) / r_max * _DIM_LEVEL,
                                0.0, 1.0,
                            )
                        elif color_mode == 2 and sun_pos_cur is not None:
                            _skip_sun = np.array(
                                [_sun_sub_idx.get(int(_hit_ci[k]), -1)
                                 for k in range(_NP)],
                                dtype=np.int32,
                            )
                            _shadowed_sun = _shadow_blocked_all(
                                _hit_pos, sun_pos_cur,
                                _sun_v0s, _sun_v1s, _sun_v2s, _skip_sun,
                            )
                            _face_ns = np.array(
                                [cone_normals[all_cones_list[_hit_ci[k]]]
                                 for k in range(_NP)]
                            )
                            _to_sun  = sun_pos_cur - _hit_pos
                            _dists_s = np.linalg.norm(_to_sun, axis=1)
                            _to_sun_u = _to_sun / np.maximum(_dists_s[:, None], 1e-12)
                            _lam = np.maximum(
                                0.0, np.einsum('ni,ni->n', _face_ns, _to_sun_u))
                            _lit = np.minimum(
                                _SUN_MAX,
                                _SUN_AMBIENT + (1.0 - _SUN_AMBIENT)
                                * _lam * _SUN_BRIGHTNESS
                                / np.maximum(_dists_s**2, 1e-12),
                            )
                            _brts = np.where(_shadowed_sun, _SUN_AMBIENT, _lit)
                        else:
                            _brts = np.full(_NP, _DIM_LEVEL)

                        # ── flashlight contribution ───────────────────────────
                        if flashlight and p_src is not None and h_proj is not None:
                            _dv      = _hit_pos - p_src             # (NP, 3)
                            _dists_f = np.linalg.norm(_dv, axis=1)  # (NP,)
                            _dv_u    = _dv / np.maximum(_dists_f[:, None], 1e-12)
                            _cos_a   = _dv_u @ h_proj               # (NP,)
                            _in_cone = _cos_a > cos_tmax
                            _skip_fl = np.full(_NP, -1, dtype=np.int32)
                            _shad_fl = _shadow_blocked_all(
                                _hit_pos, p_src,
                                _all_v0s, _all_v1s, _all_v2s, _skip_fl,
                            )
                            _fl_raw  = ((_cos_a - cos_tmax)
                                        / max(1.0 - cos_tmax, 1e-12))
                            _fl_b    = (np.where(_in_cone & ~_shad_fl,
                                                 _fl_raw**3
                                                 / (1.0 + 20.0 * _dists_f**2),
                                                 0.0))
                            if color_mode == 2:
                                _brts = np.clip(_brts + _fl_b * 0.55, 0.0, 1.0)
                            else:
                                _brts = np.clip(_DIM_LEVEL + _fl_b * _FL_BOOST,
                                                0.0, 1.0)

                        _brts  = np.clip(_brts, 0.0, 1.0)
                        _sidxs = np.clip(
                            np.round(_brts * (_n_ramp - 1)).astype(int),
                            0, _n_ramp - 1,
                        )
                        _pairs = (
                            _RADIUS_PAIR_START
                            + np.round(_brts * (_n_r - 1)).astype(int)
                        )
                        for k in range(_NP):
                            _addstr(scr, _rows_out[k], _cols_out[k],
                                    _sym_ramp[_sidxs[k]],
                                    curses.color_pair(int(_pairs[k])) | curses.A_BOLD)

        # ── screen_pt and _draw_edge closures (used by edge pass) ────────────
        def screen_pt(label: int) -> tuple[int, int] | None:
            coord = _project(ray(label), view_dir, e1_new, e2_new)
            if coord is None:
                return None
            # Columns scaled by 2× to compensate for terminal character cells
            # being ~2× taller than wide, preserving circular aspect ratio.
            col = cx + int(round(coord[0] * scale * 2))
            row = cy - int(round(coord[1] * scale))
            return (row, col)

        def _draw_edge(a: int, b: int, ch: str, attr: int) -> None:
            if not sphere_mode:
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
                return
            # Sphere mode: trace the great circle arc via SLERP.
            u = ray(a)
            v = ray(b)
            cos_a   = float(np.clip(np.dot(u, v), -1.0, 1.0))
            theta   = float(np.arccos(cos_a))
            n_steps = max(2, int(theta / _SLERP_STEP))
            sin_th  = float(np.sin(theta))
            prev: tuple[int, int] | None = None
            for i in range(n_steps + 1):
                t = i / n_steps
                w = (
                    (np.sin((1.0 - t) * theta) / sin_th) * u
                    + (np.sin(t * theta) / sin_th) * v
                ) if sin_th > 1e-9 else u
                if float(np.dot(w, view_dir)) < 0.0:
                    prev = None
                    continue
                coord = _project(w, view_dir, e1_new, e2_new)
                if coord is None:
                    prev = None
                    continue
                col_w = cx + int(round(coord[0] * scale * 2))
                row_w = cy - int(round(coord[1] * scale))
                if prev is not None:
                    _draw_line(scr, prev[0],     prev[1], row_w,     col_w, ch, attr)
                    _draw_line(scr, prev[0] + 1, prev[1], row_w + 1, col_w, ch, attr)
                prev = (row_w, col_w)

        # ── edge pass ────────────────────────────────────────────────────────
        if sphere_mode:
            sphere_front_edge: set[tuple[int, ...]] = set()
            for ct in all_cones_list:
                vs   = cone_verts[ct]
                dots = [float(np.dot(v, view_dir)) for v in vs]
                if any(d > 0 for d in dots):
                    sphere_front_edge.add(ct)

            _drawn_edges: set[tuple[int, int]] = set()
            for ct in sphere_front_edge:
                clabels = list(ct)
                for i in range(len(clabels)):
                    a, b = clabels[i], clabels[(i + 1) % len(clabels)]
                    edge = (min(a, b), max(a, b))
                    if edge in _drawn_edges:
                        continue
                    _drawn_edges.add(edge)
                    if (float(np.dot(ray(a), view_dir)) < 0 and
                            float(np.dot(ray(b), view_dir)) < 0):
                        continue
                    if pointed_facet and edge == pointed_facet:
                        continue
                    is_active = edge in active_edge_set
                    ch_e, attr_e = _edge_attrs(edge, is_active, flip_status)
                    _draw_edge(a, b, ch_e, attr_e)

        else:
            sorted_front = sorted(
                front_cones,
                key=lambda ct: float(
                    np.dot(np.mean([ray(l) for l in ct], axis=0), view_dir)
                ),
            )
            _drawn_edges_f: set[tuple[int, int]] = set()
            for ct in sorted_front:
                clabels = list(ct)
                pts = [screen_pt(l) for l in clabels]
                if any(pt is None for pt in pts):
                    continue
                for i in range(len(clabels)):
                    a, b = clabels[i], clabels[(i + 1) % len(clabels)]
                    edge = (min(a, b), max(a, b))
                    if edge in _drawn_edges_f:
                        continue
                    _drawn_edges_f.add(edge)
                    if pointed_facet and edge == pointed_facet:
                        continue
                    is_active = edge in active_edge_set
                    ch_e, attr_e = _edge_attrs(edge, is_active, flip_status)
                    _draw_edge(a, b, ch_e, attr_e)

        # ── pointed facet highlight ──────────────────────────────────────────
        if pointed_facet:
            a, b = pointed_facet
            if self._edge_map.get(pointed_facet, set()) & front_cones:
                if flip_status is not None and pointed_facet in active_edge_set:
                    flippable = flip_status.get(pointed_facet, False)
                    attr_e = (curses.color_pair(_EDGE_PAIR_BASE + (0 if flippable else 1))
                              | curses.A_BOLD)
                    _draw_edge(a, b, "*", attr_e)
                else:
                    _draw_edge(a, b, "*", curses.color_pair(5) | curses.A_BOLD)

        # ── player marker ────────────────────────────────────────────────────
        coord = _project(p, view_dir, e1_new, e2_new)
        if coord is not None:
            col = cx + int(round(coord[0] * scale * 2))
            row = cy - int(round(coord[1] * scale))
            attr = curses.color_pair(3) | curses.A_BOLD
            for dr, s in ((-1, "^^"), (0, "||")):
                r, c = row + dr, col
                if 0 <= r < rows - _HUD_ROWS and 0 <= c + 1 < cols - 1:
                    _addstr(scr, r, c, s, attr)

        # ── irregular banner ─────────────────────────────────────────────────
        if is_irregular:
            _ireg_lines = [
                "                                    ",
                "   I  R  R  E  G  U  L  A  R        ",
                "                                    ",
            ]
            _ireg_attr = curses.color_pair(_IREG_BG_PAIR) | curses.A_BOLD
            for _ii, _il in enumerate(_ireg_lines):
                if 0 <= _ii < rows - _HUD_ROWS:
                    _addstr(scr, _ii, 0, _il[: cols - 1], _ireg_attr)

        # ── HUD ──────────────────────────────────────────────────────────────
        facet_str  = str(pointed_facet) if pointed_facet else "none"
        tail       = "[q]uit"
        cone_str   = f"  cone={current_cone}"
        facet_row1 = f"  facet={facet_str}"
        agt_str    = "  [W]agent:ON" if agent_active else "  [W]agent:off"
        agt_attr   = (curses.color_pair(2) | curses.A_BOLD
                      if agent_active else curses.color_pair(4))
        sph_str    = "  [A]sphere:ON" if sphere_mode else "  [A]sphere:off"
        sph_attr   = (curses.color_pair(2) | curses.A_BOLD
                      if sphere_mode else curses.color_pair(4))
        del_str    = "  [D]del:ON" if allow_deletion else "  [D]del:off"
        del_attr   = (curses.color_pair(2) | curses.A_BOLD
                      if allow_deletion else curses.color_pair(4))
        lock_str   = "  [C]fix:ON" if locked else "  [C]fix:off"
        lock_attr  = (curses.color_pair(2) | curses.A_BOLD
                      if locked else curses.color_pair(4))
        col_str    = f"  [S]fill:{_COLOR_LABELS[color_mode]}"
        sym_str    = f"  [Z]sym:{_SYMBOL_STYLES[symbol_mode % len(_SYMBOL_STYLES)][0]}"
        lit_str    = "  [X]light:ON" if flashlight else "  [X]light:off"
        lit_attr   = (curses.color_pair(2) | curses.A_BOLD
                      if flashlight else curses.color_pair(4))

        # ── HUD row 0 (rows-2): [q]uit  cone=…  [A]sphere  [S]fill  [D]el  [W]agent  [F]dbg
        # ── HUD row 1 (rows-1):          facet=…  [Z]sym    [X]light [C]fix
        # Stacked key columns match physical keyboard columns (Q-A-Z, W-S-X, E-D-C).
        try:
            _blank = " " * (cols - 1)
            for _hr in range(_HUD_ROWS):
                scr.addstr(rows - _HUD_ROWS + _hr, 0, _blank, curses.color_pair(4))

            col = 0
            r0  = rows - _HUD_ROWS

            scr.addstr(r0, col, tail[: cols - 1], curses.color_pair(4))
            col += len(tail)

            # Column widths: max(row-0 item, row-1 item) so neither row clips.
            _cone_w = max(len(cone_str), len(facet_row1))
            _sph_w  = max(len(sph_str),  len(sym_str))
            _col_w  = max(len(col_str),  len(lit_str))
            _del_w  = max(len(del_str),  len(lock_str))

            cone_col = col
            if col < cols - 1:
                scr.addstr(r0, col, cone_str[: cols - 1 - col], curses.color_pair(4))
                col += _cone_w

            sph_col = col
            if col < cols - 1:
                scr.addstr(r0, col, sph_str[: cols - 1 - col], sph_attr)
                col += _sph_w

            col_col = col
            if col < cols - 1:
                scr.addstr(r0, col, col_str[: cols - 1 - col], curses.color_pair(4))
                col += _col_w

            del_col = col
            if col < cols - 1:
                scr.addstr(r0, col, del_str[: cols - 1 - col], del_attr)
                col += _del_w

            if col < cols - 1:
                scr.addstr(r0, col, agt_str[: cols - 1 - col], agt_attr)
                col += len(agt_str)

            if col < cols - 1:
                scr.addstr(r0, col, "  [F]dbg"[: cols - 1 - col],
                           curses.color_pair(4))

            r1 = rows - 1

            if cone_col < cols - 1:
                scr.addstr(r1, cone_col,
                           facet_row1[: sph_col - cone_col], curses.color_pair(4))
            if sph_col < cols - 1:
                scr.addstr(r1, sph_col,
                           sym_str[: col_col - sph_col], curses.color_pair(4))
            if col_col < cols - 1:
                scr.addstr(r1, col_col,
                           lit_str[: del_col - col_col], lit_attr)
            if del_col < cols - 1:
                scr.addstr(r1, del_col,
                           lock_str[: cols - 1 - del_col], lock_attr)

        except curses.error:
            pass

        return _pdbg_lines
