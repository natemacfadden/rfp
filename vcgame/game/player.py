"""
Player in R³ described by spherical coordinates (r, θ, φ), with heading in
the tangent plane of the angular part.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warnings

import numpy as np

if TYPE_CHECKING:
    from regfans import Fan, VectorConfiguration

_NORM_EPS = 1e-15


def _cartesian_to_angles(d: np.ndarray) -> tuple[float, float]:
    """Return (theta, phi) spherical angles for a unit vector d."""
    return (
        float(np.arccos(np.clip(d[2], -1.0, 1.0))),
        float(np.arctan2(d[1], d[0])),
    )


class Player:
    """
    A player in R³ with position stored as spherical coordinates.

    Position is stored as ``(r, θ, φ)`` and a unit heading in the tangent
    plane at the angular direction ``(θ, φ)``.

    Parameters
    ----------
    position : np.ndarray
        Non-zero 3-vector — direction used to compute ``θ, φ``.
    heading : np.ndarray
        Non-zero 3-vector projected onto the tangent plane at the angular
        direction.
    radius : float, optional
        Initial radial distance. Defaults to 1.0.
    height : float, optional
        Offset above the surface. Defaults to 0.05.

    Attributes
    ----------
    position : np.ndarray
        ``[r, θ, φ]`` — spherical coordinates.
    cartesian : np.ndarray
        ``[x, y, z]`` — Cartesian position, derived from ``position``.
    radius : float
        Positive scalar ``r`` — distance from the origin.
    height : float
        Small offset above the polytope surface.
    heading : np.ndarray
        Unit tangent vector at the angular direction (perpendicular to the
        radial direction).

    Raises
    ------
    ValueError
        If ``position`` is zero, ``heading`` is parallel to ``position``,
        or ``radius`` is not positive.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        position: np.ndarray,
        heading: np.ndarray,
        radius: float = 1.0,
        height: float = 0.05,
    ) -> None:
        position = np.asarray(position, dtype=float)
        heading  = np.asarray(heading,  dtype=float)

        if position.shape != (3,):
            raise ValueError(
                f"position must be a 3-vector, got shape {position.shape}"
            )
        if heading.shape != (3,):
            raise ValueError(
                f"heading must be a 3-vector, got shape {heading.shape}"
            )
        if radius <= 0.0:
            raise ValueError(f"radius must be positive, got {radius}")

        p_norm = np.linalg.norm(position)
        if p_norm < _NORM_EPS:
            raise ValueError("position must be non-zero")

        p_unit = position / p_norm
        theta, phi = _cartesian_to_angles(p_unit)
        self._position = np.array([float(radius), theta, phi])
        self._height   = float(height)

        h = heading - np.dot(heading, p_unit) * p_unit
        h_norm = np.linalg.norm(h)
        if h_norm < _NORM_EPS:
            raise ValueError(
                "heading has no component tangent to position "
                "(parallel or zero)"
            )
        self._heading = h / h_norm

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _direction(self) -> np.ndarray:
        """Unit vector corresponding to the angular part ``(θ, φ)``."""
        _, theta, phi = self._position
        st = np.sin(theta)
        return np.array([st * np.cos(phi), st * np.sin(phi), np.cos(theta)])

    @property
    def direction(self) -> np.ndarray:
        """Unit direction vector (angular part only, read-only copy)."""
        return self._direction.copy()

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def position(self) -> np.ndarray:
        """Spherical coordinates ``[r, θ, φ]`` (read-only copy)."""
        return self._position.copy()

    @property
    def cartesian(self) -> np.ndarray:
        """Cartesian position ``[x, y, z] = r * direction`` (read-only)."""
        return self._position[0] * self._direction

    @property
    def radius(self) -> float:
        """Radial distance from the origin."""
        return float(self._position[0])

    @radius.setter
    def radius(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError(f"radius must be positive, got {value}")
        self._position[0] = float(value)

    @property
    def height(self) -> float:
        """Offset above the polytope surface."""
        return self._height

    @height.setter
    def height(self, value: float) -> None:
        self._height = float(value)

    @property
    def heading(self) -> np.ndarray:
        """Unit heading tangent vector (read-only copy)."""
        return self._heading.copy()

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    def turn(self, angle: float) -> None:
        """Rotate the heading in the tangent plane. Positive turns left.

        Parameters
        ----------
        angle : float
            Rotation in radians.
        """
        d = self._direction
        self._heading = (
            np.cos(angle) * self._heading
            + np.sin(angle) * np.cross(d, self._heading)
        )

    def surface_radius(self, fan: Fan) -> float:
        """Radius where the ray through the angular direction hits the surface.

        Returns the radius at which the ray from the origin through the
        angular direction intersects the surface triangle of the current
        cone.

        Parameters
        ----------
        fan : regfans.Fan
            The fan defining the surface.

        Returns
        -------
        float
            Positive radial distance to the surface.
        """
        cone  = self.current_cone(fan)
        verts = fan.vectors(which=cone)   # shape (3, 3)
        v0, v1, v2 = verts[0], verts[1], verts[2]
        n     = np.cross(v1 - v0, v2 - v0)
        d     = self._direction
        n_norm = float(np.linalg.norm(n))
        denom  = float(np.dot(n, d))
        if n_norm < 1e-12 or abs(denom) < 1e-10:
            warnings.warn(
                f"surface_radius: degenerate cone {cone} "
                f"(|n|={n_norm:.2e}, n·d={denom:.2e}); "
                f"keeping previous radius"
            )
            return float(self._position[0])
        return float(np.dot(n, v0) / denom)

    def move(
        self, step: float, fan: Fan | None = None,
    ) -> tuple[int, int] | None:
        """Advance along the great circle by ``step`` radians.

        Updates ``(θ, φ)`` and, if ``fan`` is provided, sets
        ``r = surface_radius(fan) + height``.

        Parameters
        ----------
        step : float
            Arc length to advance in radians.
        fan : regfans.Fan or None, optional
            When provided, ``r`` is updated to track the surface.

        Returns
        -------
        tuple[int, int] or None
            Sorted label pair of the crossed facet, or ``None``.
        """
        old_cone = self.current_cone(fan) if fan is not None else None

        if fan is not None and old_cone is not None:
            verts = fan.vectors(which=old_cone)
            r0, r1, r2 = verts[0], verts[1], verts[2]
            h01 = np.cross(r0, r1); h01 = h01 if np.dot(h01, r2) >= 0 else -h01
            h12 = np.cross(r1, r2); h12 = h12 if np.dot(h12, r0) >= 0 else -h12
            h20 = np.cross(r2, r0); h20 = h20 if np.dot(h20, r1) >= 0 else -h20
            d0 = self._direction
            margins = [float(np.dot(h, d0)) for h in (h01, h12, h20)]
            norms   = [float(np.linalg.norm(h)) for h in (h01, h12, h20)]
            # Angular size of the smallest wall gap ~ margin / (|h| * |d|)
            ang_gaps = [
                m / n for m, n in zip(margins, norms) if n > 1e-12 and m > 0
            ]
            if ang_gaps and abs(step) > 0.5 * min(ang_gaps):
                warnings.warn(
                    f"move(): step={step:.4f} may skip a cone wall "
                    f"(smallest angular gap ~ {min(ang_gaps):.4f})"
                )

        d, h = self._direction, self._heading
        c, s = np.cos(step), np.sin(step)

        new_d = c * d + s * h
        new_h = -s * d + c * h

        new_d /= np.linalg.norm(new_d)
        new_h -= np.dot(new_h, new_d) * new_d
        new_h /= np.linalg.norm(new_h)

        # Store updated angular part back into spherical coordinates
        self._position[1], self._position[2] = _cartesian_to_angles(new_d)
        self._heading      = new_h

        if fan is None:
            return None

        self._position[0] = self.surface_radius(fan) + self._height

        new_cone = self.current_cone(fan)
        if new_cone == old_cone:
            return None
        shared = set(old_cone) & set(new_cone)
        if len(shared) == 2:
            return tuple(sorted(shared))
        return None

    # ------------------------------------------------------------------
    # Fan queries
    # ------------------------------------------------------------------

    def pointed_facet(self, fan: Fan) -> tuple[int, int] | None:
        """Return the facet the heading aims most directly toward.

        Parameters
        ----------
        fan : regfans.Fan
            The fan defining the cone structure.

        Returns
        -------
        tuple[int, int] or None
            Sorted label pair ``(min, max)``, or ``None``.
        """
        cone = self.current_cone(fan)
        i, j, k = cone
        facets = [(i, j, k), (j, k, i), (i, k, j)]

        best_facet: tuple[int, int] | None = None
        best_dot = 0.0

        for a, b, other in facets:
            v_a = fan.vectors(which=(a,))[0]
            v_b = fan.vectors(which=(b,))[0]
            v_c = fan.vectors(which=(other,))[0]
            n = np.cross(v_a, v_b)
            if np.dot(n, v_c) > 0:
                n = -n
            d = np.dot(self._heading, n)
            if d > best_dot:
                best_dot  = d
                best_facet = (min(a, b), max(a, b))

        return best_facet

    def current_cone(self, fan: Fan) -> tuple[int, ...]:
        """Return the label tuple of the cone containing the player's direction.

        Parameters
        ----------
        fan : regfans.Fan
            The fan to search.

        Returns
        -------
        tuple[int, ...]
            Label tuple of the containing cone.

        Raises
        ------
        ValueError
            If no cone contains the direction.
        """
        d = self._direction
        best_cone   = None
        best_margin = -np.inf  # min halfspace score across the three walls
        for cone in fan.cones():
            verts = fan.vectors(which=cone)
            r0, r1, r2 = verts[0], verts[1], verts[2]
            h01 = np.cross(r0, r1); h01 = h01 if np.dot(h01, r2) >= 0 else -h01
            h12 = np.cross(r1, r2); h12 = h12 if np.dot(h12, r0) >= 0 else -h12
            h20 = np.cross(r2, r0); h20 = h20 if np.dot(h20, r1) >= 0 else -h20
            margin = float(min(np.dot(h01, d), np.dot(h12, d), np.dot(h20, d)))
            if margin >= 0 or margin > best_margin:
                # Accept the cone with the best (largest) minimum margin;
                # a non-negative margin means d is strictly inside.
                best_margin = margin
                best_cone   = cone
                if margin >= 0:
                    return cone   # strict interior — no need to keep searching
        if best_cone is not None and best_margin > -1e-6:
            return best_cone      # on or near a wall — return closest cone
        raise ValueError("position is not contained in any cone of the fan")

    def find_circuit_for_crossing(
        self,
        old_cone: tuple[int, ...],
        new_cone: tuple[int, ...],
        fan: Fan,
    ) -> object | None:
        """Find the circuit whose support is the union of ``old_cone`` and
        ``new_cone``.

        Parameters
        ----------
        old_cone : tuple[int, ...]
            Label tuple of the cone before crossing.
        new_cone : tuple[int, ...]
            Label tuple of the cone after crossing.
        fan : regfans.Fan
            The fan containing the circuits.

        Returns
        -------
        object or None
            The matching circuit, or ``None`` if not found.
        """
        target = set(old_cone) | set(new_cone)
        for circ in fan.circuits():
            if set(circ.Z) == target:
                return circ
        return None

    def crossed_circuit(
        self,
        old_cone: tuple[int, ...],
        new_cone: tuple[int, ...],
        vc: VectorConfiguration,
    ) -> object | None:
        """Return the circuit for the wall crossing, or ``None`` if degenerate.

        Parameters
        ----------
        old_cone : tuple[int, ...]
            Label tuple of the cone before crossing.
        new_cone : tuple[int, ...]
            Label tuple of the cone after crossing.
        vc : regfans.VectorConfiguration
            The vector configuration to query.

        Returns
        -------
        object or None
            The circuit, or ``None`` if degenerate/coplanar.
        """
        shared = set(old_cone) & set(new_cone)
        c = (set(old_cone) - shared).pop()
        d = (set(new_cone) - shared).pop()
        circuit_labels = tuple(sorted(shared)) + (c, d)
        circ = vc.circuit(circuit_labels)
        if circ is None:
            warnings.warn(
                f"circuit({circuit_labels}) is None (degenerate/coplanar)"
            )
            return None
        return circ

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        r, theta, phi = self._position
        x, y, z = self.cartesian
        h = self._heading
        return (
            f"Player(r={r:.4f}, θ={theta:.4f}, φ={phi:.4f} | "
            f"xyz=[{x:.4f}, {y:.4f}, {z:.4f}] | "
            f"heading=[{h[0]:.4f}, {h[1]:.4f}, {h[2]:.4f}] | "
            f"height={self._height:.4f})"
        )
