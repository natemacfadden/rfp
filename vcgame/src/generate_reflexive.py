"""
Generate a vector configuration from the lattice points of a 3D
reflexive polytope.

Data source: http://coates.ma.ic.ac.uk/3DReflexivePolytopes/
There are 4319 polytopes, indexed 0–4318.

Each polytope page contains a "Integer points" table cell with a
3 × K matrix (3 coordinate rows, K lattice-point columns).  The
last column is always the origin, which is excluded.  All other
columns are used as vectors.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from urllib.error import URLError
from urllib.request import urlopen

from regfans import VectorConfiguration

if TYPE_CHECKING:
    from regfans import Fan

_BASE_URL    = "http://coates.ma.ic.ac.uk/3DReflexivePolytopes/{}.html"
N_POLYTOPES  = 4319   # polytope ids 0 … 4318


def _fetch_vectors(polytope_id: int) -> list[list[int]]:
    """Fetch non-origin lattice points of a polytope from the CCGGK database.

    Retrieves data for ``polytope_id`` from the
    Coates–Corti–Galkin–Golyshev–Kasprzyk database.

    Parameters
    ----------
    polytope_id : int
        Polytope index in [0, 4318].

    Returns
    -------
    list[list[int]]
        A list of integer 3-vectors (origin excluded).

    Raises
    ------
    ValueError
        If ``polytope_id`` is out of range or the page cannot be parsed.
    """
    if not 0 <= polytope_id < N_POLYTOPES:
        raise ValueError(
            f"polytope_id must be in [0, {N_POLYTOPES - 1}], got {polytope_id}"
        )

    url = _BASE_URL.format(polytope_id)
    try:
        with urlopen(url, timeout=15) as resp:
            html = resp.read().decode("utf-8")
    except URLError as exc:
        raise ValueError(
            f"Failed to fetch reflexive polytope {polytope_id}: {exc}"
        ) from exc

    # Locate the value cell that follows a cell containing "Integer points".
    match = re.search(
        r"Integer\s+points.*?<td[^>]*>(.*?)</td>",
        html,
        re.DOTALL | re.IGNORECASE,
    )
    if match is None:
        raise ValueError(
            f"Could not find 'Integer points' data for polytope {polytope_id}"
        )

    cell = match.group(1)
    rows_html = re.split(r"<br\s*/?>", cell, flags=re.IGNORECASE)

    rows: list[list[int]] = []
    for rh in rows_html:
        text = re.sub(r"<[^>]+>", "", rh)
        text = text.replace("[", "").replace("]", "").strip()
        if not text:
            continue
        nums = [int(x) for x in text.split()]
        if nums:
            rows.append(nums)

    if len(rows) != 3:
        raise ValueError(
            f"Expected 3 coordinate rows, got {len(rows)} "
            f"for polytope {polytope_id}"
        )
    if len(set(len(r) for r in rows)) != 1:
        raise ValueError(
            f"Rows have unequal length for polytope {polytope_id}"
        )

    # Transpose: rows are coordinates (x/y/z), columns are points.
    k = len(rows[0])
    vectors = []
    for j in range(k):
        pt = [rows[0][j], rows[1][j], rows[2][j]]
        if any(x != 0 for x in pt):   # exclude origin
            vectors.append(pt)

    return vectors


def reflexive_vc(polytope_id: int = 0) -> VectorConfiguration:
    """Return the VectorConfiguration of a 3D reflexive polytope.

    Parameters
    ----------
    polytope_id : int, optional
        Polytope index in [0, 4318]. Defaults to 0.

    Returns
    -------
    VectorConfiguration
        The vector configuration of non-origin lattice points.
    """
    return VectorConfiguration(_fetch_vectors(polytope_id))


def reflexive_fan(polytope_id: int = 0) -> Fan:
    """Return a triangulated fan from a 3D reflexive polytope.

    Parameters
    ----------
    polytope_id : int, optional
        Polytope index in [0, 4318]. Defaults to 0.

    Returns
    -------
    Fan
        A triangulated fan of the non-origin lattice points.
    """
    return reflexive_vc(polytope_id).triangulate()
