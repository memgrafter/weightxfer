"""Exact closest points on a triangle mesh, in float64, with numpy and a scipy KD-tree.

For each query point p the nearest mesh vertex gives an upper bound d0 on the distance to the surface. A triangle
with centroid c and radius r (the largest distance from c to a vertex) can only hold a point within d0 of p if
|p - c| <= d0 + r. Triangles are grouped by radius (each group's radius at most twice its smallest), and each group's
centroids are searched with k-nearest-neighbour queries whose k doubles until every candidate inside the bound is
found. The candidates are then measured exactly. The result is the exact closest point, not an approximation.

Ties (a point equally close to two triangles, e.g. at a shared vertex) go to the lower triangle index. Barycentric
interpolation of per-vertex data is continuous across shared edges, so the choice of face in a tie does not change
interpolated values beyond rounding.
"""
import numpy as np
from scipy.spatial import cKDTree

__all__ = ["closest_points", "point_triangle"]


def point_triangle(p, a, b, c):
    """Squared distance and barycentric coordinates (wa, wb, wc) of the closest point on triangles (a, b, c) to p.
    All inputs are (n, 3) float64. Degenerate triangles (zero area) are handled as their edges."""
    ab, ac, ap = b - a, c - a, p - a
    d00 = np.einsum("ij,ij->i", ab, ab)
    d01 = np.einsum("ij,ij->i", ab, ac)
    d11 = np.einsum("ij,ij->i", ac, ac)
    d20 = np.einsum("ij,ij->i", ap, ab)
    d21 = np.einsum("ij,ij->i", ap, ac)
    den = d00 * d11 - d01 * d01
    ok = den > 1e-30 * np.maximum(d00 * d11, 1e-300)
    sden = np.where(ok, den, 1.0)
    v = (d11 * d20 - d01 * d21) / sden
    w = (d00 * d21 - d01 * d20) / sden
    u = 1.0 - v - w
    inside = ok & (u >= 0) & (v >= 0) & (w >= 0)
    q = a + v[:, None] * ab + w[:, None] * ac
    best = np.einsum("ij,ij->i", p - q, p - q)
    best = np.where(inside, best, np.inf)
    bary = np.stack([u, v, w], 1)
    # edges ab, bc, ca: the closest point when the projection is outside (or the triangle is degenerate)
    for e0, e1, i0, i1 in ((a, b, 0, 1), (b, c, 1, 2), (c, a, 2, 0)):
        e = e1 - e0
        ee = np.einsum("ij,ij->i", e, e)
        t = np.einsum("ij,ij->i", p - e0, e) / np.where(ee > 0, ee, 1.0)
        t = np.clip(np.where(ee > 0, t, 0.0), 0.0, 1.0)
        qe = e0 + t[:, None] * e
        de = np.einsum("ij,ij->i", p - qe, p - qe)
        take = (~inside) & (de < best)
        if take.any():
            best = np.where(take, de, best)
            nb = np.zeros_like(bary)
            nb[:, i0] = 1.0 - t
            nb[:, i1] = t
            bary = np.where(take[:, None], nb, bary)
    return best, bary


def closest_points(points, verts, faces, chunk=200_000):
    """Closest point on the mesh (verts, faces) for each point.

    Returns (sqr_dist [n], face [n] int64, bary [n, 3]): squared distance, the triangle holding the closest point
    (lowest index on a tie) and the closest point's barycentric coordinates in that triangle."""
    P = np.ascontiguousarray(points, np.float64)
    V = np.ascontiguousarray(verts, np.float64)
    F = np.ascontiguousarray(faces, np.int64)
    if len(F) == 0:
        raise ValueError("closest_points: the mesh has no triangles")
    used = np.unique(F)
    d0, _ = cKDTree(V[used]).query(P)                 # distance to the nearest referenced vertex: an upper bound
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cen = (A + B + C) / 3.0
    rad = np.sqrt(np.max(np.stack([((X - cen) ** 2).sum(1) for X in (A, B, C)], 1), 1))
    # radius groups: [r_min * 2^k, r_min * 2^(k+1)); zero-radius (fully collapsed) triangles join the first group
    rpos = rad[rad > 0]
    r0 = rpos.min() if len(rpos) else 1.0
    grp = np.floor(np.log2(np.maximum(rad, r0) / r0)).astype(np.int64)
    best = np.full(len(P), np.inf)
    bface = np.full(len(P), -1, np.int64)
    bbary = np.zeros((len(P), 3))
    for g in np.unique(grp):
        tri = np.flatnonzero(grp == g)                  # ascending triangle ids
        tree = cKDTree(cen[tri])
        rmax = rad[tri].max()
        for s in range(0, len(P), chunk):
            sl = slice(s, s + chunk)
            p = P[sl]
            bound = d0[sl] * (1 + 1e-12) + rmax + 1e-12
            k = min(8, len(tri))
            todo = np.arange(len(p))
            while len(todo):
                dist, idx = tree.query(p[todo], k=k)
                if k == 1:
                    dist, idx = dist[:, None], idx[:, None]
                full = (dist[:, -1] <= bound[todo]) & (k < len(tri))   # may have missed candidates: grow k
                done = ~full
                rows = todo[done]
                dd, ii = dist[done], idx[done]
                m = dd <= bound[rows][:, None]
                r_, c_ = np.nonzero(m)
                if len(r_):
                    q = rows[r_]
                    t = tri[ii[r_, c_]]
                    sq, ba = point_triangle(p[q], A[t], B[t], C[t])
                    gq = q + s
                    # per query, the smallest squared distance, then the lowest triangle id
                    order = np.lexsort((t, sq, gq))
                    gq, t, sq, ba = gq[order], t[order], sq[order], ba[order]
                    first = np.ones(len(gq), bool)
                    first[1:] = gq[1:] != gq[:-1]
                    gq, t, sq, ba = gq[first], t[first], sq[first], ba[first]
                    better = (sq < best[gq]) | ((sq == best[gq]) & (t < bface[gq]))
                    gq, t, sq, ba = gq[better], t[better], sq[better], ba[better]
                    best[gq] = sq
                    bface[gq] = t
                    bbary[gq] = ba
                todo = todo[full]
                k = min(k * 4, len(tri))
    if (bface < 0).any():
        raise RuntimeError("closest_points: no triangle found for some points (internal error)")
    return best, bface, bbary
