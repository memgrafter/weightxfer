"""Robust skin weight transfer (Abdrashitov et al., SIGGRAPH Asia 2023): closest-point matching, inpainting of the
unmatched vertices, optional smoothing, influence limit.

The matching and inpainting follow the MIT reference code github.com/rin-23/RobustSkinWeightsTransferCode at
a60263c1 (Copyright (c) 2024 Rinat Abdrashitov; licence in NOTICE).
"""
import time

import numpy as np

from .closest import closest_points
from .geometry import vertex_normals
from .inpaint import inpaint, laplacian, smooth as smooth_weights
from .limit import limit_influences, normalize as normalize_rows

__all__ = ["match", "transfer"]


def _unit(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return np.where(n > 0, X / np.where(n > 0, n, 1.0), 0.0)


def match(src_verts, src_faces, src_weights, dst_verts, max_dist, max_angle_deg, flip_normals=True,
          src_normals=None, dst_normals=None, dst_faces=None):
    """Section 3.1. For each destination vertex: the closest point on the source surface, the source weights and
    normals interpolated there (barycentric), and MATCHED if the distance is <= max_dist and the angle between the
    interpolated source normal and the destination normal is <= max_angle_deg (or >= 180 - max_angle_deg when
    flip_normals: two-sided surfaces such as cloth shells).

    Returns (weights [n x bones], matched [n] bool, distance [n], angle_deg [n]). Normals default to
    angle-weighted vertex normals of the triangles (`vertex_normals`); pass your own (for example Blender's, which
    are computed on the original quads) to reproduce a tool that used them."""
    SV = np.asarray(src_verts, np.float64)
    SF = np.asarray(src_faces, np.int64)
    SW = np.asarray(src_weights, np.float64)
    DV = np.asarray(dst_verts, np.float64)
    if SW.ndim != 2 or len(SW) != len(SV):
        raise ValueError("src_weights must be (source vertices x bones)")
    SN = vertex_normals(SV, SF) if src_normals is None else np.asarray(src_normals, np.float64)
    if dst_normals is None:
        if dst_faces is None:
            raise ValueError("match: give dst_normals or dst_faces")
        DN = vertex_normals(DV, dst_faces)
    else:
        DN = np.asarray(dst_normals, np.float64)
    sq, face, bary = closest_points(DV, SV, SF)
    tri = SF[face]
    W = np.einsum("nk,nkb->nb", bary, SW[tri])
    n1 = _unit(np.einsum("nk,nkd->nd", bary, SN[tri]))
    n2 = _unit(DN)
    ang = np.degrees(np.arccos(np.clip(np.einsum("nd,nd->n", n1, n2), -1.0, 1.0)))
    ok_n = ang <= max_angle_deg
    if flip_normals:
        ok_n |= ang >= 180.0 - max_angle_deg
    matched = (sq <= max_dist * max_dist) & ok_n
    return W, matched, np.sqrt(sq), ang


def transfer(src_verts, src_faces, src_weights, dst_verts, dst_faces, max_dist, max_angle_deg=30.0,
             flip_normals=True, *, src_normals=None, dst_normals=None, laplacian_kind="point",
             smooth=None, max_influences=4, limit_dilate=5, prune_solved=1e-5, prune_limited=1e-4,
             normalize=True, on_orphans="nearest"):
    """Robust skin weight transfer from a skinned source mesh onto a destination mesh.

    src_verts (n_s x 3), src_faces (m_s x 3), src_weights (n_s x bones), dst_verts (n x 3), dst_faces (m x 3).
    max_dist: matching distance in the meshes' units. max_angle_deg: normal threshold (degrees).
    flip_normals: also accept normals that are flipped (angle >= 180 - max_angle_deg).
    src_normals, dst_normals: per-vertex normals (default: angle-weighted normals of the triangles).
    laplacian_kind: "point" (point-cloud Laplacian of the destination vertices: pieces near each other are linked)
        or "mesh" (the destination triangles).
    smooth: None (no smoothing) or a dict(iterations=10, alpha=0.2, distance=max_dist): the reference code's pass.
    max_influences: k (None or 0: no limit). limit_dilate: passes of the smoothed remove mask (0: plain top-k);
        see limit_influences.
    prune_solved: inpainted weights below it are set to 0 (after clamping to [0, 1]).
    prune_limited: weights below it are set to 0 after the limit.
    normalize: rows scaled to sum 1 at the end (False keeps the raw sums, as the Blender add-on writes them).
    on_orphans: what to do with vertices whose Laplacian component has no matched vertex (the system is singular
        there; libigl's min_quad_with_fixed fails on the whole mesh): "nearest" gives them the closest-point weights
        (as if matched) and reports it; "raise" raises ValueError.

    Returns (weights [n x bones] float64, matched [n] bool, report dict). report["fallback"] says whether and where
    the closest-point fallback was used; it is never silent."""
    t0 = time.perf_counter()
    DV = np.asarray(dst_verts, np.float64)
    DF = np.asarray(dst_faces, np.int64)
    W, matched, dist, ang = match(src_verts, src_faces, src_weights, DV, max_dist, max_angle_deg, flip_normals,
                                  src_normals, dst_normals, DF)
    t_match = time.perf_counter()
    nearest = W.copy()
    rep = {"vertices": int(len(DV)), "bones": int(W.shape[1]), "matched": int(matched.sum()),
           "matched_share": float(matched.mean()) if len(DV) else 0.0, "max_dist": float(max_dist),
           "max_angle_deg": float(max_angle_deg), "flip_normals": bool(flip_normals),
           "laplacian": laplacian_kind, "distance_m": {"median": float(np.median(dist)) if len(dist) else 0.0,
                                                       "max": float(dist.max()) if len(dist) else 0.0},
           "fallback": None}
    if matched.all():
        info = {"unknowns": 0, "constraints": int(len(DV)), "components": None, "components_without_match": 0,
                "vertices_without_match": np.zeros(0, np.int64), "solved": True, "residual": 0.0}
    elif not matched.any():
        info = {"unknowns": int(len(DV)), "constraints": 0, "components": None, "components_without_match": None,
                "vertices_without_match": np.arange(len(DV)), "solved": False, "residual": None,
                "error": "no destination vertex matched"}
    else:
        kind = laplacian_kind
        if kind == "point" and len(DV) <= 30:     # the point Laplacian uses 30 neighbours per point
            kind = "mesh"
            rep["laplacian"] = "mesh (point needs more than 30 vertices)"
        L, M = laplacian(DV, DF, kind)
        W, info = inpaint(L, M, W, matched)
    orphans = info.pop("vertices_without_match")
    rep["inpaint"] = info
    if not info["solved"]:
        # the solve failed as a whole (no match at all, or a singular/non-finite factorisation): every unmatched
        # vertex falls back to its closest-point weights
        orphans = np.flatnonzero(~matched)
    if len(orphans):
        if on_orphans == "raise":
            raise ValueError(f"weightxfer: {len(orphans)} vertices cannot be inpainted "
                             f"({info.get('error') or str(info['components_without_match']) + ' pieces without a match'})")
        if on_orphans != "nearest":
            raise ValueError(f"on_orphans must be 'nearest' or 'raise', not {on_orphans!r}")
        W[orphans] = nearest[orphans]
        rep["fallback"] = {"rule": "closest-point weights", "vertices": int(len(orphans)),
                           "share": float(len(orphans) / len(DV)),
                           "reason": info.get("error") or f"{info['components_without_match']} Laplacian components "
                                                           "without a matched vertex"}
    t_inpaint = time.perf_counter()
    if smooth:
        s = dict(iterations=10, alpha=0.2, distance=max_dist)
        s.update(smooth)
        W, sel = smooth_weights(DV, DF, W, matched, s["distance"], s["iterations"], s["alpha"])
        rep["smooth"] = dict(s, vertices=int(sel.sum()))
    else:
        rep["smooth"] = None
    # negatives to 0 and tiny weights pruned before the limit; values above 1 are clamped AFTER the limit (the
    # order the Blender add-on uses: a solved weight of 1.09 scaled by a remove mask of 0.0006 stays 1)
    W = np.maximum(W, 0.0)
    W[W < prune_solved] = 0.0
    if max_influences:
        before = int(((W >= prune_limited).sum(1) > max_influences).sum())
        W = limit_influences(W, DF, max_influences, limit_dilate, prune_limited)
        rep["limit"] = {"k": int(max_influences), "dilate": int(limit_dilate), "vertices_over_k": before}
    else:
        rep["limit"] = None
    W = np.minimum(W, 1.0)
    if normalize:
        empty = W.sum(1) <= 0
        W = normalize_rows(W)
        if empty.any():          # nothing left (all weights pruned): closest-point weights, limited, normalised
            Wn = np.clip(nearest[empty], 0, 1)
            if max_influences:
                Wn = limit_influences(Wn, None, max_influences, 0, 0.0)
            W[empty] = normalize_rows(Wn)
        rep["empty_rows_refilled"] = int(empty.sum())
    rep["normalized"] = bool(normalize)
    t_end = time.perf_counter()
    rep["seconds"] = {"match": round(t_match - t0, 3), "inpaint": round(t_inpaint - t_match, 3),
                      "total": round(t_end - t0, 3)}
    return W, matched, rep
