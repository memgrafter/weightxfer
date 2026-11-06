"""Weight inpainting (Abdrashitov et al. 2023, section 3.2) and the optional smoothing pass (section 3.3).

Ported from the MIT reference code github.com/rin-23/RobustSkinWeightsTransferCode at a60263c1 (src/utils.py and
cpp_src/main.cpp, Copyright (c) 2024 Rinat Abdrashitov; licence in NOTICE), with libigl replaced by scipy and
robust-laplacian.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.sparse.csgraph import connected_components

__all__ = ["laplacian", "inpaint", "smooth"]


def laplacian(verts, faces=None, kind="point", n_neighbors=30, mollify_factor=1e-5):
    """(L, M): the robust Laplacian of Sharp and Crane 2020 (package robust-laplacian, MIT), L positive
    semi-definite, M the diagonal lumped mass matrix.

    kind="point": point-cloud Laplacian of the vertices (ignores the faces; separate pieces that lie close together
    are connected, so a piece without a matched vertex can still get weights). kind="mesh": the intrinsic Delaunay
    mesh Laplacian (works on non-manifold meshes; pieces stay separate)."""
    import robust_laplacian
    V = np.ascontiguousarray(verts, np.float64)
    if kind == "point":
        if len(V) <= n_neighbors:
            raise ValueError(f"point Laplacian needs more than n_neighbors={n_neighbors} points, got {len(V)}")
        L, M = robust_laplacian.point_cloud_laplacian(V, mollify_factor=mollify_factor, n_neighbors=n_neighbors)
    elif kind == "mesh":
        L, M = robust_laplacian.mesh_laplacian(V, np.ascontiguousarray(faces, np.int64), mollify_factor=mollify_factor)
    else:
        raise ValueError(f"laplacian kind must be 'point' or 'mesh', not {kind!r}")
    return sp.csr_matrix(L), sp.csr_matrix(M)


def inpaint(L, M, weights, matched):
    """Solve for the unmatched rows of `weights` (n x bones): minimise trace(W^T Q W) with Q = L + L M^-1 L
    (the reference code's -L + L M^-1 L with libigl's negative semi-definite cotangent matrix), the matched rows
    fixed. The rows of the solution sum to the same value as the constraints when every constraint row sums to 1
    (Q annihilates constants).

    Returns (W, info). info: unknowns, constraints, components (of the Laplacian's graph), components_without_match,
    vertices_without_match (indices: their component has no matched vertex, so the system is singular there;
    they are NOT solved and keep their input rows), solved (bool), residual (max abs of Q_uu x - rhs, relative)."""
    W = np.array(weights, np.float64, copy=True)
    m = np.asarray(matched, bool)
    n = len(W)
    info = {"unknowns": int((~m).sum()), "constraints": int(m.sum())}
    G = L.copy()
    G.data[:] = 1.0
    ncomp, lab = connected_components(G, directed=False)
    has = np.zeros(ncomp, bool)
    has[lab[m]] = True
    orphan = ~has[lab]
    info.update(components=int(ncomp), components_without_match=int((~has).sum()),
                vertices_without_match=np.flatnonzero(orphan))
    u = np.flatnonzero(~m & ~orphan)
    k = np.flatnonzero(m)
    info["solved"] = False
    info["residual"] = 0.0
    if len(u) == 0:
        info["solved"] = True
        return W, info
    Minv = sp.diags(1.0 / M.diagonal())
    Q = (L + L @ Minv @ L).tocsr()
    Quu = Q[u][:, u].tocsc()
    rhs = -(Q[u][:, k] @ W[k])
    try:
        lu = sla.splu(Quu, permc_spec="COLAMD")
        X = lu.solve(rhs)
    except RuntimeError as e:                      # "Factor is exactly singular"
        info["error"] = str(e)
        return W, info
    if not np.isfinite(X).all():
        info["error"] = "non-finite solution"
        return W, info
    r = Quu @ X - rhs
    scale = max(float(np.abs(rhs).max()), 1e-300)
    info["residual"] = float(np.abs(r).max() / scale)
    W[u] = X
    info["solved"] = True
    return W, info


def smooth(verts, faces, weights, matched, distance, iterations=10, alpha=0.2):
    """The reference code's optional smoothing (section 3.3): the unmatched vertices and every vertex reachable
    from one of them through not-yet-selected vertices within `distance` of it are smoothed `iterations` times:
    w_i <- (1 - alpha) w_i + alpha * mean of the neighbours' w, in vertex order and in place (later vertices see
    the new values of earlier ones), as in the reference code. Returns (W, selected mask)."""
    V = np.asarray(verts, np.float64)
    F = np.asarray(faces, np.int64)
    W = np.array(weights, np.float64, copy=True)
    n = len(V)
    from .geometry import adjacency
    A = adjacency(F, n)
    ip, ix = A.indptr, A.indices
    sel = ~np.asarray(matched, bool)
    for vid in np.flatnonzero(sel):               # breadth-first from each unmatched vertex (cpp_src/main.cpp)
        queue = [vid]
        seen = {vid}
        h = 0
        while h < len(queue):
            vv = queue[h]
            h += 1
            for nn in ix[ip[vv]:ip[vv + 1]]:
                if not sel[nn] and np.linalg.norm(V[vid] - V[nn]) < distance:
                    sel[nn] = True
                    if nn not in seen:
                        queue.append(nn)
                        seen.add(nn)
    idx = np.flatnonzero(sel)
    for _ in range(iterations):
        for i in idx:
            nb = ix[ip[i]:ip[i + 1]]
            if len(nb) == 0:
                continue
            W[i] = (1.0 - alpha) * W[i] + (alpha / len(nb)) * W[nb].sum(0)
    return W, sel
