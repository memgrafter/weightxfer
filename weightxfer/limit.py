"""Influence limits: top-k per vertex (with an optional smoothed remove mask) and bone-chain limits per part."""
import numpy as np
import scipy.sparse as sp

from .geometry import adjacency

__all__ = ["limit_influences", "normalize", "chain_bones", "restrict_bones"]


def _topk_mask(W, k):
    idx = np.argsort(-W, axis=1, kind="stable")[:, :k]
    keep = np.zeros(W.shape, bool)
    np.put_along_axis(keep, idx, True, axis=1)
    return keep


def limit_influences(weights, faces=None, k=4, dilate=5, prune=1e-4):
    """At most k non-zero weights per vertex.

    First, weights below `prune` are set to 0. dilate = 0: plain top-k (ties: the lower bone index wins).
    dilate > 0 (needs `faces`): a smoothed remove mask, so a bone that is dropped at one vertex fades out over its
    neighbours instead of stopping at an edge:
      1. a vertex is LIMITED if it has more than k non-zero weights; at a limited vertex every bone outside its top
         k (including bones it has no weight for) gets mask 1; everything else 0;
      2. `dilate` times: mask <- max(mask, mean of the mask over the vertex and its edge neighbours);
      3. w <- w * (1 - mask), then top-k, then weights below `prune` set to 0 again.
    With dilate=5 and prune=1e-4 this is the behaviour of the Robust Weight Transfer Blender add-on's group limit
    (v1.1.9, default settings), found by black-box tests (README, "How it matches the add-on"). Rows are NOT
    renormalised (call normalize). If no vertex has more than k weights, nothing changes."""
    W = np.array(weights, np.float64, copy=True)
    W[W < prune] = 0.0
    nz = (W > 0).sum(1)
    limited = nz > k
    if not limited.any():
        return W
    if dilate and faces is not None:
        keep = _topk_mask(W, k)
        R = ((~keep) & limited[:, None]).astype(np.float64)
        A = adjacency(faces, len(W))
        deg = np.asarray(A.sum(1)).ravel()
        S = (sp.diags(1.0 / (deg + 1.0)) @ (A + sp.identity(len(W), format="csr"))).tocsr()
        for _ in range(dilate):
            R = np.maximum(R, S @ R)
        W *= 1.0 - np.clip(R, 0.0, 1.0)
    elif dilate:
        raise ValueError("limit_influences: dilate > 0 needs the faces")
    W *= _topk_mask(W, k) & (W > 0)
    W[W < prune] = 0.0
    return W


def normalize(weights, fallback=None):
    """Rows scaled to sum 1. Rows that sum to 0 get weight 1 on bone `fallback` (or stay 0 if None)."""
    W = np.array(weights, np.float64, copy=True)
    s = W.sum(1)
    ok = s > 0
    W[ok] /= s[ok, None]
    if fallback is not None:
        W[~ok, fallback] = 1.0
    return W


def chain_bones(part_bones, parents):
    """The bone chain of a part: its own bones, their parents and their children (bone indices).
    parents: parent index per bone (-1 for a root)."""
    par = np.asarray(parents)
    allow = set()
    for b in part_bones:
        allow.add(int(b))
        if par[b] >= 0:
            allow.add(int(par[b]))
        allow.update(int(c) for c in np.flatnonzero(par == b))
    return sorted(allow)


def restrict_bones(weights, allowed, fallback):
    """Weights on bones outside `allowed` set to 0, rows renormalised; a row left with no weight gets 1 on bone
    `fallback`. Returns (W, rows_reassigned)."""
    W = np.array(weights, np.float64, copy=True)
    mask = np.zeros(W.shape[1], bool)
    mask[list(allowed)] = True
    W[:, ~mask] = 0.0
    empty = W.sum(1) <= 0
    W = normalize(W)
    W[empty, fallback] = 1.0
    return W, int(empty.sum())
