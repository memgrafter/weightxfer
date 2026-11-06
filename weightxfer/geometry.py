"""Mesh helpers: vertex normals, vertex adjacency, connected components (numpy and scipy only)."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

__all__ = ["vertex_normals", "adjacency", "components"]


def vertex_normals(verts, faces):
    """Unit vertex normals: face normals weighted by the corner angle at the vertex (as Blender computes them).
    A vertex with no non-degenerate face gets (0, 0, 0)."""
    V = np.asarray(verts, np.float64)
    F = np.asarray(faces, np.int64)
    N = np.zeros_like(V)
    P = V[F]                                            # (m, 3 corners, 3)
    fn = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    ln = np.linalg.norm(fn, axis=1, keepdims=True)
    fn = np.where(ln > 0, fn / np.where(ln > 0, ln, 1), 0)
    for k in range(3):
        e1 = P[:, (k + 1) % 3] - P[:, k]
        e2 = P[:, (k + 2) % 3] - P[:, k]
        l1 = np.linalg.norm(e1, axis=1)
        l2 = np.linalg.norm(e2, axis=1)
        c = np.einsum("ij,ij->i", e1, e2) / np.maximum(l1 * l2, 1e-300)
        ang = np.arccos(np.clip(c, -1.0, 1.0))
        np.add.at(N, F[:, k], fn * ang[:, None])
    ln = np.linalg.norm(N, axis=1, keepdims=True)
    return np.where(ln > 0, N / np.where(ln > 0, ln, 1), 0)


def adjacency(faces, n):
    """Symmetric 0/1 vertex adjacency (CSR, n x n, no diagonal) from the triangles' edges."""
    F = np.asarray(faces, np.int64)
    i = np.concatenate([F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0]])
    j = np.concatenate([F[:, 1], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 2]])
    keep = i != j
    A = sp.csr_matrix((np.ones(keep.sum()), (i[keep], j[keep])), shape=(n, n))
    A.data[:] = 1.0
    A.sort_indices()
    return A


def components(A):
    """Connected components of a sparse graph: (count, label per vertex)."""
    return connected_components(A, directed=False)
