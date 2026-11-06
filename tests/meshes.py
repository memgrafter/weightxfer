"""Small test meshes (numpy only)."""
import numpy as np


def grid(nx, ny, size=1.0, z=0.0):
    """A flat grid in the xy plane: (nx * ny) vertices, 2 (nx - 1)(ny - 1) triangles, normals +z."""
    xs, ys = np.meshgrid(np.linspace(0, size, nx), np.linspace(0, size, ny), indexing="ij")
    V = np.stack([xs.ravel(), ys.ravel(), np.full(nx * ny, z)], 1)
    F = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b, c, d = i * ny + j, (i + 1) * ny + j, (i + 1) * ny + j + 1, i * ny + j + 1
            F += [(a, b, c), (a, c, d)]
    return V, np.array(F, np.int64)


def two_bone_weights(V, size=1.0):
    """Smooth 2-bone weights along x (rows sum to 1)."""
    t = np.clip(V[:, 0] / size, 0, 1)
    return np.stack([1 - t, t], 1)


def strip(n, step=0.01):
    """2 x n vertices along x, quads split into triangles: vertex 2i + j at (i step, j step, 0)."""
    V = np.array([[i * step, j * step, 0.0] for i in range(n) for j in range(2)])
    F = []
    for i in range(n - 1):
        q = [2 * i, 2 * i + 2, 2 * i + 3, 2 * i + 1]
        F += [(q[0], q[1], q[2]), (q[0], q[2], q[3])]
    return V, np.array(F, np.int64)
