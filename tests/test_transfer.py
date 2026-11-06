import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import weightxfer as wx
from weightxfer.closest import closest_points, point_triangle

from meshes import grid, strip, two_bone_weights


# --- closest points -------------------------------------------------------------------------------------------

def brute(P, V, F):
    best = np.full(len(P), np.inf)
    for t in F:
        s, _ = point_triangle(P, *(np.repeat(V[i][None], len(P), 0) for i in t))
        best = np.minimum(best, s)
    return best


def test_closest_points_exact_vs_brute_force():
    rng = np.random.default_rng(0)
    V = rng.normal(size=(200, 3))
    F = rng.integers(0, 200, size=(300, 3))
    F[:5, 1] = F[:5, 0]                                 # degenerate triangles (an edge)
    F[5, :] = F[5, 0]                                   # a point triangle
    P = rng.normal(size=(300, 3)) * 1.5
    sq, face, bary = closest_points(P, V, F)
    assert np.array_equal(sq, brute(P, V, F))
    q = np.einsum("nk,nkd->nd", bary, V[F[face]])
    assert np.allclose(((P - q) ** 2).sum(1), sq, atol=1e-12)
    assert np.allclose(bary.sum(1), 1) and (bary >= -1e-12).all()


def test_closest_points_mixed_triangle_sizes():
    V, F = grid(30, 30)
    big = np.array([[-5, -5, 0.5], [6, -5, 0.5], [0, 6, 0.5]])  # one huge triangle above the grid
    V2 = np.vstack([V, big])
    F2 = np.vstack([F, [[len(V), len(V) + 1, len(V) + 2]]])
    P = np.random.default_rng(1).uniform(-1, 2, size=(500, 3))
    sq, _, _ = closest_points(P, V2, F2)
    assert np.array_equal(sq, brute(P, V2, F2))


# --- transfer: analytic cases ----------------------------------------------------------------------------------

def test_identical_meshes_copy_the_weights():
    V, F = grid(12, 12)
    W = two_bone_weights(V)
    out, m, rep = wx.transfer(V, F, W, V, F, max_dist=0.01, max_angle_deg=30)
    assert m.all() and rep["matched"] == len(V) and rep["fallback"] is None
    assert np.allclose(out, W, atol=1e-12)


def test_shifted_copy_within_and_beyond_the_distance():
    V, F = grid(12, 12)
    W = two_bone_weights(V)
    out, m, rep = wx.transfer(V, F, W, V + [0, 0, 0.02], F, max_dist=0.05, max_angle_deg=30)
    assert m.all() and np.allclose(out, W, atol=1e-12)
    # beyond max_dist: nothing matches -> closest-point fallback for all, reported
    out, m, rep = wx.transfer(V, F, W, V + [0, 0, 0.2], F, max_dist=0.05, max_angle_deg=30)
    assert not m.any()
    assert rep["fallback"]["vertices"] == len(V) and "no destination vertex matched" in rep["fallback"]["reason"]
    assert np.allclose(out, W, atol=1e-12)
    with pytest.raises(ValueError):
        wx.transfer(V, F, W, V + [0, 0, 0.2], F, max_dist=0.05, on_orphans="raise")


def test_normal_threshold_and_flip():
    V, F = grid(12, 12)
    W = two_bone_weights(V)
    Ff = F[:, ::-1]                                      # same surface, normals flipped
    _, m, _ = wx.transfer(V, F, W, V, Ff, max_dist=0.01, max_angle_deg=30, flip_normals=True)
    assert m.all()
    _, m, rep = wx.transfer(V, F, W, V, Ff, max_dist=0.01, max_angle_deg=30, flip_normals=False)
    assert not m.any() and rep["fallback"] is not None


def flap_mesh():
    """Source: a 1 x 1 grid. Destination: the same grid (z = 0.005) plus a flap that leaves it at x = 1 and
    rises to z = 1 (far from the source, so unmatched)."""
    V, F = grid(16, 16)
    nx = ny = 16
    D = V + [0, 0, 0.005]
    flap = []
    for i in range(1, 10):                               # 9 rows rising at the x = 1 edge
        for j in range(ny):
            flap.append([1.0 + 0.02 * i, j / (ny - 1), 0.005 + 0.11 * i])
    D2 = np.vstack([D, flap])
    F2 = [tuple(f) for f in F]
    edge = [(nx - 1) * ny + j for j in range(ny)]
    rows = [edge] + [[len(D) + (i - 1) * ny + j for j in range(ny)] for i in range(1, 10)]
    for r0, r1 in zip(rows[:-1], rows[1:]):
        for j in range(ny - 1):
            F2 += [(r0[j], r1[j], r1[j + 1]), (r0[j], r1[j + 1], r0[j + 1])]
    return V, F, D2, np.array(F2, np.int64), len(D)


def test_unmatched_flap_is_inpainted():
    V, F, D, DF, n_base = flap_mesh()
    W = two_bone_weights(V)
    out, m, rep = wx.transfer(V, F, W, D, DF, max_dist=0.05, max_angle_deg=30, max_influences=None,
                              normalize=False)
    # the root row (x = 1) has normals tilted by the flap: unmatched too
    assert m[:n_base - 16].all() and not m[n_base - 16:].any()
    assert rep["fallback"] is None and rep["inpaint"]["solved"] and rep["inpaint"]["residual"] < 1e-8
    # matched rows keep the interpolated source weights; the flap continues the weights at its root
    W_exp, _, _, _ = wx.match(V, F, W, D, 0.05, 30, dst_faces=DF)
    assert np.allclose(out[m], W_exp[m], atol=1e-12)
    flap = out[n_base + 16:]
    assert np.allclose(flap.sum(1), 1, atol=1e-6)        # partition of unity (Q annihilates constants)
    assert (flap[:, 1] > 0.9).all()                      # the root is at x = 1: bone 1 has weight 1 there


def test_constant_weights_inpaint_to_the_constant():
    """The reference code's sphere-to-plane example: constant source weights (0.3, 0.7) stay constant."""
    V, F, D, DF, _ = flap_mesh()
    W = np.tile([0.3, 0.7], (len(V), 1))
    out, m, rep = wx.transfer(V, F, W, D, DF, max_dist=0.05, max_angle_deg=30, normalize=False)
    assert not m.all() and np.allclose(out, [0.3, 0.7], atol=1e-9)


def test_island_without_a_match_falls_back_and_is_reported():
    V, F, D, DF, _ = flap_mesh()
    W = two_bone_weights(V)
    IV, IF = grid(8, 8, size=0.3)
    IV = IV + [5, 5, 5]                                  # far away: its own Laplacian component
    D2 = np.vstack([D, IV])
    DF2 = np.vstack([DF, IF + len(D)])
    for kind in ("point", "mesh"):
        out, m, rep = wx.transfer(V, F, W, D2, DF2, max_dist=0.05, laplacian_kind=kind)
        fb = rep["fallback"]
        assert fb is not None and fb["vertices"] == len(IV), (kind, fb)
        assert rep["inpaint"]["components_without_match"] >= 1
        assert np.allclose(out.sum(1), 1) and np.isfinite(out).all()
        W_near, _, _, _ = wx.match(V, F, W, D2, 0.05, 30, dst_faces=DF2)
        assert np.allclose(out[len(D):], W_near[len(D):], atol=1e-9)
    with pytest.raises(ValueError):
        wx.transfer(V, F, W, D2, DF2, max_dist=0.05, on_orphans="raise")


def test_small_mesh_uses_the_mesh_laplacian():
    V, F = grid(5, 5)                                    # 25 vertices: too few for the 30-neighbour point Laplacian
    W = two_bone_weights(V)
    D = V.copy()
    D[V[:, 0] > 0.7, 2] += 0.5                           # lift a band: unmatched
    out, m, rep = wx.transfer(V, F, W, D, F, max_dist=0.05)
    assert rep["laplacian"].startswith("mesh") and not m.all() and rep["fallback"] is None


def test_smoothing_touches_only_the_selected_region():
    V, F, D, DF, n_base = flap_mesh()
    W = two_bone_weights(V)
    a, m, _ = wx.transfer(V, F, W, D, DF, max_dist=0.05, max_influences=None, normalize=False)
    b, _, rep = wx.transfer(V, F, W, D, DF, max_dist=0.05, max_influences=None, normalize=False,
                            smooth={"iterations": 10, "alpha": 0.2, "distance": 0.2})
    assert rep["smooth"]["vertices"] > (~m).sum()
    far = D[:, 0] < 0.7                                  # more than 0.2 from every unmatched vertex
    assert np.array_equal(a[far], b[far])
    assert not np.allclose(a, b)


# --- influence limit ------------------------------------------------------------------------------------------

# Bone-4 weights on the strip, vertices 28..52, as the Robust Weight Transfer add-on v1.1.9 wrote them (black-box
# run, default settings, source = destination): vertex 40 holds 5 groups and drops group 4; the others hold
# groups 0, 1, 2, 4 = (0.4, 0.3, 0.2, 0.1).
SPREAD_ADDON = [0.1, 0.1, 0.099968, 0.1, 0.099392, 0.09984, 0.095968, 0.098368, 0.084288, 0.092032, 0.056608,
                0.075584, 0.0, 0.049856, 0.056608, 0.049856, 0.084288, 0.075584, 0.095968, 0.092032, 0.099392,
                0.098368, 0.099968, 0.09984, 0.1]
# the same with two vertices (36 and 42) dropping group 4, vertices 24..55
TWO_ADDON = [0.1, 0.1, 0.099968, 0.1, 0.099392, 0.09984, 0.095968, 0.098368, 0.084288, 0.092032, 0.056512,
             0.075488, 0.0, 0.048768, 0.04144, 0.042432, 0.04144, 0.051296, 0.0, 0.042432, 0.056512, 0.048768,
             0.084288, 0.075488, 0.095968, 0.092032, 0.099392, 0.098368, 0.099968, 0.09984, 0.1, 0.1]


def strip_weights(centres):
    V, F = strip(40)
    W = np.tile([0.4, 0.3, 0.2, 0.0, 0.1, 0.0], (len(V), 1))
    for c in centres:
        W[c] = [0.35, 0.25, 0.2, 0.12, 0.08, 0.0]
    return F, W


def test_limit_matches_the_addon_on_the_strip():
    F, W = strip_weights([40])
    out = wx.limit_influences(W, F, k=4, dilate=5)
    assert np.allclose(out[28:53, 4], SPREAD_ADDON, atol=1e-6)
    assert np.allclose(out[:, :3], W[:, :3]) and out[40, 3] == 0.12
    F, W = strip_weights([36, 42])
    out = wx.limit_influences(W, F, k=4, dilate=5)
    assert np.allclose(out[24:56, 4], TWO_ADDON, atol=1e-6)


def test_limit_basics():
    F, W = strip_weights([40])
    plain = wx.limit_influences(W, None, k=4, dilate=0)
    assert ((plain > 0).sum(1) <= 4).all() and plain[40, 4] == 0 and np.array_equal(plain[39], W[39])
    W4 = np.tile([0.4, 0.3, 0.2, 0.1], (10, 1))
    assert np.array_equal(wx.limit_influences(W4, None, k=4), W4)      # nothing over k: unchanged
    tiny = np.array([[0.5, 0.49995, 0.00005]])
    assert np.array_equal(wx.limit_influences(tiny, None, k=4, dilate=0), [[0.5, 0.49995, 0.0]])


def test_chain_helpers():
    parents = np.array([-1, 0, 1, 2, 1, 4])               # 0 root, 1 spine, 2 arm_l, 3 hand_l, 4 arm_r, 5 hand_r
    assert wx.chain_bones([2], parents) == [1, 2, 3]
    W = np.array([[0.5, 0.2, 0.3, 0, 0, 0], [0, 0, 0, 0, 0.6, 0.4]])
    out, n = wx.restrict_bones(W, [1, 2, 3], fallback=2)
    assert n == 1 and np.allclose(out[0], [0, 0.4, 0.6, 0, 0, 0]) and np.array_equal(out[1], [0, 0, 1, 0, 0, 0])


# --- determinism ----------------------------------------------------------------------------------------------

def run_flap_bytes():
    V, F, D, DF, _ = flap_mesh()
    out, m, _ = wx.transfer(V, F, two_bone_weights(V), D, DF, max_dist=0.05)
    return hashlib.sha256(out.tobytes() + m.tobytes()).hexdigest()


def test_deterministic_in_process_and_in_a_new_process(tmp_path):
    h = run_flap_bytes()
    assert run_flap_bytes() == h
    code = ("import sys; sys.path.insert(0, %r); import test_transfer as t; print(t.run_flap_bytes())"
            % os.path.dirname(__file__))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert r.stdout.strip() == h


def test_command_line_round_trip(tmp_path):
    V, F, D, DF, _ = flap_mesh()
    W = two_bone_weights(V)
    inp, out = tmp_path / "in.npz", tmp_path / "out.npz"
    np.savez(inp, src_verts=V, src_faces=F, src_weights=W, names=np.array(["flap"]), verts__flap=D, faces__flap=DF)
    subprocess.run([sys.executable, "-m", "weightxfer", str(inp), str(out), "--max-dist", "0.05"], check=True,
                   capture_output=True)
    z = np.load(out)
    ref, m, _ = wx.transfer(V, F, W, D, DF, max_dist=0.05)
    assert np.array_equal(z["weights__flap"], ref) and np.array_equal(z["matched__flap"], m)
    rep = json.loads((tmp_path / "out.json").read_text())
    assert rep["reports"]["flap"]["matched"] == int(m.sum())
