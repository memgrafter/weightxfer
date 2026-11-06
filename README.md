# weightxfer

Robust skin weight transfer from a skinned body onto a garment (or any other mesh), after Abdrashitov et al.,
"Robust Skin Weights Transfer via Weight Inpainting" (SIGGRAPH Asia 2023 Technical Communications,
[doi:10.1145/3610543.3626180](https://doi.org/10.1145/3610543.3626180)). It runs on the CPU and needs only numpy,
scipy and [robust-laplacian](https://github.com/nmwsharp/robust-laplacian-py) (MIT). It does not use libigl or any GPL code.

On 11 garment parts of a production character (138k vertices), it gives the same weights as the Robust Weight
Transfer Blender add-on (v1.1.9, default settings). The largest difference is 5.2e-8, which is float32 rounding.
A pose-metrics test that used both sets of weights produced identical numbers (see "How it matches the Blender
add-on").

## Use

```python
import weightxfer as wx

W, matched, report = wx.transfer(
    body_verts, body_faces, body_weights,      # (n, 3), (m, 3), (n, bones)
    cloth_verts, cloth_faces,                  # (k, 3), (l, 3) triangles
    max_dist=0.05, max_angle_deg=30, flip_normals=True)
# W: (k, bones) float64, rows sum to 1, at most 4 non-zero; matched: (k,) bool; report: dict (below)
```

Main options of `transfer` (keyword-only):

| option | default | meaning |
|---|---|---|
| `src_normals`, `dst_normals` | angle-weighted normals of the triangles | per-vertex normals; pass the host's own to reproduce a tool that used them (Blender computes them on the original quads) |
| `laplacian_kind` | `"point"` | `"point"`: point-cloud Laplacian of the destination vertices (pieces that lie near each other are linked); `"mesh"`: the destination triangles |
| `smooth` | `None` | `{"iterations": 10, "alpha": 0.2, "distance": max_dist}`: the reference code's smoothing pass |
| `max_influences` | `4` | influences per vertex (`None`: no limit) |
| `limit_dilate` | `5` | passes of the smoothed remove mask (0: plain top-k), see `limit_influences` |
| `normalize` | `True` | rows scaled to sum 1 (`False` keeps the sums the add-on writes) |
| `on_orphans` | `"nearest"` | vertices that cannot be inpainted get their closest-point weights and are reported; `"raise"` raises `ValueError` |

Other functions: `match` (step 1 alone), `laplacian`, `inpaint`, `smooth`, `limit_influences`, `normalize`,
`chain_bones` and `restrict_bones` (keep a part on its own bone chain), `closest_points`, `vertex_normals`.

Command line (one source, several destinations, through `.npz` files, so that a host such as Blender can call it
from its own Python):

```
python -m weightxfer IN.npz OUT.npz [--max-dist 0.05] [--max-angle 30] [--no-flip] [--laplacian point|mesh]
    [--smooth 10,0.2] [--k 4] [--dilate 5] [--no-normalize] [--on-orphans nearest|raise]
```

`IN.npz` holds `src_verts`, `src_faces`, `src_weights`, optionally `src_normals`, and `names`. For each name it
holds `verts__NAME` and `faces__NAME`, and optionally `normals__NAME`. `OUT.npz` holds `weights__NAME` and
`matched__NAME`. `OUT.json` holds the settings and one report per name.

## Install and test

```
git clone <this repository> weightxfer && cd weightxfer
python3 -m venv .venv
.venv/bin/pip install -e ".[test]"
.venv/bin/python -m pytest
```

The tests are self-contained and need no data. One test compares with the Blender add-on on recorded production
data. That data is not in this repository, so the test is skipped unless `WEIGHTXFER_ADDON_IO` points to a
recorded file (format: `tools/equivalence.py`; `WEIGHTXFER_SAVED` adds the pipeline's weights).

## Method

1. **Matching** (paper section 3.1). For each destination vertex, find the closest point on the source surface
   (exact, in float64; `weightxfer/closest.py`). Interpolate the source weights and normals there with barycentric
   weights. The vertex is matched if the distance is at most `max_dist` and the normals differ by at most
   `max_angle_deg`. With `flip_normals`, normals that differ by at least 180 - `max_angle_deg` also pass, which
   suits two-sided shells.
2. **Inpainting** (paper section 3.2). The unmatched rows minimise trace(Wᵀ Q W) with Q = L + L M⁻¹ L, and the
   matched rows are fixed. L is the robust Laplacian (Sharp and Crane 2020) and M is its lumped mass matrix. One
   sparse LU factorisation (scipy SuperLU) serves all bones. Q maps constants to zero, so rows still sum to 1.
3. **Smoothing** (optional, section 3.3, off by default as in the add-on). Smoothing starts from the unmatched
   vertices and spreads, through vertices not yet selected, to every vertex within `distance` of the unmatched
   vertex it was reached from. Each selected vertex is updated `iterations` times with w ← (1 − α) w + α ·
   mean(neighbours), in vertex order and in place, as in the reference code.
4. **Clean-up and limit.** Negative weights are set to 0 and weights below 1e-5 are dropped. Weights below 1e-4
   are dropped, then at most k influences are kept per vertex. A bone dropped at one vertex fades out over the
   neighbours through a smoothed remove mask (`limit_influences`). Values above 1 are clamped to 1, and rows are
   normalised if `normalize`.
5. **Bone chains** (helper). `chain_bones(part_bones, parents)` returns a part's bones with their parents and
   children. `restrict_bones(W, allowed, fallback)` keeps only those bones, renormalises, and gives a row with no
   weight left to `fallback`.

The results are deterministic: the same inputs and the same versions of numpy, scipy and robust-laplacian give
the same bytes. All steps are single-threaded numpy/scipy code with fixed orders, and the tests check this in
the same process and in a new one. The equivalence numbers below were measured with Python 3.13, numpy 2.5.4,
scipy 1.18.1 and robust-laplacian 1.0.0. The test suite also passes from a fresh clone with Python 3.15 and
robust-laplacian 1.1.0.

### Robustness

If a piece of the destination has no matched vertex (an island, or a component of the Laplacian's graph), the
system is singular there. libigl's `min_quad_with_fixed` then fails for the whole mesh. The add-on stopped with
"Optimization failed" on a skirt part at 2 cm. weightxfer finds those components first, solves the rest, and
gives the orphaned vertices their closest-point weights. On that skirt part at 2 cm, one component of 1,179
vertices fell back this way; 2,758 vertices were matched and 3,764 inpainted. `report["fallback"]` records the rule, the
number of vertices and the reason. If nothing matches, or the factorisation fails, every unmatched vertex falls
back the same way. With `on_orphans="raise"`, a `ValueError` is raised instead. The point Laplacian needs more
than 30 vertices, so smaller meshes use the mesh Laplacian, and the report says so.

`report` holds: vertices, bones, matched (count and share), distance (median, max), laplacian, inpaint (unknowns,
constraints, components, components_without_match, solved, residual), fallback, smooth, limit (k, dilate,
vertices over k), normalized, empty_rows_refilled, seconds (measured: match, inpaint, total).

## How it matches the reference code

The method follows the MIT reference code `rin-23/RobustSkinWeightsTransferCode` at a60263c1 (`src/utils.py` and
`cpp_src/main.cpp`), with these differences:

- **Laplacian.** The reference uses libigl's cotangent matrix and Voronoi mass. weightxfer uses the robust
  Laplacian (point cloud or mesh), which is defined on non-manifold meshes and point sets. Q follows the C++
  version, L + L M⁻¹ L. The Python version scales the cotangent matrix by 2, which weights the two terms
  differently.
- **Normal angle.** The reference takes arccos of a dot product that can come out slightly above 1, which gives
  NaN, so the vertex is not matched. weightxfer clamps the dot product to [-1, 1].
- **Closest point.** The reference uses libigl's AABB tree. weightxfer uses an exact float64 search (KD-tree
  candidates and exact point-triangle distances). In a tie the lower triangle index wins. Interpolated values
  are the same either way, because the interpolation is continuous across shared edges.
- **No libigl solver.** `min_quad_with_fixed` is replaced by a direct solve of the unknown block. Unmatched
  islands are handled as described under Robustness.
- **Additions not in the reference:** `flip_normals`, the influence limit, pruning and clamping (taken from the
  Blender add-on's observed behaviour, below), and normalisation.

## How it matches the Blender add-on

The add-on (sentfromspacevr/robust-weight-transfer, GPL-3.0) was used only as a tool. The author of this
repository did not read its source code or its README. Before this work, a licence review in the same project
had listed which libigl functions the add-on calls (point-to-mesh distance, barycentric coordinates, the
constrained quadratic solve, connected components, adjacency), and that list was known. The behaviour was found
by running the add-on on our own inputs and on synthetic test meshes, and by reading its settings' names and
descriptions as Blender lists them at run time.
It behaves as follows (Point mode, default settings, 4 groups):

- It takes the normals from the host (Blender's vertex normals) and matches as in step 1. The match masks were
  identical on all 11 test parts.
- It inpaints with the robust point-cloud Laplacian (30 neighbours) and Q = L + L M⁻¹ L. It then sets negative
  weights to 0 and drops weights below 1e-5. With its limit switched off, the largest difference to weightxfer
  was 3e-8.
- Its limit (setting "Limit Groups per Vertex"): weights below 1e-4 are dropped. A vertex with more than 4
  weights is *limited*. At a limited vertex, every bone outside its top 4 gets remove-mask 1, including bones
  the vertex has no weight for. The mask is then updated 5 times with mask ← max(mask, mean of the mask over the
  vertex and its edge neighbours). Then w ← w (1 − mask), the top 4 are kept, and weights below 1e-4 are
  dropped. Values above 1 are clamped only after this step.
  Changing the add-on's `dilation_repeat` setting (0 to 4) did not change its output in our runs: it always
  made 5 passes. Five synthetic strips were reproduced to 2e-8; two of them are in the tests.
- It does not renormalise the rows (sums from 0.57 to 1.51 on the test parts).

### Result on the test parts

Settings: 0.05 m, 30°, flipped normals allowed, 4 groups, no smoothing. The source has 13,380 vertices; the parts
have 4,142 to 60,649 vertices.

- Against the add-on's vertex-group weights, the largest difference per part is 2.98e-8 to 5.22e-8. The add-on
  stores float32.
- After the pipeline's own top-4 step (float32), the largest difference is 5.96e-8 and the bone indices are
  identical.
- The pipeline's pose metrics were rerun with these weights: stretch per part and pose, left-behind pieces, and
  leg-through counts over 51 walk frames, for 8 weight variants. All 1,333 numbers per variant were identical to
  the add-on run.

### Sensitivity

`tools/equivalence.py --sensitivity` measures how far each choice moves the weights. The figures below are rows
that change by more than 0.01 after the top-4 step, out of the 138k vertices:

| change | rows changed |
|---|---|
| fresh normals of the welded triangles instead of the stale host normals (matched 50,664 instead of 49,430) | 25,019 |
| a plain top-4 limit instead of the smoothed remove mask | 4,691 |
| smoothing on (10 passes, α = 0.2) | 27,762 |
| the mesh Laplacian | 53,275; 572 vertices on islands fall back |
| 2 cm instead of 5 cm (matched 33,494) | 65,734; 1,179 vertices fall back |

The add-on's normals come from Blender. For a mesh that was edited with bmesh and written back, they can be
stale normals from before the edit, so a caller who wants the add-on's result must pass the same normals.

## Limits

- CPU only. Times were measured on an Apple Silicon Mac shared with other jobs, against a 13k-vertex body. For
  parts of 4k to 11k vertices, weightxfer took 0.1 to 0.6 s in the pipeline run (0.3 to 1.3 s in a run under
  heavier load) and the add-on 0.1 to 0.3 s. For the 60k-vertex part, weightxfer took 5.6 s in the pipeline run
  and 7.6 to 12 s under load (closest points 4.0 s, inpainting 3.3 s in one of those runs); the add-on took
  2.3 s. The closest-point search is exact but pure numpy, which is slower than a BVH.
- The smoothing pass is a Python loop (sequential by design, as in the reference) and is slow on large meshes.
- The point Laplacian links pieces that lie within a vertex's 30 nearest neighbours. Whether two garment walls or
  neighbouring parts should share weights is a modelling choice. Use `laplacian_kind="mesh"` to keep pieces apart.
- Inpainted weights follow the matched ones smoothly. They do not know about cloth that should hang freely from
  one bone: transfer from a body spreads leg weights into a cloak where it touches the legs. Assign such parts to
  their own bones instead.
- Only triangles (quads must be split first).

## Files

- `weightxfer/transfer.py`: `match`, `transfer`.
- `weightxfer/inpaint.py`: `laplacian`, `inpaint`, `smooth`.
- `weightxfer/limit.py`: `limit_influences`, `normalize`, `chain_bones`, `restrict_bones`.
- `weightxfer/closest.py`: exact closest points on a triangle mesh.
- `weightxfer/geometry.py`: vertex normals, adjacency.
- `weightxfer/__main__.py`: the command line.
- `tests/`: `python -m pytest`.
- `tools/equivalence.py`: the comparison with recorded add-on inputs and outputs, and the sensitivity of each
  choice.
- `tools/blend_dump.py`: Blender script that writes a skinned mesh (positions, triangles, Blender normals,
  vertex-group weights) to `.npz`.

## Licence

MIT (`LICENSE`). The matching, inpainting and smoothing code is ported from the MIT reference code by Rinat
Abdrashitov. Its copyright notice and licence are in `NOTICE`. There is no GPL code and no libigl. libigl's
Python wheel bundles Shewchuk's Triangle, whose terms forbid commercial distribution, together with CGAL (GPL)
and TetGen (AGPL). The dependencies are numpy and scipy (BSD-3-Clause) and robust-laplacian (MIT).
