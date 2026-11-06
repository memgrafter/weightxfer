"""Command line: weights for several destination meshes from one skinned source, through .npz files (so a host such
as Blender can call weightxfer in its own Python environment).

  python -I -m weightxfer IN.npz OUT.npz [--max-dist 0.05] [--max-angle 30] [--no-flip] [--laplacian point|mesh]
      [--smooth ITERATIONS,ALPHA] [--k 4] [--dilate 5] [--no-normalize] [--on-orphans nearest|raise]

IN.npz: src_verts (n x 3), src_faces (m x 3), src_weights (n x bones), optional src_normals (n x 3); names (the
destination names) and per name: verts__NAME, faces__NAME, optional normals__NAME.
OUT.npz: weights__NAME (float64, n x bones), matched__NAME (bool). OUT.json (next to OUT.npz, same stem): the
settings and one report per name (matched counts, inpaint solve, fallback, limit, seconds).
"""
import argparse
import json
import os
import sys

import numpy as np

from .transfer import transfer


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m weightxfer", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--max-dist", type=float, default=0.05)
    ap.add_argument("--max-angle", type=float, default=30.0)
    ap.add_argument("--no-flip", action="store_true")
    ap.add_argument("--laplacian", choices=("point", "mesh"), default="point")
    ap.add_argument("--smooth", default=None, help="ITERATIONS,ALPHA (e.g. 10,0.2); off by default")
    ap.add_argument("--k", type=int, default=4, help="influences per vertex (0: no limit)")
    ap.add_argument("--dilate", type=int, default=5, help="remove-mask passes of the limit (0: plain top-k)")
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--on-orphans", choices=("nearest", "raise"), default="nearest")
    a = ap.parse_args(argv)
    z = np.load(a.inp, allow_pickle=False)
    smooth = None
    if a.smooth:
        it, al = a.smooth.split(",")
        smooth = {"iterations": int(it), "alpha": float(al)}
    settings = {"max_dist": a.max_dist, "max_angle_deg": a.max_angle, "flip_normals": not a.no_flip,
                "laplacian": a.laplacian, "smooth": smooth, "k": a.k, "dilate": a.dilate,
                "normalize": not a.no_normalize, "on_orphans": a.on_orphans}
    src = dict(src_verts=z["src_verts"], src_faces=z["src_faces"], src_weights=z["src_weights"],
               src_normals=z["src_normals"] if "src_normals" in z else None)
    out, reps = {}, {}
    for name in [str(x) for x in z["names"]]:
        W, m, rep = transfer(src["src_verts"], src["src_faces"], src["src_weights"], z[f"verts__{name}"],
                             z[f"faces__{name}"], a.max_dist, a.max_angle, not a.no_flip,
                             src_normals=src["src_normals"],
                             dst_normals=z[f"normals__{name}"] if f"normals__{name}" in z else None,
                             laplacian_kind=a.laplacian, smooth=smooth, max_influences=a.k, limit_dilate=a.dilate,
                             normalize=not a.no_normalize, on_orphans=a.on_orphans)
        out[f"weights__{name}"] = W
        out[f"matched__{name}"] = m
        reps[name] = rep
        fb = f" FALLBACK {rep['fallback']}" if rep["fallback"] else ""
        print(f"weightxfer {name}: {rep['vertices']} vertices, matched {rep['matched']} "
              f"({rep['matched_share']:.1%}), {rep['seconds']['total']} s{fb}", file=sys.stderr, flush=True)
    np.savez(a.out, **out)
    with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
        json.dump({"settings": settings, "reports": reps}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
