"""Equivalence of weightxfer with the Robust Weight Transfer Blender add-on on recorded inputs and outputs.

  python -I tools/equivalence.py ADDON_IO.npz [--saved WEIGHTS.npz] [--sensitivity] [--json OUT.json]

ADDON_IO.npz: what the add-on received and returned, per part, recorded by the caller (in our pipeline:
parts_weights.py --dump-inputs): sv, st, sn (source positions, triangles, Blender vertex normals), src_w (source
vertex-group weights, vertices x bones), parts (names), and per part tv_<p>, tt_<p>, tn_<p> (destination positions,
triangles, the normals the add-on read) and w_tool_<p> (the add-on's vertex-group weights, float32).
Settings: 0.05 m, 30 degrees, flipped normals allowed, Point mode, 4 groups, no smoothing (the add-on defaults).
--saved: the pipeline's weights.npz (pos, part, part_names, idx_robust, w_robust: 4 influences per vertex after
its top-4 step, applied twice: weights < 1e-3 dropped, rows renormalised, float32); compared after the same
steps.
--sensitivity: also how far each of our choices moves the weights from the add-on's (fresh normals, mesh
Laplacian, plain top-k, smoothing, 2 cm).
"""
import argparse
import json
import sys
import time

import numpy as np

import weightxfer as wx

SETTINGS = dict(max_dist=0.05, max_angle_deg=30.0, flip_normals=True)


def top4(W):
    """The pipeline's step (parts_weights.py): 4 largest, < 1e-3 dropped, renormalised, float32; the pipeline
    applies it twice (per part, and again when it writes weights.npz), so a weight that the first renormalisation
    takes below 1e-3 is dropped by the second."""
    return _top4(_top4(W))


def _top4(W):
    idx = np.argsort(-W, 1, kind="stable")[:, :4]
    w = np.take_along_axis(W, idx, 1)
    w[w < 1e-3] = 0
    s = w.sum(1, keepdims=True)
    w = np.where(s > 0, w / np.maximum(s, 1e-12), 0).astype(np.float32)
    out = np.zeros(W.shape)
    np.put_along_axis(out, idx, w.astype(np.float64), 1)
    return out


def run(z, p, **kw):
    args = dict(SETTINGS, src_normals=z["sn"].astype(np.float64), dst_normals=z[f"tn_{p}"].astype(np.float64),
                normalize=False)
    args.update(kw)
    md, ma, fl = args.pop("max_dist"), args.pop("max_angle_deg"), args.pop("flip_normals")
    return wx.transfer(z["sv"].astype(np.float64), z["st"].astype(np.int64), z["src_w"].astype(np.float64),
                       z[f"tv_{p}"].astype(np.float64), z[f"tt_{p}"].astype(np.int64), md, ma, fl, **args)


def diff(A, B):
    d = np.abs(A - B)
    return {"max": float(d.max()), "mean": float(d.mean()), "rows_over_1e-6": int((d.max(1) > 1e-6).sum()),
            "rows_over_0.01": int((d.max(1) > 0.01).sum())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("addon_io")
    ap.add_argument("--saved")
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    z = np.load(a.addon_io)
    parts = [str(p) for p in z["parts"]]
    sv = np.load(a.saved) if a.saved else None
    res = {"parts": {}, "sensitivity": {}}
    print(f"{'part':12s} {'verts':>6s} {'matched':>8s} {'add-on':>7s}  vs add-on raw: max / mean       "
          f"vs saved top-4: max / mean   seconds")
    for p in parts:
        t = time.perf_counter()
        W, m, rep = run(z, p)
        sec = time.perf_counter() - t
        WT = z[f"w_tool_{p}"].astype(np.float64)
        r = {"vertices": len(W), "matched": int(m.sum()), "fallback": rep["fallback"], "raw": diff(W, WT),
             "seconds": round(sec, 2)}
        line = f"{p:12s} {len(W):6d} {int(m.sum()):8d} {'':>7s}  {r['raw']['max']:10.3g} / {r['raw']['mean']:9.3g}"
        if sv is not None:
            k = [str(x) for x in sv["part_names"]].index(p)
            s = sv["part"] == k
            S = np.zeros((int(s.sum()), len(sv["bones"])))
            np.put_along_axis(S, sv["idx_robust"][s].astype(np.int64), sv["w_robust"][s].astype(np.float64), 1)
            r["saved"] = diff(top4(W), S)
            line += f"   {r['saved']['max']:10.3g} / {r['saved']['mean']:9.3g}"
        print(line + f"   {sec:6.2f}", flush=True)
        res["parts"][p] = r
    if a.sensitivity:
        variants = {
            "fresh part normals (from the welded triangles)": dict(dst_normals=None),
            "fresh part normals + source normals from the triangles": dict(dst_normals=None, src_normals=None),
            "mesh Laplacian": dict(laplacian_kind="mesh"),
            "plain top-4 (no smoothed remove mask)": dict(limit_dilate=0),
            "smoothing 10 x 0.2 (reference code default)": dict(smooth={"iterations": 10, "alpha": 0.2}),
            "max distance 2 cm": dict(max_dist=0.02),
            "max distance 10 cm": dict(max_dist=0.10),
            "no flipped normals": dict(flip_normals=False),
        }
        for name, kw in variants.items():
            rows = {}
            for p in parts:
                W, m, rep = run(z, p, **kw)
                WT = z[f"w_tool_{p}"].astype(np.float64)
                d = diff(top4(W), top4(WT))
                rows[p] = {"matched": int(m.sum()), "fallback": rep["fallback"], "max": d["max"], "mean": d["mean"],
                           "rows_over_0.01": d["rows_over_0.01"], "laplacian": rep["laplacian"]}
            res["sensitivity"][name] = rows
            print(f"\n{name}: after top-4, vs the add-on")
            for p, r in rows.items():
                fb = f"  FALLBACK {r['fallback']['vertices']} vertices ({r['fallback']['reason']})" if r["fallback"] else ""
                print(f"  {p:12s} matched {r['matched']:6d}  max {r['max']:.3g}  mean {r['mean']:.3g}  "
                      f"rows > 0.01: {r['rows_over_0.01']}{fb}", flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    main()
    sys.exit(0)
