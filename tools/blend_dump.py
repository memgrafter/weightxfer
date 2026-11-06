"""Blender: dump a skinned mesh (and optionally a second mesh) to an .npz for weightxfer tests. Our own code.
  blender -b --factory-startup --python tools/blend_dump.py -- FILE.blend OUT.npz SOURCE_OBJ [TARGET_OBJ]
SOURCE_OBJ: a mesh with an Armature modifier. Writes, in world space of the EVALUATED mesh and of the base mesh:
  src_pos (evaluated), src_pos_base, src_tri (loop triangles), src_nrm (Blender vertex normals, evaluated),
  src_w (V x bones: raw vertex-group weights of the armature's bones, not normalised), bones, deform (bool per bone).
TARGET_OBJ (optional): tgt_pos, tgt_tri, tgt_nrm the same way (no weights).
"""
import sys
import numpy as np
import bpy

a = sys.argv[sys.argv.index("--") + 1:]
blend, out, sname = a[:3]
tname = a[3] if len(a) > 3 else None
bpy.ops.wm.open_mainfile(filepath=blend)
dg = bpy.context.evaluated_depsgraph_get()


def arrays(ob):
    ev = ob.evaluated_get(dg)
    me = ev.to_mesh()
    mw = np.array(ob.matrix_world)
    n = len(me.vertices)
    P = np.empty(n * 3); me.vertices.foreach_get("co", P); P = P.reshape(-1, 3)
    P = P @ mw[:3, :3].T + mw[:3, 3]
    N = np.empty(n * 3); me.vertex_normals.foreach_get("vector", N); N = N.reshape(-1, 3)
    N = N @ np.linalg.inv(mw[:3, :3])  # normal matrix (inverse transpose), row vectors
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-30)
    me.calc_loop_triangles()
    T = np.empty(len(me.loop_triangles) * 3, np.int64); me.loop_triangles.foreach_get("vertices", T)
    base = ob.data
    B = np.empty(len(base.vertices) * 3); base.vertices.foreach_get("co", B); B = B.reshape(-1, 3)
    B = B @ mw[:3, :3].T + mw[:3, 3]
    ev.to_mesh_clear()
    print("DUMP", ob.name, "verts", n, "tris", len(T) // 3, "modifiers", [(m.name, m.type) for m in ob.modifiers],
          "matrix_world identity", np.allclose(mw, np.eye(4)), flush=True)
    return P, T.reshape(-1, 3), N, B


so = bpy.data.objects[sname]
arm = next(m.object for m in so.modifiers if m.type == "ARMATURE")
bones = [b.name for b in arm.data.bones]
deform = np.array([b.use_deform for b in arm.data.bones])
P, T, N, B = arrays(so)
gi = {g.index: g.name for g in so.vertex_groups}
bi = {b: i for i, b in enumerate(bones)}
W = np.zeros((len(so.data.vertices), len(bones)))
for v in so.data.vertices:
    for g in v.groups:
        if gi[g.group] in bi:
            W[v.index, bi[gi[g.group]]] = g.weight
print("POSE identity", all(np.allclose(np.array(pb.matrix_basis), np.eye(4)) for pb in arm.pose.bones), flush=True)
out_d = dict(src_pos=P, src_pos_base=B, src_tri=T, src_nrm=N, src_w=W, bones=np.array(bones), deform=deform)
if tname:
    tp, tt, tn, _ = arrays(bpy.data.objects[tname])
    out_d.update(tgt_pos=tp, tgt_tri=tt, tgt_nrm=tn)
np.savez_compressed(out, **out_d)
print("WROTE", out, flush=True)
