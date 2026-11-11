"""Export the trained body (MyoFullBody mm106, 354 muscles, no fingers) as the viewer's model file.

Bone meshes, the body tree, every muscle as its chain of attachment sites, the contact
capsules / ellipsoids as a rough body volume, and the rest pose (qpos0). Same layout as
mocap-to-myofullbody's export_viewer.py model files (format elvw-model-1).
Run with the mjlab_musculoskeletal venv (it builds the body exactly as training does):

  MYOFULLBODY_MODEL=mm106 MYOFULLBODY_PHANTOM_INERTIA=1e-2 \\
    ~/Projects/mjlab_musculoskeletal/.venv/bin/python tools/export_model.py viewer/models
"""
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elvw import write_elvw  # noqa: E402

MODEL_NAME = "mm106"


def body_model():
  assert os.environ.get("MYOFULLBODY_MODEL") == "mm106", "set MYOFULLBODY_MODEL=mm106"
  from myosuite_mjlab.tasks.liftbox.robot_cfg import human_model
  return human_model()


def export_model(m, out_dir):
  T = mujoco.mjtGeom
  bodies = [m.body(b).name for b in range(m.nbody)]
  parents = [int(m.body_parentid[b]) for b in range(m.nbody)]
  meshes, verts, faces = [], [], []
  vofs = fofs = 0
  for g in range(m.ngeom):
    if m.geom_type[g] != T.mjGEOM_MESH or m.geom_rgba[g][3] == 0:
      continue
    mi = int(m.geom_dataid[g])
    a, n = m.mesh_vertadr[mi], m.mesh_vertnum[mi]
    v = m.mesh_vert[a:a + n].astype(np.float64)
    fa, fn = m.mesh_faceadr[mi], m.mesh_facenum[mi]
    f = m.mesh_face[fa:fa + fn].astype(np.uint16)
    lo, hi = v.min(0), v.max(0)
    sc = np.maximum(hi - lo, 1e-6) / 65534.0
    verts.append(np.round((v - lo) / sc - 32767).astype(np.int16))
    faces.append(f)
    meshes.append(dict(name=m.geom(g).name, body=int(m.geom_bodyid[g]),
                       pos=[float(x) for x in m.geom_pos[g]], quat=[float(x) for x in m.geom_quat[g]],
                       lo=lo.tolist(), scale=sc.tolist(), vofs=vofs, nv=int(n), fofs=fofs, nf=int(fn),
                       rgba=[round(float(x), 3) for x in m.geom_rgba[g]]))
    vofs += int(n)
    fofs += int(fn)
  muscles, site_body, site_pos, site_index = [], [], [], {}
  for a in range(m.nu):
    if m.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_TENDON:
      continue
    t = int(m.actuator_trnid[a][0])
    chain = []
    for j in range(m.tendon_adr[t], m.tendon_adr[t] + m.tendon_num[t]):
      if m.wrap_type[j] == mujoco.mjtWrap.mjWRAP_SITE:
        s = int(m.wrap_objid[j])
        if s not in site_index:
          site_index[s] = len(site_body)
          site_body.append(int(m.site_bodyid[s]))
          site_pos.append(m.site_pos[s])
        chain.append(site_index[s])
    muscles.append(dict(name=m.actuator(a).name, sites=chain))
  kinds = {T.mjGEOM_CAPSULE: "capsule", T.mjGEOM_ELLIPSOID: "ellipsoid"}
  vol = [g for g in range(m.ngeom) if m.geom_type[g] in kinds and (m.geom_contype[g] or m.geom_conaffinity[g])]
  volumes = [dict(type=kinds[m.geom_type[g]], body=int(m.geom_bodyid[g]),
                  pos=[float(x) for x in m.geom_pos[g]], quat=[float(x) for x in m.geom_quat[g]],
                  size=[float(x) for x in m.geom_size[g]]) for g in vol]
  d = mujoco.MjData(m)
  mujoco.mj_kinematics(m, d)
  meta = dict(format="elvw-model-1", source="musclemimic-models 1.0.6 myofullbody.xml (mm106, no fingers)",
              bodies=bodies, parents=parents, meshes=meshes, muscles=muscles, volumes=volumes, up="z",
              model=MODEL_NAME, title="MyoFullBody (MuscleMimic 1.0.6) · 354 muscles")
  arrays = dict(verts=np.concatenate(verts), faces=np.concatenate(faces),
                site_body=np.array(site_body, np.uint16), site_pos=np.array(site_pos, np.float32),
                rest_pos=d.xpos.astype(np.float32), rest_quat=d.xquat.astype(np.float32))
  os.makedirs(out_dir, exist_ok=True)
  path = os.path.join(out_dir, f"{MODEL_NAME}_model.elvw")
  size = write_elvw(path, meta, arrays)
  print(f"{path}: {len(meshes)} meshes, {vofs} vertices, {fofs} faces, {len(muscles)} muscles, "
        f"{len(site_body)} sites, {len(volumes)} volumes, {size / 1e6:.2f} MB")


if __name__ == "__main__":
  export_model(body_model(), sys.argv[1])
