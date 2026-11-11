"""Forward kinematics of the trained body and reference resampling (from the showcase exporter)."""
import mujoco
import numpy as np


class FK:
  """Root pose + named joint angles -> every body's world pose, on the trained body
  (coupled joints, e.g. the patellae, follow their equality polynomials)."""

  def __init__(self, m):
    self.m, self.d = m, mujoco.MjData(m)
    self.adr = {m.joint(j).name: int(m.jnt_qposadr[j]) for j in range(1, m.njnt)}
    self.eqs = [(int(m.jnt_qposadr[m.eq_obj1id[e]]),
                 int(m.jnt_qposadr[m.eq_obj2id[e]]) if m.eq_obj2id[e] >= 0 else -1, m.eq_data[e, :5].copy())
                for e in range(m.neq) if m.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT]
    self.coupled = {m.joint(int(m.eq_obj1id[e])).name for e in range(m.neq)
                    if m.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT}

  def __call__(self, root_pos, root_quat, joints, names):
    m, d, q0 = self.m, self.d, self.m.qpos0
    cols = [(self.adr[n], i) for i, n in enumerate(names) if n in self.adr and n not in self.coupled]
    F = len(root_pos)
    P, Q, qp = np.zeros((F, m.nbody, 3)), np.zeros((F, m.nbody, 4)), np.zeros((F, m.nq))
    for f in range(F):
      q = q0.copy()
      q[:3], q[3:7] = root_pos[f], root_quat[f]
      for a, i in cols:
        q[a] = joints[f, i]
      for _ in range(4):
        for a1, a2, c in self.eqs:
          x = q[a2] - q0[a2] if a2 >= 0 else 0.0
          q[a1] = q0[a1] + c[0] + c[1] * x + c[2] * x**2 + c[3] * x**3 + c[4] * x**4
      d.qpos[:] = q
      mujoco.mj_kinematics(m, d)
      P[f], Q[f], qp[f] = d.xpos, d.xquat, q
    return P, Q, qp


def resample_ref(mo, po, c, n_out, ref_fps, sim_fps):
  """Reference root pose and joints of clip c at the rollout rate (as compare_rollouts.load_ours)."""
  st, n0 = int(mo["length_starts"][c]), int(mo["motion_num_frames"][c])
  rp, rq, rj = mo["root_pos"][st:st + n0], mo["root_quat"][st:st + n0], po["qpos_joints"][st:st + n0]
  t = np.arange(n_out) * ref_fps / sim_fps
  i0 = np.minimum(np.floor(t).astype(int), n0 - 1)
  i1 = np.minimum(i0 + 1, n0 - 1)
  w = np.clip(t - i0, 0, 1)[:, None]
  q1 = rq[i1] * np.sign(np.sum(rq[i0] * rq[i1], -1, keepdims=True))
  rq = (1 - w) * rq[i0] + w * q1
  rq /= np.linalg.norm(rq, axis=-1, keepdims=True)
  return (1 - w) * rp[i0] + w * rp[i1], rq, (1 - w) * rj[i0] + w * rj[i1]
