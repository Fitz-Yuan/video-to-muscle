"""Per-frame camera of the source video in the motion's world frame, so the 3D view can film the simulation from the
same viewpoint as the video.

GVHMR gives the body twice: in the camera frame (smpl_params_incam) and in a gravity-aligned Y-up world
(smpl_params_global). The root orientation gives the camera rotation, R_cw = R_incam R_global^T, and the pelvis
joint (transl + rest pelvis J0; the root rotates about it) gives the camera centre c = p_world - R_cw^T p_cam.
cam_to_pelvis (= R_cw^T p_cam) is the camera->pelvis vector as the image sees it; the site anchors the camera on
the simulated reference's pelvis with it, so the pelvis lands where it is in the video. Both are then mapped to Z-up exactly as smpl_to_amass_npz.py maps the motion ((x, y, z) -> (x, -z, y)).
OpenCV camera axes: +z looks forward, -y is up. Vertical field of view from K_fullimg.

Run with GVHMR's python from GVHMR's folder (needs hmr4d + smplx):
  cd ~/Projects/GVHMR && ~/miniconda3/envs/gvhmr/bin/python <this> hmr4d_results.pt VIDEO.mp4 OUT.npz
"""
import subprocess
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation as sR

res, video, out = sys.argv[1:4]
from hmr4d.utils.smplx_utils import make_smplx  # noqa: E402  (GVHMR)

r = torch.load(res, map_location="cpu", weights_only=False)
g, c = r["smpl_params_global"], r["smpl_params_incam"]
betas = g["betas"].float().mean(0, keepdim=True)
z = lambda n: torch.zeros(1, n)  # noqa: E731
with torch.no_grad():
  J0 = make_smplx("supermotion")(body_pose=z(63), betas=betas, global_orient=z(3), transl=z(3)).joints[0, 0].double().numpy()
Rg = sR.from_rotvec(g["global_orient"].double().numpy())
Rc = sR.from_rotvec(c["global_orient"].double().numpy())
Rcw = Rc * Rg.inv()                          # world (Y-up) -> camera
pw = g["transl"].double().numpy() + J0       # pelvis joint, world
pc = c["transl"].double().numpy() + J0       # pelvis joint, camera
cen = pw - Rcw.inv().apply(pc)               # camera centre, world
fwd = Rcw.inv().apply([0, 0, 1.0])          # camera axes in world
up = Rcw.inv().apply([0, -1.0, 0])
Z = sR.from_euler("x", 90, degrees=True)     # Y-up -> Z-up, as smpl_to_amass_npz.py
K = r["K_fullimg"][0].double().numpy()
wh = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                     "-of", "csv=p=0", video], capture_output=True, text=True).stdout.strip().split(",")
W, H = int(wh[0]), int(wh[1])
fov = float(np.degrees(2 * np.arctan(H / 2 / K[1, 1])))
np.savez(out, cam_pos=Z.apply(cen), cam_fwd=Z.apply(fwd), cam_up=Z.apply(up), pelvis=Z.apply(pw),
         cam_to_pelvis=Z.apply(Rcw.inv().apply(pc)),
         fov_deg=fov, width=W, height=H, fps=30.0)
print(f"{out}: {len(cen)} frames, fov {fov:.1f} deg, {W}x{H}, camera height {Z.apply(cen)[:, 2].mean():.2f} m "
      f"(pelvis {Z.apply(pw)[:, 2].mean():.2f} m), distance {np.linalg.norm(cen - pw, axis=1).mean():.2f} m")
