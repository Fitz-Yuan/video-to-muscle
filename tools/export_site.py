"""One processed video clip -> the site's data: site/data/clips/<id>.elvw (simulated body, muscle activations, the
video's camera), site/data/video/<id>.mp4 + .jpg (web copy of the input video and its poster), and an entry in
site/data/catalog.json.

Inputs of a clip directory (projects/realtime_video/scripts/site_clip_v1.sh + site_policy_clip_v1.sh):
  <tag>.mp4                        the trimmed 30 fps input video
  vid_<tag>.amass.npz              GVHMR motion in AMASS form (pelvis = trans + J0h)
  camera.npz                       tools/gvhmr_camera.py
  policy/rollouts_<run>.npz        record_video_demo.py rollout of the final policy (100 Hz)

The camera comes in GVHMR's world; the simulation lives in the retargeted reference's world, which GMR shifted (put on
the ground). The constant shift is the mean of (reference root - GVHMR pelvis) over the clip.

Run with the mjlab_musculoskeletal venv (MYOFULLBODY_MODEL=mm106 MYOFULLBODY_PHANTOM_INERTIA=1e-2):
  python tools/export_site.py CLIP_DIR --id walk_dog --title "Walking the dog" --group walk [--run final_it24300]
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elvw import q16, write_elvw  # noqa: E402
from export_model import MODEL_NAME, body_model  # noqa: E402
from fk import FK, resample_ref  # noqa: E402

SITE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "site")
J0H = np.array([-0.0017950595290382927, -0.2233334459925592, 0.02821912553071819])  # smpl_to_amass_npz.py
FPS = 50.0
FALL_DROP, FALL_S = 0.3, 0.3


def smooth(x, n):
  """centred moving average along axis 0 (edges padded)"""
  if n <= 1:
    return x
  p = np.pad(x, [(n // 2, n - 1 - n // 2)] + [(0, 0)] * (x.ndim - 1), mode="edge")
  k = np.ones(n) / n
  return np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 0, p)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("clip_dir")
  ap.add_argument("--id", required=True)
  ap.add_argument("--title", required=True)
  ap.add_argument("--group", required=True, choices=["task", "walk", "jog"])
  ap.add_argument("--run", default="final_it24300")
  ap.add_argument("--source", default="Mixkit")
  ap.add_argument("--source-url", default="")
  ap.add_argument("--site", default=SITE, help="site folder to write into")
  ap.add_argument("--ghost", action="store_true", help="also store the video estimate (reference) as a ghost body")
  a = ap.parse_args()
  C = os.path.abspath(a.clip_dir)
  tag = os.path.basename(C)
  z = dict(np.load(f"{C}/policy/rollouts_{a.run}.npz", allow_pickle=True))
  mo = dict(np.load(str(z["motion_npz"]), allow_pickle=True))
  po = dict(np.load(str(z["poses_npz"]), allow_pickle=True))
  am = np.load(f"{C}/vid_{tag}.amass.npz", allow_pickle=True)
  cam = np.load(f"{C}/camera.npz")
  sim_fps, ref_fps = float(z["fps"]), float(mo["motion_fps"][0])
  step = int(round(sim_fps / FPS))

  m = body_model()
  fk = FK(m)
  jn = [str(x) for x in z["joint_names"]]
  pj = [str(x) for x in po["joint_names"]]
  n_ref = int(round((int(mo["motion_num_frames"][0]) - 1) * sim_fps / ref_fps)) + 1
  F = min(int(z["alive"][:, 0].astype(bool).sum()), n_ref)
  idx = np.arange(0, F, step)
  sp, sq, _ = fk(z["root_pos"][idx, 0], z["root_quat"][idx, 0], z["joint_pos"][idx, 0], jn)
  rp, rq, rj = resample_ref(mo, po, 0, F, ref_fps, sim_fps)
  gp, gq, _ = fk(rp[idx], rq[idx], rj[idx], pj)

  pel = m.body("pelvis").id
  low = (gp[:, pel, 2] - sp[:, pel, 2]) >= FALL_DROP
  need, run, fell = int(round(FALL_S * FPS)), 0, None
  for i, v in enumerate(low):
    run = run + 1 if v else 0
    if run >= need:
      fell = (i - need + 1) / FPS
      break

  # camera: GVHMR world -> reference world (constant shift), 30 fps -> 50 fps, lightly smoothed (0.2 s)
  pel_g = am["trans"] + J0H                                    # GVHMR pelvis, Z-up, 30 fps
  st = int(mo["length_starts"][0])
  t30 = np.arange(len(pel_g)) / 30.0
  ri = np.minimum(np.round(t30 * ref_fps).astype(int), int(mo["motion_num_frames"][0]) - 1)
  shift = (mo["root_pos"][st + ri] - pel_g).mean(0)  # reported only
  t = idx / sim_fps
  interp = lambda A: np.stack([np.interp(t, np.arange(len(A)) / 30.0, A[:, k]) for k in range(3)], 1)  # noqa: E731
  # anchor the camera on the reference pelvis (root) with the camera->pelvis vector of the in-camera estimate
  if "cam_to_pelvis" in cam.files:
    cpos = smooth(rp[idx] - interp(cam["cam_to_pelvis"]), 11)
  else:
    cpos = smooth(interp(cam["cam_pos"] + shift), 11)
  cfwd = smooth(interp(cam["cam_fwd"]), 11)
  cup = smooth(interp(cam["cam_up"]), 11)
  cfwd /= np.linalg.norm(cfwd, axis=1, keepdims=True)
  cup /= np.linalg.norm(cup, axis=1, keepdims=True)

  extent = max(np.abs(sp).max(), np.abs(gp).max())
  ps = max(2e-4, float(extent) / 32000)
  act = np.clip(np.asarray(z["act"][idx, 0], np.float32), 0, 1)
  act_names = [str(x).split("/")[-1] for x in z["act_names"]]
  arrays = dict(body_pos=q16(sp, ps), body_quat=q16(sq, 1 / 32767),
                muscle_act=np.round(act * 254).astype(np.uint8),
                effort=act.mean(1).astype(np.float32),
                cam_pos=cpos.astype(np.float32), cam_fwd=cfwd.astype(np.float32), cam_up=cup.astype(np.float32))
  if a.ghost:
    arrays.update(ghost_pos=q16(gp, ps), ghost_quat=q16(gq, 1 / 32767))
  vid_s = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                f"{C}/{tag}.mp4"], capture_output=True, text=True).stdout)
  info = dict(id=a.id, title=a.title, group=a.group, seconds=round(len(idx) / FPS, 2), video_s=round(vid_s, 2),
              fell_s=fell, source=a.source, source_url=a.source_url, width=int(cam["width"]), height=int(cam["height"]))
  meta = dict(format="elvw-clip-3", trial=a.id, frames=len(idx), fps=FPS, pos_scale=ps, model=MODEL_NAME,
              fov_deg=float(cam["fov_deg"]), aspect=float(cam["width"]) / float(cam["height"]),
              muscle_act=dict(array="muscle_act", muscles=act_names, scale=1 / 254),
              ghost="video estimate" if a.ghost else None, showcase=info)
  os.makedirs(f"{a.site}/data/clips", exist_ok=True)
  os.makedirs(f"{a.site}/data/video", exist_ok=True)
  size = write_elvw(f"{a.site}/data/clips/{a.id}.elvw", meta, arrays)
  # web copy of the input: H.264, no audio, fast start; poster at 1 s
  subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{C}/{tag}.mp4", "-an", "-vf", "scale='min(1280,iw)':-2",
                  "-c:v", "libx264", "-preset", "slow", "-crf", "25", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                  f"{a.site}/data/video/{a.id}.mp4"], check=True)
  subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "1", "-i", f"{C}/{tag}.mp4", "-frames:v", "1", "-vf",
                  "scale=480:-2", "-q:v", "4", f"{a.site}/data/video/{a.id}.jpg"], check=True)
  info.update(file=f"data/clips/{a.id}.elvw", video=f"data/video/{a.id}.mp4", poster=f"data/video/{a.id}.jpg",
              bytes=size)
  cat_path = f"{a.site}/data/catalog.json"
  cat = json.load(open(cat_path)) if os.path.exists(cat_path) else dict(clips=[])
  cat["clips"] = [c for c in cat["clips"] if c["id"] != a.id] + [info]
  json.dump(cat, open(cat_path, "w"), indent=1)
  print(f"{a.id}: {len(idx)} frames, {'fell at %.2f s' % fell if fell else 'no fall'}, {size / 1e6:.2f} MB, "
        f"camera shift {np.round(shift, 2)}, fov {float(cam['fov_deg']):.1f}")


if __name__ == "__main__":
  main()
