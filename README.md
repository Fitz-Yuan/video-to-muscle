# Video to Muscle

Everyday video, reproduced in physics by a muscle-driven full-body policy (MyoFullBody, 354 muscles).

The page shows each input video next to the simulation, filmed from the video's own camera. It holds results only
(simulated body poses, muscle activations and the camera path); the policy weights and training code are not released.

## Layout
- `site/` the static page (open `site/index.html` through any web server), the body model and `data/` (clips, videos, catalog).
- `tools/` the exporters used to build `site/data`:
  - `gvhmr_camera.py` the source video's camera from GVHMR (run in GVHMR's environment);
  - `export_site.py` one processed clip -> `site/data/clips/<id>.elvw`, the web video and poster, and a `catalog.json` entry;
  - `export_model.py`, `elvw.py`, `fk.py` the body model file and the container format.

## Preview locally
    cd site && python3 -m http.server 8000    # then open http://localhost:8000/
