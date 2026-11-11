"""ELVW container (the format of the mocap-to-myofullbody browser viewer).

b"ELVW", uint32 version, uint32 header length, UTF-8 JSON header
{"meta": {...}, "arrays": {name: [dtype, shape, byte offset]}}, padding to a multiple of 8,
then the arrays (little endian, each 8-byte aligned). Positions are int16 with
meta["pos_scale"] metres per unit; quaternions int16 / 32767 (w, x, y, z).
Copied from mocap-to-myofullbody/pipeline/export_viewer.py so this repository stands alone.
"""
import json
import os
import struct

import numpy as np

MISSING = -32768


def write_elvw(path, meta, arrays):
  index, blobs, off = {}, [], 0
  for name, a in arrays.items():
    a = np.ascontiguousarray(a)
    index[name] = [a.dtype.str.lstrip("<|"), list(a.shape), off]
    b = a.tobytes()
    blobs.append(b + b"\0" * (-len(b) % 8))
    off += len(blobs[-1])
  head = json.dumps({"meta": meta, "arrays": index}, separators=(",", ":")).encode()
  head += b" " * (-(12 + len(head)) % 8)
  tmp = path + ".part"
  with open(tmp, "wb") as f:
    f.write(b"ELVW" + struct.pack("<II", 1, len(head)) + head)
    for b in blobs:
      f.write(b)
  os.replace(tmp, path)
  return os.path.getsize(path)


def q16(x, scale):
  x = np.asarray(x, np.float64)
  y = np.round(x / scale)
  out = np.clip(np.nan_to_num(y, nan=MISSING), -32767, 32767).astype(np.int16)
  out[np.isnan(x).any(-1) if x.ndim > 1 else np.isnan(x)] = MISSING
  return out
