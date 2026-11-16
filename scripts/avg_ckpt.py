#!/usr/bin/env python3
"""Average checkpoint kiểu icefall: trung bình key `model` của N checkpoint.

Output chỉ có key `model` (+ `epoch` informative) — đúng convention của
`average_checkpoints` icefall: load bằng `load_checkpoint` cho warm-start/eval,
KHÔNG chứa optimizer/scheduler (resume từ nó là warm-start, không phải resume thật).

Dùng: python3 avg_ckpt.py --inputs a.pt b.pt c.pt --out avg.pt
"""
import argparse
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    avg = None
    for i, f in enumerate(a.inputs):
        ck = torch.load(f, map_location="cpu", weights_only=False)
        sd = ck["model"]
        if avg is None:
            avg = {k: v.to(torch.float64) for k, v in sd.items()}
        else:
            for k in avg:
                avg[k] += sd[k].to(torch.float64)
    n = len(a.inputs)
    out_sd = {k: (v / n).to(torch.float32) for k, v in avg.items()}
    # batchnorm/dtypes số nguyên (num_batches_tracked) nếu có thì lấy từ file cuối
    last = torch.load(a.inputs[-1], map_location="cpu", weights_only=False)["model"]
    for k, v in last.items():
        if not torch.is_floating_point(v):
            out_sd[k] = v
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": out_sd, "epoch": None, "params": a.inputs}, a.out)
    print(f"✅ {a.out} = trung bình {n} checkpoint: {a.inputs}")


if __name__ == "__main__":
    main()
