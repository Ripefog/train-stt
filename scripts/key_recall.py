#!/usr/bin/env python3
"""Key-recall: % từ khóa đặc trưng của ref xuất hiện đúng trong hyp.

Định nghĩa (đã dùng cho bảng benchmark trong README.md):
  - key = từ (whitespace token) xuất hiện ở ≤ 3% số cut của val (df ≤ max(2, 0.03·N))
    — tức từ hiếm/đặc trưng, không phải stopword phổ thông.
  - key-recall = tổng (key của ref ∩ từ hyp) / tổng key của ref, gộp toàn corpus.

Cách chạy:
  python3 scripts/key_recall.py eval_v4_avg_avg-96-100.jsonl \
      eval_v4_avg_avg-96-100.int8.jsonl eval_v4_int4.jsonl

File jsonl: dòng 1 là {"_modes": [...]}, các dòng sau {"id", "ref", "hyp_<mode>": ...}.
Tập key được tính trên ref của FILE ĐẦ TIÊN để mọi bundle dùng cùng chuẩn so sánh.
"""
import collections
import json
import sys

import jiwer

RARE = 0.03
MODES = ["greedy", "beam16+hw+bp1"]


def load(path):
    lines = open(path, encoding="utf-8").read().splitlines()
    return [json.loads(l) for l in lines[1:]]


def main():
    paths = sys.argv[1:]
    rows0 = load(paths[0])
    df = collections.Counter()
    for d in rows0:
        df.update(set(d["ref"].split()))
    thresh = max(2, RARE * len(rows0))
    key_sets = [set(w for w in d["ref"].split() if df[w] <= thresh) for d in rows0]

    print(f"val: {len(rows0)} cut, tổng key: {sum(map(len, key_sets)):,}, "
          f"ngưỡng hiếm: df ≤ {thresh:.0f} cut")
    print(f"{'file':46} {'mode':18} {'WER':>6} {'CER':>6} {'key-recall':>10}")
    for path in paths:
        rows = rows0 if path == paths[0] else load(path)
        for m in MODES:
            refs = [d["ref"] for d in rows]
            hyps = [d.get("hyp_" + m, "") for d in rows]
            hit = tot = 0
            for d, ks in zip(rows, key_sets):
                hw = set(d.get("hyp_" + m, "").split())
                hit += len(ks & hw)
                tot += len(ks)
            print(f"{path:46} {m:18} {100*jiwer.wer(refs, hyps):6.2f} "
                  f"{100*jiwer.cer(refs, hyps):6.2f} {100*hit/tot:9.2f}%")


if __name__ == "__main__":
    main()
