#!/usr/bin/env python3
"""Dump từng cặp (reference, hypothesis) của tập val + phân tích lỗi.

Mục đích: gap val−train ~29% trơ với can thiệp âm học (BAND_CLAMP đổi val WER 0,2 pp).
Script này trả lời: lỗi nằm ở ĐÂU — sai từ, thêm từ, hay mất từ; và có phải lệch quy
ước viết không.

CHỈ ĐỌC checkpoint + val. Không train, không ghi vào exp/.

Chạy trong container (xem lệnh ở cuối file). Dùng lại đúng machinery của train.py nên
số liệu so được trực tiếp với dòng "VALID WER" trong log training.

Khác một điểm CÓ CHỦ Ý so với train.py: ngoài macro-WER (trung bình WER từng batch —
cái train.py log) còn tính micro-WER (gộp toàn corpus, chuẩn công nghiệp). Hai con số
này lệch nhau khi độ dài câu không đều, và val có câu 0,4s lẫn 13s.
"""
import argparse
import collections
import gzip
import json
import logging
import sys
from pathlib import Path

import sentencepiece as spm
import torch

sys.path.insert(0, ".")

import train as T  # noqa: E402  (train.py đã được copy vào cwd)
from asr_datamodule import LibriSpeechAsrDataModule  # noqa: E402
from beam_search import greedy_search_batch  # noqa: E402
from icefall.checkpoint import load_checkpoint  # noqa: E402


def get_args():
    p = T.get_parser()
    LibriSpeechAsrDataModule.add_arguments(p)
    p.add_argument("--ckpt", type=str, required=True,
                   help="đường dẫn checkpoint, vd exp/epoch-35.pt")
    p.add_argument("--out", type=str, required=True, help="file JSONL đầu ra")
    p.add_argument("--min-duration", type=float, default=1.0,
                   help="giống EVAL_MIN_DURATION để so được với log training")
    p.add_argument("--show", type=int, default=40, help="in bao nhiêu cặp ra stdout")
    return p.parse_args()


def ctc_greedy(m, encoder_out, encoder_out_lens, sp):
    """CTC argmax → gộp lặp → bỏ blank. Copy nguyên logic train.py."""
    tokens_batch = m.ctc_output(encoder_out).argmax(dim=-1)
    out = []
    for toks, length in zip(tokens_batch, encoder_out_lens):
        prev, filt = -1, []
        for t in toks[:length].tolist():
            if t != 0 and t != prev:
                filt.append(t)
            prev = t
        out.append(sp.decode(filt))
    return out


def main():
    args = get_args()
    args.return_cuts = True          # cần cut.id để truy ngược file audio
    params = T.get_params()
    params.update(vars(args))

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    logging.info(f"device={device}  ckpt={params.ckpt}")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)
    params.blank_id = sp.piece_to_id("<blk>")
    params.sos_id = params.eos_id = sp.piece_to_id("<sos/eos>")
    params.vocab_size = sp.get_piece_size()
    params.dtype = torch.float32
    params.use_autocast = False

    model = T.get_model(params)
    load_checkpoint(params.ckpt, model)      # KHÔNG strict=False: muốn biết nếu lệch key
    model.to(device).eval()
    logging.info(f"params={sum(p.numel() for p in model.parameters())}")

    ls = LibriSpeechAsrDataModule(args)
    # val chỉ 549 cut nên to_eager() rẻ, và cần len() để log + để filter không lazy
    cuts = ls.dev_clean_cuts().to_eager()
    n0 = len(cuts)
    if params.min_duration > 0:
        cuts = cuts.filter(lambda c: c.duration >= params.min_duration).to_eager()
    logging.info(f"val: {n0} cut → {len(cuts)} cut (duration >= {params.min_duration}s)")
    dl = ls.valid_dataloaders(cuts)

    rows = []
    # macro = trung bình WER theo batch (train.py log cái này)
    macro = collections.defaultdict(float)
    n_batch = 0

    import jiwer
    with torch.no_grad():
        for batch in dl:
            feature = batch["inputs"].to(device)
            feature_lens = batch["supervisions"]["num_frames"].to(device)
            texts = batch["supervisions"]["text"]
            cut_ids = [c.id for c in batch["supervisions"]["cut"]]
            srcs = [c.recording.sources[0].source for c in batch["supervisions"]["cut"]]
            durs = [c.duration for c in batch["supervisions"]["cut"]]

            keep = (feature_lens >= 20).cpu()
            if not keep.all():
                k = keep.tolist()
                feature = feature[keep.to(device)]
                feature_lens = feature_lens[keep.to(device)]
                texts = [t for t, v in zip(texts, k) if v]
                cut_ids = [t for t, v in zip(cut_ids, k) if v]
                srcs = [t for t, v in zip(srcs, k) if v]
                durs = [t for t, v in zip(durs, k) if v]
            if not texts:
                continue

            enc, enc_lens = model.forward_encoder(feature, feature_lens)
            hyps = {
                "ctc": ctc_greedy(model, enc, enc_lens, sp),
                "transducer": sp.decode(
                    greedy_search_batch(model=model, encoder_out=enc,
                                        encoder_out_lens=enc_lens)),
            }
            n_batch += 1
            for mo, hh in hyps.items():
                macro[mo] += jiwer.wer(
                    texts, hh,
                    reference_transform=jiwer.wer_standardize,
                    hypothesis_transform=jiwer.wer_standardize)

            for i, ref in enumerate(texts):
                rows.append({
                    "id": cut_ids[i], "dur": round(durs[i], 2),
                    "src": Path(srcs[i]).name, "ref": ref,
                    "hyp_ctc": hyps["ctc"][i],
                    "hyp_transducer": hyps["transducer"][i],
                })

    logging.info(f"decode xong {len(rows)} cut / {n_batch} batch")

    # ── phân tích: micro-WER + phân rã S/D/I trên toàn corpus ────────────────
    report = {}
    for mo in ("ctc", "transducer"):
        refs = [r["ref"] for r in rows]
        hyps = [r["hyp_" + mo] for r in rows]
        o = jiwer.process_words(
            refs, hyps,
            reference_transform=jiwer.wer_standardize,
            hypothesis_transform=jiwer.wer_standardize)
        nref = o.substitutions + o.deletions + o.hits
        report[mo] = {
            "micro_wer": o.wer * 100, "macro_wer": macro[mo] / n_batch * 100,
            "sub": o.substitutions, "del": o.deletions, "ins": o.insertions,
            "hit": o.hits, "n_ref_words": nref,
            "sub_pct": o.substitutions / nref * 100,
            "del_pct": o.deletions / nref * 100,
            "ins_pct": o.insertions / nref * 100,
        }
        # WER từng câu + gom cặp sai
        subs, dels, inss = collections.Counter(), collections.Counter(), collections.Counter()
        for ali, rw, hw in zip(o.alignments, o.references, o.hypotheses):
            for ch in ali:
                if ch.type == "substitute":
                    for a, b in zip(rw[ch.ref_start_idx:ch.ref_end_idx],
                                    hw[ch.hyp_start_idx:ch.hyp_end_idx]):
                        subs[(a, b)] += 1
                elif ch.type == "delete":
                    for a in rw[ch.ref_start_idx:ch.ref_end_idx]:
                        dels[a] += 1
                elif ch.type == "insert":
                    for b in hw[ch.hyp_start_idx:ch.hyp_end_idx]:
                        inss[b] += 1
        report[mo]["top_sub"] = subs.most_common(30)
        report[mo]["top_del"] = dels.most_common(20)
        report[mo]["top_ins"] = inss.most_common(20)

        for r, rw, hw in zip(rows, o.references, o.hypotheses):
            r["wer_" + mo] = (jiwer.process_words([" ".join(rw)], [" ".join(hw)]).wer
                              if rw else None)
            r["nw_" + mo] = len(rw)

    out = Path(params.out)
    with out.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"_report": report}, ensure_ascii=False) + "\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logging.info(f"ghi {out}")

    # ── in ra stdout ─────────────────────────────────────────────────────────
    for mo in ("ctc", "transducer"):
        d = report[mo]
        print(f"\n===== {mo.upper()} =====")
        print(f"  micro-WER (toàn corpus)      : {d['micro_wer']:.2f}%")
        print(f"  macro-WER (TB theo batch)    : {d['macro_wer']:.2f}%   <- train.py log cái này")
        print(f"  từ trong ref                 : {d['n_ref_words']}")
        print(f"  substitution {d['sub']:6d}  = {d['sub_pct']:5.2f}%")
        print(f"  deletion     {d['del']:6d}  = {d['del_pct']:5.2f}%")
        print(f"  insertion    {d['ins']:6d}  = {d['ins_pct']:5.2f}%")
        print(f"  top substitution (ref → hyp):")
        for (a, b), n in d["top_sub"][:20]:
            print(f"      {n:4d}x  {a!r:22s} → {b!r}")
        print(f"  top deletion (ref có, hyp mất): "
              + ", ".join(f"{w}({n})" for w, n in d["top_del"][:12]))
        print(f"  top insertion (hyp thêm)      : "
              + ", ".join(f"{w}({n})" for w, n in d["top_ins"][:12]))

    print(f"\n===== {params.show} cặp REF/HYP (xấu nhất theo transducer WER) =====")
    rr = sorted([r for r in rows if r.get("wer_transducer") is not None],
                key=lambda r: -r["wer_transducer"])
    for r in rr[:params.show]:
        print(f"\n[{r['dur']:5.2f}s  WER {r['wer_transducer']*100:6.1f}%  {r['nw_transducer']:3d} từ]  {r['src'][:40]}")
        print(f"  REF : {r['ref']}")
        print(f"  TDR : {r['hyp_transducer']}")
        print(f"  CTC : {r['hyp_ctc']}")

    print(f"\n===== 15 cặp NGẪU NHIÊN (WER trung bình, để thấy trường hợp thường) =====")
    mid = [r for r in rr if 0.05 < r["wer_transducer"] < 0.6]
    step = max(1, len(mid) // 15)
    for r in mid[::step][:15]:
        print(f"\n[{r['dur']:5.2f}s  WER {r['wer_transducer']*100:6.1f}%]  {r['src'][:40]}")
        print(f"  REF : {r['ref']}")
        print(f"  TDR : {r['hyp_transducer']}")

    nperf = sum(1 for r in rr if r["wer_transducer"] == 0)
    print(f"\ncâu đúng hoàn toàn (WER 0%): {nperf}/{len(rr)} = {nperf/len(rr)*100:.1f}%")


if __name__ == "__main__":
    main()
