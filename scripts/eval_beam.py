#!/usr/bin/env python3
"""So sánh một checkpoint qua nhiều chế độ decode transducer, báo cáo micro-WER.

Vì sao tách khỏi dump_val.py: dump_val chỉ chạy greedy. Ở đây encoder được forward
MỘT lần mỗi batch rồi dùng chung cho cả greedy lẫn mọi beam size — nên thêm beam gần
như chỉ tốn phần search, không tốn lại phần encoder.

Chỉ báo micro-WER (gộp toàn corpus). Macro bị bỏ có chủ ý: nó phụ thuộc cách chia
batch nên đổi MAX_DURATION là đổi số, không so được giữa các lần chạy.

Ghi JSONL per-cut để chạy paired bootstrap ở ngoài.
CHỈ ĐỌC checkpoint + val. Không train, không ghi vào exp/.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import sentencepiece as spm
import torch

sys.path.insert(0, ".")

import train as T  # noqa: E402  (train.py đã được copy vào cwd)
from asr_datamodule import LibriSpeechAsrDataModule  # noqa: E402
from beam_search import greedy_search_batch, modified_beam_search  # noqa: E402
from icefall.checkpoint import load_checkpoint  # noqa: E402
from icefall.context_graph import ContextGraph  # noqa: E402


def get_args():
    p = T.get_parser()
    LibriSpeechAsrDataModule.add_arguments(p)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out", type=str, required=True, help="file JSONL đầu ra")
    p.add_argument("--beams", type=str, default="4,8,16",
                   help="danh sách beam size cho modified_beam_search")
    p.add_argument("--blank-penalties", type=str, default="0",
                   help="DANH SÁCH blank_penalty, dấu PHẨY phân tách các giá trị — "
                        "'0,1,2' = chạy 3 mức bp=0, bp=1, bp=2 trong cùng một lần. "
                        "Thập phân dùng dấu CHẤM: '0.5' là một giá trị, còn '0,5' là "
                        "HAI giá trị (0 và 5). Trừ thẳng vào logit blank trước softmax "
                        "(beam_search.py: logits[:, 0] -= blank_penalty) → model nói "
                        "nhiều hơn: deletion giảm, insertion tăng.")
    p.add_argument("--min-duration", type=float, default=1.0,
                   help="giống EVAL_MIN_DURATION để so được với log training")
    p.add_argument("--hotwords", type=str, default="",
                   help="file hotwords kiểu sherpa-onnx: 'CỤM TỪ :score', "
                        "score bỏ trống = dùng --context-score")
    p.add_argument("--context-score", type=float, default=2.0,
                   help="điểm thưởng mặc định mỗi token của hotword")
    return p.parse_args()


def build_context_graph(path: str, sp, context_score: float):
    """Đọc file hotwords sherpa-onnx → ContextGraph.

    BẮT BUỘC lowercase: BPE của repo này đã được lowercase toàn bộ piece
    (entrypoint.sh bước 2), nên cụm viết HOA encode ra <unk> (id 2) sạch trơn —
    context graph khi đó vô dụng. Đã kiểm: 'ÉP PÊ TÊ' → [205,2,205,2,205,2].
    """
    phrases, token_ids, scores = [], [], []
    skipped = []
    unk = sp.piece_to_id("<unk>")
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        score = 0.0                      # 0 = dùng self.context_score
        if ":" in line:
            head, _, tail = line.rpartition(":")
            if tail.strip().replace(".", "", 1).isdigit():
                line, score = head.strip(), float(tail.strip())
        phrase = line.lower()
        ids = sp.encode(phrase)
        if not ids or unk in ids:
            skipped.append((raw.strip(), ids))
            continue
        phrases.append(phrase)
        token_ids.append(ids)
        scores.append(score)

    logging.info(f"hotwords: nạp {len(phrases)} cụm từ {path}"
                 + (f", BỎ {len(skipped)} cụm encode ra <unk>" if skipped else ""))
    for s, ids in skipped[:10]:
        logging.warning(f"  bỏ: {s!r} → {ids}")

    g = ContextGraph(context_score)
    g.build(token_ids=token_ids, phrases=phrases, scores=scores)
    return g, len(phrases)


def main():
    args = get_args()
    args.return_cuts = True
    params = T.get_params()
    params.update(vars(args))

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    beams = [int(b) for b in str(params.beams).split(",") if b.strip()]
    bps = [float(x) for x in str(params.blank_penalties).split(",") if x.strip()]

    # specs = (label, kind, beam, blank_penalty, dùng context_graph)
    # bp=0 giữ nhãn trơn để so trực tiếp được với các lần đo trước.
    def _sfx(bp):
        return "" if bp == 0 else f"+bp{bp:g}"
    specs = []
    for bp in bps:
        specs.append((f"greedy{_sfx(bp)}", "greedy", None, bp, False))
    for b in beams:
        for bp in bps:
            specs.append((f"beam{b}{_sfx(bp)}", "beam", b, bp, False))
        if params.hotwords:
            for bp in bps:
                specs.append((f"beam{b}+hw{_sfx(bp)}", "beam", b, bp, True))
    modes = [s[0] for s in specs]
    logging.info(f"device={device}  ckpt={params.ckpt}  {len(modes)} mode: {modes}")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)
    params.blank_id = sp.piece_to_id("<blk>")
    params.sos_id = params.eos_id = sp.piece_to_id("<sos/eos>")
    params.vocab_size = sp.get_piece_size()
    params.dtype = torch.float32
    params.use_autocast = False

    model = T.get_model(params)
    load_checkpoint(params.ckpt, model)   # KHÔNG strict=False: muốn biết nếu lệch key
    model.to(device).eval()
    logging.info(f"params={sum(p.numel() for p in model.parameters())}")

    ctx_graph = None
    if params.hotwords:
        ctx_graph, n_hw = build_context_graph(
            params.hotwords, sp, params.context_score)
        logging.info(f"context_score mặc định={params.context_score}, {n_hw} cụm")

    ls = LibriSpeechAsrDataModule(args)
    cuts = ls.dev_clean_cuts().to_eager()
    n0 = len(cuts)
    if params.min_duration > 0:
        cuts = cuts.filter(lambda c: c.duration >= params.min_duration).to_eager()
    logging.info(f"val: {n0} cut → {len(cuts)} cut (duration >= {params.min_duration}s)")
    dl = ls.valid_dataloaders(cuts)

    import jiwer
    rows = []
    with torch.no_grad():
        for batch in dl:
            feature = batch["inputs"].to(device)
            feature_lens = batch["supervisions"]["num_frames"].to(device)
            texts = batch["supervisions"]["text"]
            cut_ids = [c.id for c in batch["supervisions"]["cut"]]

            keep = (feature_lens >= 20).cpu()
            if not keep.all():
                k = keep.tolist()
                feature = feature[keep.to(device)]
                feature_lens = feature_lens[keep.to(device)]
                texts = [t for t, v in zip(texts, k) if v]
                cut_ids = [t for t, v in zip(cut_ids, k) if v]
            if not texts:
                continue

            # encoder chạy 1 lần, mọi mode dùng chung
            enc, enc_lens = model.forward_encoder(feature, feature_lens)
            hyps = {}
            for label, kind, b, bp, use_ctx in specs:
                if kind == "greedy":
                    toks = greedy_search_batch(
                        model=model, encoder_out=enc, encoder_out_lens=enc_lens,
                        blank_penalty=bp)
                else:
                    toks = modified_beam_search(
                        model=model, encoder_out=enc, encoder_out_lens=enc_lens,
                        beam=b, blank_penalty=bp,
                        context_graph=ctx_graph if use_ctx else None)
                hyps[label] = sp.decode(toks)

            for i, ref in enumerate(texts):
                r = {"id": cut_ids[i], "ref": ref}
                for mo in modes:
                    r["hyp_" + mo] = hyps[mo][i]
                rows.append(r)

    logging.info(f"decode xong {len(rows)} cut × {len(modes)} mode")

    report = {}
    for mo in modes:
        o = jiwer.process_words(
            [r["ref"] for r in rows], [r["hyp_" + mo] for r in rows],
            reference_transform=jiwer.wer_standardize,
            hypothesis_transform=jiwer.wer_standardize)
        n = o.substitutions + o.deletions + o.hits
        report[mo] = {"micro_wer": o.wer * 100, "sub": o.substitutions,
                      "del": o.deletions, "ins": o.insertions, "n_ref_words": n,
                      "sub_pct": o.substitutions / n * 100,
                      "del_pct": o.deletions / n * 100,
                      "ins_pct": o.insertions / n * 100}
        # per-cut để bootstrap ghép cặp ở ngoài
        for r, rw, hw in zip(rows, o.references, o.hypotheses):
            pw = jiwer.process_words([" ".join(rw) or "@"], [" ".join(hw) or "@"])
            r["err_" + mo] = pw.substitutions + pw.deletions + pw.insertions
            r["nw"] = len(rw)

    # ── Recall thuật ngữ ─────────────────────────────────────────────────────
    # WER tổng KHÔNG thấy được thuật ngữ: 108/6.687 từ = 1,6%, nằm dưới ngưỡng
    # nhiễu 0,45 pp. Lấy value NHIỀU ÂM TIẾT trong word_mapping.json — bỏ value
    # một âm tiết vì chúng trùng từ Việt thường ("không" khớp 380k cut train).
    import re as _re
    import collections as _c
    try:
        _wm = json.loads(Path("/app/word_mapping.json").read_text(encoding="utf-8"))
        _terms = sorted({v for k, v in _wm["words"].items()
                         if not k.startswith("_") and " " in v}, key=len, reverse=True)
        _rx = _re.compile(r"(?<!\w)(" + "|".join(_re.escape(t) for t in _terms) + r")(?!\w)")
        _ref_tot = 0
        for r in rows:
            _ref_tot += len(_rx.findall(r["ref"]))
        for mo in modes:
            hit = 0
            for r in rows:
                rc = _c.Counter(_rx.findall(r["ref"]))
                hc = _c.Counter(_rx.findall(r["hyp_" + mo]))
                for t, n in rc.items():
                    hit += min(n, hc.get(t, 0))
            report[mo]["term_hit"] = hit
            report[mo]["term_total"] = _ref_tot
            report[mo]["term_recall"] = hit / max(_ref_tot, 1) * 100
        logging.info(f"recall thuật ngữ: {_ref_tot} lần xuất hiện, {len(_terms)} cụm")
    except Exception as e:  # noqa: BLE001
        logging.warning(f"không tính được recall thuật ngữ: {e}")

    out = Path(params.out)
    with out.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"_report": report}, ensure_ascii=False) + "\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logging.info(f"ghi {out}")

    print(f"\n{'mode':<18s} {'micro-WER':>10s} {'sub%':>7s} {'del%':>7s} {'ins%':>7s}"
          f" {'thuật ngữ':>14s}")
    for mo in modes:
        d = report[mo]
        tr = (f"{d['term_hit']}/{d['term_total']} = {d['term_recall']:.1f}%"
              if "term_hit" in d else "—")
        print(f"{mo:<18s} {d['micro_wer']:9.2f}% {d['sub_pct']:6.2f}% "
              f"{d['del_pct']:6.2f}% {d['ins_pct']:6.2f}% {tr:>14s}")


if __name__ == "__main__":
    main()
