#!/usr/bin/env python3
"""Probe âm học: thuật ngữ bị sai là do ENCODER không nghe ra, hay do SEARCH không lôi ra?

Bối cảnh (results.md): recall thuật ngữ trần ~45%. beam cho đúng +0,0, hotwords chỉ
+2,8 pp, lr sweep bão hoà 43–45% dù lr chênh 20 lần. Nên ~55% còn sai KHÔNG nằm trong
tầm với của tầng ngôn ngữ — nhưng chưa có phép đo trực tiếp nào nói vì sao.

Phép đo ở đây KHÔNG train, KHÔNG decode lại: nó ép model đi theo REF (teacher forcing)
rồi đọc xem tại đúng bước phát ra token của thuật ngữ, model cho token đúng bao nhiêu
điểm so với token nó thích nhất.

  1. Căn REF vào audio bằng Viterbi trên lattice RNN-T đầy đủ (không prune):
     alpha[t,u] = max(alpha[t-1,u] + logP(blank|t,u), alpha[t,u-1] + logP(y_u|t,u-1)).
     Backtrack cho biết token y_u được phát ở frame nào.
  2. Ở đúng (t,u) đó, đọc phân phối joiner đầy đủ trên vocab:
       gap = logP(token tốt nhất khác blank) − logP(token ĐÚNG)
     gap = 0  ⇒ model đã chọn đúng token, chỉ blank/search làm hỏng.
     gap lớn ⇒ model không hề coi token đúng là ứng viên ⇒ phía âm học.
  3. So gap với NGÂN SÁCH BOOST của hotwords: ContextGraph cộng `context_score` cho
     mỗi token của cụm. Nếu tổng gap của cụm < tổng boost thì về nguyên tắc hotwords
     LẬT ĐƯỢC vị trí đó ⇒ lỗi nằm ở danh sách/điểm số, không phải ở tai model.
     Nếu gap ≫ boost thì không mức boost hợp lý nào cứu được ⇒ phải chạm encoder.

Chứng cứ phụ: head CTC (chưa từng finetune, đóng băng qua mọi run) ở cửa sổ frame đã
căn — nếu CTC cũng không thấy token đúng thì hai head độc lập cùng nói "không nghe ra".

Nhóm ĐỐI CHỨNG là bắt buộc: script đo cả những lần thuật ngữ ĐÚNG. Không có nó thì
"gap = 3,2" không đọc được — phải biết gap của ca đúng để lấy làm mốc.

CHỈ ĐỌC checkpoint + val. Không ghi vào exp/.
"""
import argparse
import collections
import json
import logging
import re
import sys
from pathlib import Path

import sentencepiece as spm
import torch

sys.path.insert(0, ".")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import train as T  # noqa: E402  (train.py đã được copy vào cwd)
import metrics as M  # noqa: E402
from asr_datamodule import LibriSpeechAsrDataModule  # noqa: E402
from icefall.checkpoint import load_checkpoint  # noqa: E402


def get_args():
    p = T.get_parser()
    LibriSpeechAsrDataModule.add_arguments(p)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--dump", type=str, required=True,
                   help="JSONL của eval_beam.py — dùng để biết vị trí nào SAI, "
                        "khỏi phải decode lại")
    p.add_argument("--mode", type=str, default="beam16+hw",
                   help="mode trong dump dùng để phân loại hit/miss")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--word-mapping", type=str, default="/app/word_mapping.json")
    p.add_argument("--variants", type=str, default="/app/hotword_variants.tsv")
    p.add_argument("--min-duration", type=float, default=1.0)
    p.add_argument("--u-chunk", type=int, default=8,
                   help="số bước nhãn tính joiner mỗi lần (chặn peak memory)")
    p.add_argument("--ctc-window", type=int, default=2,
                   help="cửa sổ ±N frame khi tra posterior CTC")
    p.add_argument("--boost", type=float, default=3.0,
                   help="context_score mỗi token của hotword — ngân sách để so gap. "
                        "hotwords.txt của repo đang dùng :3 cho cụm wifi.")
    return p.parse_args()


# ── ánh xạ span ký tự -> span token BPE ──────────────────────────────────────
def piece_spans(sp, text):
    """[(start,end)] theo ký tự của TEXT cho từng piece của sp.encode(text).

    SentencePiece thêm một '▁' ở đầu (khoảng trắng ảo) — phải trừ ra, nếu không
    mọi offset lệch 1 và span token của thuật ngữ trượt sang từ bên cạnh.
    """
    pieces = sp.encode(text, out_type=str)
    rec, spans = "", []
    for pc in pieces:
        s = pc.replace("▁", " ")
        spans.append((len(rec), len(rec) + len(s)))
        rec += s
    off = 1 if rec.startswith(" ") else 0
    spans = [(a - off, b - off) for a, b in spans]
    if rec[off:] != text:
        return None, pieces          # normalize của SP đổi text ⇒ bỏ cut này
    return spans, pieces


def tokens_of_span(spans, c0, c1):
    """chỉ số token phủ [c0,c1) — dùng nửa mở, chạm biên không tính."""
    return [i for i, (a, b) in enumerate(spans) if a < c1 and b > c0]


# ── Viterbi trên lattice RNN-T đầy đủ ────────────────────────────────────────
@torch.no_grad()
def align_and_score(model, enc, y, u_chunk):
    """Căn y vào enc, trả về (emit_frame[u], gap[u], lp_correct[u], top1[u]).

    enc: (T, C) encoder output của MỘT cut. y: list token id (không có blank).
    Trả gap theo token ĐÚNG so với token tốt nhất KHÁC BLANK tại chính (t,u) đó.
    """
    device = enc.device
    Tn, U = enc.shape[0], len(y)
    blank = model.decoder.blank_id

    sos_y = torch.tensor([[blank] + y], dtype=torch.int64, device=device)
    dec = model.decoder(sos_y)                       # (1, U+1, D)
    am = model.joiner.encoder_proj(enc.unsqueeze(0))  # (1, T, C)
    lm = model.joiner.decoder_proj(dec)               # (1, U+1, C)

    # logP(blank | t,u) và logP(y_{u+1} | t,u) cho mọi (t,u) — chỉ giữ 2 số/ô.
    lp_blank = torch.full((Tn, U + 1), -1e9, device=device)
    lp_sym = torch.full((Tn, U + 1), -1e9, device=device)
    for u0 in range(0, U + 1, u_chunk):
        u1 = min(u0 + u_chunk, U + 1)
        logits = model.joiner(am[:, :, None, :], lm[:, None, u0:u1, :],
                              project_input=False)          # (1,T,chunk,V)
        lp = torch.log_softmax(logits.float(), dim=-1)[0]    # (T,chunk,V)
        lp_blank[:, u0:u1] = lp[:, :, blank]
        for j, u in enumerate(range(u0, u1)):
            if u < U:
                lp_sym[:, u] = lp[:, j, y[u]]
        del logits, lp

    b = lp_blank.cpu().numpy()
    s = lp_sym.cpu().numpy()

    NEG = float("-inf")
    alpha = [[NEG] * (U + 1) for _ in range(Tn + 1)]
    bp = [[0] * (U + 1) for _ in range(Tn + 1)]   # 0 = tới từ blank, 1 = từ emit
    alpha[0][0] = 0.0
    for t in range(Tn + 1):
        for u in range(U + 1):
            if t > 0 and alpha[t - 1][u] > NEG:
                cand = alpha[t - 1][u] + b[t - 1][u]
                if cand > alpha[t][u]:
                    alpha[t][u], bp[t][u] = cand, 0
            if u > 0 and t < Tn and alpha[t][u - 1] > NEG:
                cand = alpha[t][u - 1] + s[t][u - 1]
                if cand > alpha[t][u]:
                    alpha[t][u], bp[t][u] = cand, 1
    if alpha[Tn][U] == NEG:
        return None

    emit = [-1] * U
    t, u = Tn, U
    while t > 0 or u > 0:
        if bp[t][u] == 1:
            emit[u - 1] = t          # y_u phát ở frame t (0-index)
            u -= 1
        else:
            t -= 1
    return emit, float(alpha[Tn][U])


@torch.no_grad()
def row_stats(model, enc, dec_out, t, y_true):
    """Phân phối joiner đầy đủ tại một ô (t,u) → hạng + gap của token đúng."""
    blank = model.decoder.blank_id
    am = model.joiner.encoder_proj(enc[t][None, None, None, :])
    lm = model.joiner.decoder_proj(dec_out[None, None, None, :])
    lp = torch.log_softmax(
        model.joiner(am, lm, project_input=False).float(), dim=-1).reshape(-1)
    lp_true = lp[y_true].item()
    lp2 = lp.clone()
    lp2[blank] = -1e9                       # ứng viên tốt nhất KHÁC blank
    top1 = int(lp2.argmax())
    rank = int((lp2 > lp_true).sum()) + 1   # hạng trong nhóm không-blank
    return {"lp_true": lp_true, "lp_top1": lp2[top1].item(), "top1": top1,
            "rank": rank, "lp_blank": lp[blank].item()}


def main():
    args = get_args()
    args.return_cuts = True
    params = T.get_params()
    params.update(vars(args))
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    # ── thuật ngữ + nhóm biến thể: dùng ĐÚNG logic của metrics.py ────────────
    wm, tsv = Path(args.word_mapping), Path(args.variants)
    terms, term2key, ftel = M.load_terms(wm)
    term2group, group2strs = M.load_groups(tsv, wm, term2key, M.MIN_VARIANT_LEN)
    rx_exact = re.compile(r"(?<!\w)(" + "|".join(re.escape(t) for t in terms) + r")(?!\w)")
    all_strs = sorted(set(terms) | set(term2group), key=len, reverse=True)
    rx_any = re.compile(r"(?<!\w)(" + "|".join(re.escape(t) for t in all_strs) + r")(?!\w)")

    lines = Path(args.dump).read_text(encoding="utf-8").splitlines()
    cuts_dump = [json.loads(x) for x in lines[1:]]
    key = "hyp_" + args.mode
    if key not in cuts_dump[0]:
        raise SystemExit(f"dump không có mode {args.mode!r}: {list(cuts_dump[0])}")

    # Mỗi LẦN XUẤT HIỆN trong ref là một mục đo. hit/miss theo nhóm tương đương
    # (prod map biến thể về canonical — xem CLAUDE.md).
    targets = {}
    n_hit = n_miss = 0
    for c in cuts_dump:
        ref, hyp = M.norm(c["ref"]), M.norm(c[key])
        occ = [(m.group(1), m.start(1), m.end(1)) for m in rx_exact.finditer(ref)]
        if not occ:
            continue
        cg = collections.Counter()
        for s, n in M.counts(rx_any, hyp).items():
            cg[term2group.get(s, s)] += n
        seen = collections.Counter()
        items = []
        for term, c0, c1 in occ:
            g = term2group.get(term, term)
            seen[g] += 1
            hit = seen[g] <= cg.get(g, 0)
            items.append({"term": term, "c0": c0, "c1": c1, "hit": bool(hit),
                          "ftel": term in ftel})
            n_hit += hit
            n_miss += not hit
        targets[c["id"]] = {"ref": ref, "hyp": hyp, "items": items}
    logging.info(f"{len(targets)} cut có thuật ngữ — {n_hit} lần ĐÚNG / {n_miss} lần SAI "
                 f"(mode={args.mode})")

    # ── model ────────────────────────────────────────────────────────────────
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)
    params.blank_id = sp.piece_to_id("<blk>")
    params.sos_id = params.eos_id = sp.piece_to_id("<sos/eos>")
    params.vocab_size = sp.get_piece_size()
    params.dtype = torch.float32
    params.use_autocast = False
    model = T.get_model(params)
    load_checkpoint(params.ckpt, model)
    model.to(device).eval()
    logging.info(f"device={device} ckpt={params.ckpt}")

    ls = LibriSpeechAsrDataModule(args)
    cs = ls.dev_clean_cuts().to_eager()
    if params.min_duration > 0:
        cs = cs.filter(lambda c: c.duration >= params.min_duration).to_eager()
    cs = cs.filter(lambda c: c.id in targets).to_eager()
    logging.info(f"probe trên {len(cs)} cut")
    dl = ls.valid_dataloaders(cs)

    rows, n_skip = [], 0
    with torch.no_grad():
        for batch in dl:
            feats = batch["inputs"].to(device)
            flens = batch["supervisions"]["num_frames"].to(device)
            ids = [c.id for c in batch["supervisions"]["cut"]]
            enc_b, enc_lens = model.forward_encoder(feats, flens)
            ctc_b = model.ctc_output(enc_b)          # đã là log-softmax

            for i, cid in enumerate(ids):
                tgt = targets.get(cid)
                if tgt is None:
                    continue
                ref = tgt["ref"]
                spans, pieces = piece_spans(sp, ref)
                if spans is None:
                    n_skip += 1
                    continue
                y = sp.encode(ref)
                if len(y) != len(pieces):
                    n_skip += 1
                    continue
                enc = enc_b[i, : enc_lens[i]]
                ctc = ctc_b[i, : enc_lens[i]]
                res = align_and_score(model, enc, y, args.u_chunk)
                if res is None:
                    n_skip += 1
                    continue
                emit, total_lp = res

                sos_y = torch.tensor([[model.decoder.blank_id] + y],
                                     dtype=torch.int64, device=device)
                dec = model.decoder(sos_y)[0]        # (U+1, D)

                for it in tgt["items"]:
                    tk = tokens_of_span(spans, it["c0"], it["c1"])
                    if not tk:
                        n_skip += 1
                        continue
                    per = []
                    for u in tk:
                        t = emit[u]
                        if t < 0 or t >= enc.shape[0]:
                            continue
                        st = row_stats(model, enc, dec[u], t, y[u])
                        lo = max(0, t - args.ctc_window)
                        hi = min(ctc.shape[0], t + args.ctc_window + 1)
                        st["ctc_lp"] = float(ctc[lo:hi, y[u]].max())
                        st["ctc_top1"] = int(ctc[t].argmax())
                        st["t"] = int(t)
                        st["piece"] = pieces[u]
                        st["gap"] = st["lp_top1"] - st["lp_true"]
                        per.append(st)
                    if not per:
                        n_skip += 1
                        continue
                    gap_sum = sum(p["gap"] for p in per)
                    rows.append({
                        "id": cid, "term": it["term"], "hit": it["hit"],
                        "ftel": it["ftel"], "n_tok": len(per),
                        "gap_sum": gap_sum, "gap_max": max(p["gap"] for p in per),
                        "rank_max": max(p["rank"] for p in per),
                        "budget": args.boost * len(per),
                        "reachable": gap_sum < args.boost * len(per),
                        "ctc_lp_min": min(p["ctc_lp"] for p in per),
                        "align_lp": total_lp,
                        "tokens": [{k: (round(v, 3) if isinstance(v, float) else v)
                                    for k, v in p.items()} for p in per],
                        "ref": ref, "hyp": tgt["hyp"],
                    })
            logging.info(f"  ... {len(rows)} vị trí đã probe")

    # ── tổng hợp ─────────────────────────────────────────────────────────────
    def agg(sel):
        g = [r["gap_sum"] for r in rows if sel(r)]
        if not g:
            return None
        g = sorted(g)
        return {"n": len(g), "median": g[len(g) // 2],
                "mean": sum(g) / len(g),
                "p90": g[int(0.9 * (len(g) - 1))],
                "reachable": sum(1 for r in rows if sel(r) and r["reachable"]),
                "rank1": sum(1 for r in rows if sel(r) and r["rank_max"] == 1)}

    summary = {
        "mode": args.mode, "ckpt": args.ckpt, "boost_per_token": args.boost,
        "n_probed": len(rows), "n_skipped": n_skip,
        "MISS": agg(lambda r: not r["hit"]),
        "HIT (đối chứng)": agg(lambda r: r["hit"]),
        "MISS_ftel": agg(lambda r: not r["hit"] and r["ftel"]),
    }
    out = Path(args.out)
    with out.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"_summary": summary}, ensure_ascii=False) + "\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n" + "=" * 78)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 78)
    print(f"{'nhóm':<18}{'n':>5}{'gap trung vị':>15}{'lật được bằng boost':>24}")
    for k in ("MISS", "HIT (đối chứng)", "MISS_ftel"):
        a = summary[k]
        if a:
            print(f"{k:<18}{a['n']:>5}{a['median']:>15.2f}"
                  f"{a['reachable']:>15} / {a['n']}")
    print(f"\nghi {len(rows)} vị trí → {out}")


if __name__ == "__main__":
    main()
