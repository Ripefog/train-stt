#!/usr/bin/env python3
"""Dò vùng nghi "nuốt chữ" (deletion) trong audio, KHÔNG cần transcript tham chiếu.

Khác `probe_acoustic.py`: probe cần reference để teacher-force, nên chỉ chạy được trên
val có label. Script này chỉ dùng tín hiệu tự sinh của model + tín hiệu acoustic thô, nên
chạy được trên audio bất kỳ (prod, call ghi âm, file lẻ).

5 detector độc lập, mỗi cái bắt một mode lỗi khác nhau:

  D1 bp-delta       greedy bp=0 vs greedy bp=HI. Chỗ nào bp cao mọc thêm token = chỗ
                    model đang nén blank. Nhắm thẳng vào deletion.
  D2 hw-delta       beam vs beam+hotwords. Chỗ nào hotword làm mọc token = branch đúng
                    CÓ trong beam nhưng thua score → lỗi lexical bias, không phải acoustic.
  D3 blank-hesit    Dọc greedy path: frame nào p(blank) cao NHƯNG entropy của phân bố
                    non-blank cũng cao = model "im lặng trong lúc phân vân". Khác hẳn
                    silence thật (p_blank cao + entropy thấp).
  D4 ctc-vs-rnnt    CTC head phát symbol ở frame mà RNN-T không phát gì. CTC ít bị LM
                    bias hơn nên bắt được thứ RNN-T nuốt. Hai head dùng CHUNG trục frame
                    encoder ⇒ align miễn phí, không phải warp timestamp.
  D5 nbest-var      Phương sai số token/bin thời gian giữa 16 hypothesis trong beam.
                    Bin nào các hyp cãi nhau về số âm tiết = vùng bất ổn.
  D6 dur-outlier    Khoảng cách giữa 2 lần phát token liên tiếp. Tiếng Việt đơn âm tiết
                    nên một token "dài" bất thường thường là 2 âm tiết bị gộp 1.
  D7 nucleus-gap    Đếm nhân âm tiết bằng envelope năng lượng (băng 300-3000Hz). Tiếng
                    Việt gần như 1 nhân = 1 âm tiết, nên nhân nào không có token phủ là
                    ứng viên bị nuốt. Thuần acoustic, không dùng model.
  D8 vad-gap        Vùng có tiếng (energy VAD) mà không token nào phủ.

⚠️ Timestamp của RNN-T là **thời điểm PHÁT token**, không phải biên âm tiết. Transducer
phát trễ so với acoustic (thường 100-300ms) và độ trễ đó không hằng số. Nên D6 đo
inter-emission interval, chỉ là proxy của độ dài âm tiết — dùng để xếp hạng nghi ngờ,
đừng đọc như số đo thời lượng thật.

⚠️ Ngưỡng của cả 5 detector đang là giá trị ĐOÁN, chưa hiệu chuẩn trên val có label.
Precision/recall thật sự bao nhiêu thì chưa biết — phải chạy bước validate trên val
(có reference → tính deletion thật bằng align) rồi mới chỉnh ngưỡng.
"""
import argparse
import difflib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, ".")
sys.path.insert(0, "/app/scripts")

import train as T  # noqa: E402
from beam_search import greedy_search_batch, modified_beam_search  # noqa: E402
from icefall.checkpoint import load_checkpoint  # noqa: E402
from infer_wav import _BeamCapture, build_context_graph, load_fbank  # noqa: E402

SAMPLE_RATE = 16000
FRAME = 0.04  # 1 frame encoder sau subsampling ×4 = 40ms


def get_args():
    p = T.get_parser()
    p.add_argument("--wav", type=str, nargs="+", required=True)
    p.add_argument("--ckpt", type=str, nargs="+", required=True,
                   help="nhan:duong_dan hoac duong_dan")
    p.add_argument("--beam", type=int, default=16)
    p.add_argument("--bp-hi", type=float, default=2.0,
                   help="muc blank penalty cao dung cho D1 (doi chieu voi bp=0)")
    p.add_argument("--hotwords", type=str, default="")
    p.add_argument("--context-score", type=float, default=2.0)
    p.add_argument("--dur-k", type=float, default=2.0,
                   help="D6: nguong = median + k*MAD cua inter-emission interval")
    p.add_argument("--vad-gap", type=float, default=0.15,
                   help="D8: vung co tieng khong token nao phu, dai hon nay thi bao")
    p.add_argument("--nbest-k", type=float, default=2.0,
                   help="D5: nguong = mean + k*std cua do lech so token giua cac hyp")
    p.add_argument("--hesit-k", type=float, default=2.5,
                   help="D3: nguong entropy = median + k*MAD, tren cac frame p_blank cao")
    p.add_argument("--hesit-pblank", type=float, default=0.7,
                   help="D3: chi xet frame co p(blank) tren muc nay")
    p.add_argument("--emit-lag", type=float, default=0.35,
                   help="D4/D7/D8: cua so sau moc acoustic de tim token RNN-T. Transducer "
                        "phat TRE so voi acoustic, khong noi bien thi bao oan hang loat.")
    p.add_argument("--nucleus-prom", type=float, default=3.0,
                   help="D7: do noi toi thieu (dB) de mot dinh envelope tinh la nhan am tiet")
    p.add_argument("--json", type=str, default="",
                   help="ghi ket qua day du ra file JSON")
    return p.parse_args()


# ---------------------------------------------------------------- tiện ích chung
def syllables(sp, tokens, timestamps):
    """Gộp BPE piece thành âm tiết. Piece mở đầu bằng '▁' là đầu âm tiết mới.

    Trả list dict {text, t (giây, thời điểm phát piece đầu), n_piece}.
    """
    out = []
    for tok, ts in zip(tokens, timestamps):
        piece = sp.id_to_piece(tok)
        if piece.startswith("▁") or not out:
            out.append({"text": piece.lstrip("▁"), "t": ts * FRAME, "n_piece": 1})
        else:
            out[-1]["text"] += piece
            out[-1]["n_piece"] += 1
    return out


def robust(vals):
    """median + MAD (scale về sigma tương đương). Dùng thay mean/std vì n nhỏ."""
    v = np.asarray(vals, dtype=float)
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826
    return med, (mad if mad > 1e-6 else float(v.std()) or 1e-6)


def align_gain(sp, base_toks, hi_toks, hi_ts):
    """Token mà bản `hi` có thêm so với `base`. Trả [(thời điểm, chuỗi thêm)].

    difflib trên dãy piece — không dùng text, vì text đã mất ranh giới token.
    """
    a = [sp.id_to_piece(t) for t in base_toks]
    b = [sp.id_to_piece(t) for t in hi_toks]
    out = []
    for tag, _, _, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag in ("insert", "replace") and j2 > j1:
            t = hi_ts[j1] * FRAME if j1 < len(hi_ts) else float("nan")
            out.append({"t": round(t, 2),
                        "gain": "".join(b[j1:j2]).replace("▁", " ").strip(),
                        "kind": tag})
    return out


# ------------------------------------------------- D3: greedy có dump phân bố
def greedy_probe(model, enc, enc_len, blank_id, context_size):
    """Chạy lại greedy nhưng giữ phân bố xác suất từng frame.

    Sao y `greedy_search_batch` (beam_search.py) cho N=1, bỏ phần pack/unsort. Phải sao
    chứ không gọi lại được: hàm gốc chỉ trả logit của token ĐƯỢC CHỌN, mà D3 cần cả
    phân bố để tính entropy.
    """
    hyp = [-1] * (context_size - 1) + [blank_id]
    dec_in = torch.tensor([hyp], device=enc.device, dtype=torch.int64)
    dec_out = model.joiner.decoder_proj(model.decoder(dec_in, need_pad=False))
    enc_proj = model.joiner.encoder_proj(enc[0])          # (T, C)

    frames = []
    for t in range(enc_len):
        logits = model.joiner(enc_proj[t].view(1, 1, 1, -1),
                              dec_out.unsqueeze(1), project_input=False)
        logits = logits.squeeze(1).squeeze(1)             # (1, V)
        p = logits.softmax(-1)[0]
        p_blank = float(p[blank_id])
        q = p.clone()
        q[blank_id] = 0
        s = float(q.sum())
        ent = 0.0
        if s > 1e-9:
            q = q / s
            nz = q[q > 1e-9]
            ent = float(-(nz * nz.log()).sum() / np.log(len(p) - 1))  # chuẩn hoá 0..1
        top = int(q.argmax())
        frames.append({"t": t * FRAME, "p_blank": p_blank, "ent": ent,
                       "top": top, "p_top": float(q[top]) * (1 - p_blank)})

        y = int(logits.argmax(-1))
        if y != blank_id:
            hyp.append(y)
            dec_in = torch.tensor([hyp[-context_size:]], device=enc.device,
                                  dtype=torch.int64)
            dec_out = model.joiner.decoder_proj(model.decoder(dec_in, need_pad=False))
    return frames


def d3_blank_hesit(frames, sp, p_thr, k, emitted_ts):
    """Frame p_blank cao + entropy non-blank cao = do dự, khác silence thật.

    So entropy trong NỘI BỘ nhóm frame blank-dominant của chính file này, không dùng
    ngưỡng tuyệt đối — mức entropy nền phụ thuộc độ ồn của từng file.
    """
    cand = [f for f in frames if f["p_blank"] > p_thr]
    if len(cand) < 8:
        return []
    med, sig = robust([f["ent"] for f in cand])
    thr = med + k * sig
    hot = [f for f in cand if f["ent"] > thr]

    emitted = np.asarray(emitted_ts) * FRAME if emitted_ts else np.array([])
    out, run = [], []
    for f in hot + [None]:                       # gộp frame liền nhau thành 1 vùng
        if run and (f is None or f["t"] - run[-1]["t"] > 2.5 * FRAME):
            best = max(run, key=lambda x: x["ent"])
            if len(run) >= 2 and not (emitted.size and
                                      (abs(emitted - best["t"]) < 0.08).any()):
                out.append({"det": "D3 blank-hesit", "t": round(run[0]["t"], 2),
                            "why": f"{len(run)} frame p_blank={best['p_blank']:.2f} nhung "
                                   f"entropy={best['ent']:.3f} (nguong {thr:.3f}); "
                                   f"muon noi '{sp.id_to_piece(best['top'])}'"})
            run = []
        if f is not None:
            run.append(f)
    return out


# ------------------------------------------------- D4: CTC head vs RNN-T head
def ctc_emissions(model, enc, enc_len, blank_id):
    """CTC greedy: argmax mỗi frame, gộp lặp + bỏ blank. Trả [(frame, token_id)]."""
    lp = model.ctc_output(enc)[0][:enc_len]       # (T, V), đã là log-softmax
    best = lp.argmax(-1).tolist()
    out, prev = [], -1
    for t, s in enumerate(best):
        if s != blank_id and s != prev:
            out.append((t, s))
        prev = s
    return out


def d4_ctc_vs_rnnt(ctc, rnnt_ts, sp, lag):
    """CTC phát symbol mà RNN-T không phát gì quanh đó.

    Cửa sổ lệch [-0.08, +lag] vì RNN-T phát trễ hơn CTC, không đối xứng.
    """
    r = np.asarray(rnnt_ts) * FRAME if rnnt_ts else np.array([])
    out = []
    for t, s in ctc:
        tc = t * FRAME
        if r.size and ((r >= tc - 0.08) & (r <= tc + lag)).any():
            continue
        out.append({"det": "D4 ctc-vs-rnnt", "t": round(tc, 2),
                    "why": f"CTC phat '{sp.id_to_piece(s)}' ma RNN-T im"})
    return out


# ---------------------------------------------------------- D7/D8: acoustic
def frame_energy(path, band=(300.0, 3000.0)):
    """RMS dB theo frame 25ms/hop 10ms, sau khi lọc băng bằng FFT.

    Lọc băng để envelope bám nhân âm tiết (nguyên âm) chứ không bám âm xát/nhiễu nền.
    Làm bằng rFFT thẳng cho khỏi phụ thuộc scipy.signal.
    """
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    x = torch.from_numpy(data.T)[:1]
    if sr != SAMPLE_RATE:
        x = torchaudio.functional.resample(x, sr, SAMPLE_RATE)
    x = x[0].numpy().astype(np.float64)
    dur = len(x) / SAMPLE_RATE

    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SAMPLE_RATE)
    Xb = X.copy()
    Xb[(f < band[0]) | (f > band[1])] = 0
    xb = np.fft.irfft(Xb, n=len(x))

    win, hop = int(0.025 * SAMPLE_RATE), int(0.010 * SAMPLE_RATE)
    n = 1 + max(0, (len(x) - win) // hop)
    def rms_db(sig):
        fr = np.lib.stride_tricks.as_strided(
            sig, shape=(n, win), strides=(sig.strides[0] * hop, sig.strides[0]))
        return 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-8)
    return rms_db(np.ascontiguousarray(x)), rms_db(np.ascontiguousarray(xb)), dur


def find_nuclei(db_band, prom, min_sep=0.08, hop=0.010):
    """Đỉnh envelope băng thoại = nhân âm tiết.

    Nhận đỉnh theo thứ tự cao→thấp, loại đỉnh quá gần đỉnh đã nhận, rồi bỏ đỉnh nào
    không đủ "nổi" so với đáy giữa nó và đỉnh liền kề.
    """
    sm = np.convolve(db_band, np.ones(5) / 5, mode="same")   # làm mượt 50ms
    floor = np.percentile(sm, 10)
    cand = [i for i in range(2, len(sm) - 2)
            if sm[i] == sm[max(0, i - 4):i + 5].max() and sm[i] > floor + 6]
    keep = []
    for i in sorted(cand, key=lambda j: -sm[j]):
        if all(abs(i - j) >= min_sep / hop for j in keep):
            keep.append(i)
    keep.sort()
    out = []
    for idx, i in enumerate(keep):
        ok = True
        for j in (keep[idx - 1] if idx else None, keep[idx + 1] if idx + 1 < len(keep) else None):
            if j is None:
                continue
            valley = sm[min(i, j):max(i, j) + 1].min()
            if sm[i] - valley < prom and sm[j] > sm[i]:
                ok = False
        if ok:
            out.append(i * hop)
    return out


def d7_nucleus_gap(nuclei, syls, lag):
    ts = np.array([s["t"] for s in syls]) if syls else np.array([])
    return [{"det": "D7 nucleus-gap", "t": round(t, 2),
             "why": "nhan am tiet khong co token nao phu"}
            for t in nuclei
            if not (ts.size and ((ts >= t - 0.08) & (ts <= t + lag)).any())]


# ---------------------------------------------------------------- D8: energy VAD
def energy_vad(path, thr_db=8.0, min_voiced=0.10, close_gap=0.10):
    """VAD bằng năng lượng thuần (numpy). Không dùng webrtcvad để khỏi thêm dependency.

    Ngưỡng = sàn nhiễu (percentile 10 của RMS dB) + thr_db. Thô, nhưng đủ cho việc
    chỉ cần biết "chỗ này có tiếng người hay im".
    """
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    x = torch.from_numpy(data.T)[:1]
    if sr != SAMPLE_RATE:
        x = torchaudio.functional.resample(x, sr, SAMPLE_RATE)
    x = x[0].numpy()

    win, hop = int(0.025 * SAMPLE_RATE), int(0.010 * SAMPLE_RATE)
    n = 1 + max(0, (len(x) - win) // hop)
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n, win), strides=(x.strides[0] * hop, x.strides[0]))
    db = 20 * np.log10(np.sqrt((frames ** 2).mean(1)) + 1e-8)
    voiced = db > (np.percentile(db, 10) + thr_db)

    regions, i = [], 0
    while i < n:
        if not voiced[i]:
            i += 1
            continue
        j = i
        while j < n and voiced[j]:
            j += 1
        regions.append([i * 0.010, (j - 1) * 0.010 + 0.025])
        i = j
    merged = []
    for r in regions:  # nối 2 vùng cách nhau < close_gap
        if merged and r[0] - merged[-1][1] < close_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [r for r in merged if r[1] - r[0] >= min_voiced], len(x) / SAMPLE_RATE


# ---------------------------------------------------------------- 5 detector
def d1_d2_delta(sp, base_toks, hi_toks, hi_ts, tag, note):
    return [{"det": tag, "t": g["t"], "why": f"{note}: +'{g['gain']}'"}
            for g in align_gain(sp, base_toks, hi_toks, hi_ts)]


def d5_nbest_var(hyps, enc_len, k, bin_s=0.24):
    """Phương sai số token/bin giữa các hypothesis còn trong beam.

    Bin nào các hyp bất đồng nhiều về số âm tiết = vùng model không chắc phải phát
    mấy chữ ở đó — chính là dấu hiệu của nuốt/gộp.
    """
    nb = max(1, int(np.ceil(enc_len * FRAME / bin_s)))
    counts = []
    for h in hyps:
        c = np.zeros(nb)
        for ts in h.timestamp:
            c[min(nb - 1, int(ts * FRAME / bin_s))] += 1
        counts.append(c)
    if len(counts) < 2:
        return []
    std = np.asarray(counts).std(0)
    thr = std.mean() + k * std.std()
    return [{"det": "D5 nbest-var", "t": round(i * bin_s, 2),
             "why": f"16 hyp lech nhau {std[i]:.2f} token (nguong {thr:.2f})"}
            for i in range(nb) if std[i] > thr and std[i] > 0.4]


def d6_dur_outlier(syls, enc_len, k):
    if len(syls) < 4:
        return []
    ends = [s["t"] for s in syls[1:]] + [enc_len * FRAME]
    dur = [e - s["t"] for s, e in zip(syls, ends)]
    med, sig = robust(dur)
    thr = med + k * sig
    return [{"det": "D6 dur-outlier", "t": round(s["t"], 2),
             "why": f"'{s['text']}' keo {d:.2f}s (median {med:.2f}s, nguong {thr:.2f}s)"}
            for s, d in zip(syls, dur) if d > thr]


def d8_vad_gap(syls, regions, min_gap, lag):
    """Vùng có tiếng mà không token nào rơi vào. Nới biên về sau vì transducer phát
    trễ — không nới thì mọi vùng cuối câu đều bị báo oan."""
    ts = np.array([s["t"] for s in syls]) if syls else np.array([])
    out = []
    for a, b in regions:
        if b - a < min_gap:
            continue
        if ts.size and ((ts >= a - 0.05) & (ts <= b + lag)).any():
            continue
        out.append({"det": "D8 vad-gap", "t": round(a, 2),
                    "why": f"co tieng {a:.2f}-{b:.2f}s ({b - a:.2f}s) ma khong token nao phu"})
    return out


# ---------------------------------------------------------------- chạy 1 file
def run_file(model, sp, ctx, params, wav, device):
    feat = load_fbank(wav, device)
    x, x_lens = feat.unsqueeze(0), torch.tensor([feat.shape[0]], device=device)
    enc, enc_lens = model.forward_encoder(x, x_lens)
    enc_len = int(enc_lens[0])

    def greedy(bp):
        r = greedy_search_batch(model=model, encoder_out=enc, encoder_out_lens=enc_lens,
                                blank_penalty=bp, return_timestamps=True)
        return r.hyps[0], r.timestamps[0]

    def beam(cg, bp=0.0):
        with _BeamCapture() as cap:
            r = modified_beam_search(model=model, encoder_out=enc, encoder_out_lens=enc_lens,
                                     beam=params.beam, context_graph=cg, blank_penalty=bp,
                                     return_timestamps=True)
        return r.hyps[0], r.timestamps[0], cap.lists[0]

    g0_t, g0_ts = greedy(0.0)
    ghi_t, ghi_ts = greedy(params.bp_hi)
    b0_t, b0_ts, b0_list = beam(None)
    bhw_t, bhw_ts = (beam(ctx)[:2] if ctx is not None else (None, None))

    syls = syllables(sp, g0_t, g0_ts)          # audit chế độ prod = greedy bp=0
    regions, _ = energy_vad(wav)
    _, db_band, dur_s = frame_energy(wav)
    nuclei = find_nuclei(db_band, params.nucleus_prom)

    frames = greedy_probe(model, enc, enc_len, params.blank_id, params.context_size)
    ctc = ctc_emissions(model, enc, enc_len, params.blank_id) \
        if getattr(model, "use_ctc", False) else []

    hits = []
    hits += d1_d2_delta(sp, g0_t, ghi_t, ghi_ts, "D1 bp-delta",
                        f"bp={params.bp_hi:g} moc them")
    if bhw_t is not None:
        hits += d1_d2_delta(sp, b0_t, bhw_t, bhw_ts, "D2 hw-delta", "hotword moc them")
    hits += d3_blank_hesit(frames, sp, params.hesit_pblank, params.hesit_k, g0_ts)
    hits += d4_ctc_vs_rnnt(ctc, g0_ts, sp, params.emit_lag)
    hits += d5_nbest_var(list(b0_list.data.values()), enc_len, params.nbest_k)
    hits += d6_dur_outlier(syls, enc_len, params.dur_k)
    hits += d7_nucleus_gap(nuclei, syls, params.emit_lag)
    hits += d8_vad_gap(syls, regions, params.vad_gap, params.emit_lag)

    return {"wav": wav, "dur_s": round(dur_s, 2),
            "text_greedy": sp.decode(g0_t),
            "text_greedy_bphi": sp.decode(ghi_t),
            "text_beam": sp.decode(b0_t),
            "text_beam_hw": sp.decode(bhw_t) if bhw_t is not None else None,
            "text_ctc": sp.decode([s for _, s in ctc]) if ctc else None,
            "n_voiced_region": len(regions),
            "n_nucleus": len(nuclei), "n_ctc_sym": len(ctc),
            "syllables": syls,
            "hits": sorted(hits, key=lambda h: h["t"])}


def cluster(hits, win=0.5):
    """Gộp hit của các detector khác nhau nằm gần nhau (< win giây) thành 1 vùng.

    Số detector đồng thuận trong 1 vùng là tín hiệu mạnh hơn bất kỳ detector đơn lẻ nào
    — mỗi cái nhìn một mặt khác nhau, trùng nhau thì khó là ngẫu nhiên.
    """
    groups = []
    for h in sorted(hits, key=lambda x: x["t"]):
        if groups and h["t"] - groups[-1]["t0"] <= win:
            groups[-1]["t1"] = max(groups[-1]["t1"], h["t"])
            groups[-1]["hits"].append(h)
        else:
            groups.append({"t0": h["t"], "t1": h["t"], "hits": [h]})
    for g in groups:
        g["dets"] = sorted({h["det"].split()[0] for h in g["hits"]})
        g["score"] = len(g["dets"])
    return sorted(groups, key=lambda g: (-g["score"], g["t0"]))


def main():
    args = get_args()
    params = T.get_params()
    params.update(vars(args))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)
    params.blank_id = sp.piece_to_id("<blk>")
    params.sos_id = params.eos_id = sp.piece_to_id("<sos/eos>")
    params.vocab_size = sp.get_piece_size()
    params.dtype = torch.float32
    params.use_autocast = False

    # D4 cần head CTC, mà `--use-ctc` mặc định False và `load_checkpoint` dùng
    # strict=False ⇒ nếu không bật, ctc_output trong checkpoint bị BỎ IM LẶNG.
    # Chỉ bật khi checkpoint thật sự có weight đó, không thì D4 sẽ đọc weight random.
    params.use_ctc = True

    ctx = build_context_graph(params.hotwords, sp, params.context_score)[0] \
        if params.hotwords else None

    report = {}
    for spec in params.ckpt:
        name, _, path = spec.rpartition(":") if ":" in spec else ("", "", spec)
        name = name or Path(path).stem
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck.get("model", ck)
        has_ctc = any(k.startswith("ctc_output") for k in sd)
        del ck, sd
        params.use_ctc = has_ctc
        if not has_ctc:
            logging.warning(f"{name}: checkpoint KHONG co ctc_output → tat D4")

        model = T.get_model(params)
        load_checkpoint(path, model)
        model.to(device).eval()

        print(f"\n{'=' * 78}\n{name}   ({path})\n{'=' * 78}")
        report[name] = []
        with torch.no_grad():
            for wav in params.wav:
                res = run_file(model, sp, ctx, params, wav, device)
                report[name].append(res)

                print(f"\n--- {Path(wav).name}  ({res['dur_s']}s | RNN-T "
                      f"{len(res['syllables'])} am tiet | envelope {res['n_nucleus']} nhan "
                      f"| CTC {res['n_ctc_sym']} symbol | VAD {res['n_voiced_region']} vung) ---")
                print(f"  greedy      : {res['text_greedy']}")
                print(f"  greedy+bp{params.bp_hi:g}  : {res['text_greedy_bphi']}")
                print(f"  beam{params.beam}      : {res['text_beam']}")
                if res["text_beam_hw"]:
                    print(f"  beam{params.beam}+hw   : {res['text_beam_hw']}")
                if res["text_ctc"]:
                    print(f"  ctc greedy  : {res['text_ctc']}")

                groups = cluster(res["hits"])
                if not groups:
                    print("  → khong detector nao bao.")
                for g in groups:
                    near = [s["text"] for s in res["syllables"]
                            if g["t0"] - 0.6 <= s["t"] <= g["t1"] + 0.6]
                    print(f"  [{g['score']}/8] {g['t0']:5.2f}-{g['t1']:5.2f}s  "
                          f"{'+'.join(g['dets'])}   quanh do greedy nghe ra: "
                          f"\"{' '.join(near) or '(khong co gi)'}\"")
                    for h in g["hits"]:
                        print(f"          {h['det']:16s} {h['t']:5.2f}s  {h['why']}")
        del model
        torch.cuda.empty_cache()

    if params.json:
        Path(params.json).write_text(json.dumps(report, ensure_ascii=False, indent=1))
        print(f"\nJSON: {params.json}")


if __name__ == "__main__":
    main()
