#!/usr/bin/env python3
"""Infer batch: transcript mọi wav trong một thư mục bằng 1 checkpoint.

Load model MỘT lần rồi decode tuần tự (greedy + beam16 + hotwords tùy chọn),
ghi JSONL: {id, dur, hyp_greedy, hyp_beam16, hyp_beam16_hw}.

Dùng trong container, cwd = zipformer dir (như eval_beam.py):
  python3 /app/scripts/infer_batch.py --ckpt exp/avg-37-40.pt \
      --wav-dir /wavs --out /out/transcripts.jsonl --hotwords /app/hotwords.txt
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

import sentencepiece as spm
import torch

import train as T  # noqa: E402  (train.py đã cp vào cwd)
from beam_search import greedy_search_batch, modified_beam_search  # noqa: E402
from icefall.checkpoint import load_checkpoint  # noqa: E402
from icefall.context_graph import ContextGraph  # noqa: E402


def main():
    # Dùng parser của train.py làm nền — mọi default kiến trúc (num_encoder_layers,
    # dims, dropout…) khớp CHÍNH XÁC config đã train, không phải đoán lại.
    p = T.get_parser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--wav-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--hotwords", default="")
    p.add_argument("--context-score", type=float, default=2.0)
    a, _unknown = p.parse_known_args()

    params = T.get_params()
    params.update(vars(a))
    params.return_cuts = False
    device = torch.device("cuda", 0)

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

    ctx = None
    if params.hotwords:
        from eval_beam import build_context_graph  # noqa: E402
        ctx, n = build_context_graph(params.hotwords, sp, params.context_score)
        print(f"hotwords: {n} cụm", file=sys.stderr)

    import kaldifeat
    opts = kaldifeat.FbankOptions()
    opts.device = device
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = 16000
    opts.mel_opts.num_bins = 80
    opts.mel_opts.high_freq = -400
    extractor = kaldifeat.Fbank(opts)

    wavs = sorted(Path(params.wav_dir).glob("*.wav"))
    out_f = open(params.out, "w", encoding="utf-8")
    for i, w in enumerate(wavs):
        import soundfile as sf
        data, sr = sf.read(str(w), dtype="float32", always_2d=True)
        x = torch.from_numpy(data[:, 0]).unsqueeze(0)  # (1, T) — wav đã mono 16k
        assert sr == 16000, f"{w}: sr={sr}"
        feats = extractor(x[0].to(device))  # kaldifeat: (T, 80)
        feats = feats.unsqueeze(0)          # (1, T, 80) — như batch["inputs"]
        fl = torch.tensor([feats.shape[1]], device=device)

        with torch.no_grad():
            enc, enc_lens = model.forward_encoder(feats, fl)
            hyps = {}
            ids = greedy_search_batch(model=model, encoder_out=enc,
                                      encoder_out_lens=enc_lens)
            hyps["greedy"] = sp.decode(ids)[0]
            ids = modified_beam_search(model=model, encoder_out=enc,
                                       encoder_out_lens=enc_lens, beam=16)
            hyps["beam16"] = sp.decode(ids)[0]
            if ctx is not None:
                ids = modified_beam_search(model=model, encoder_out=enc,
                                           encoder_out_lens=enc_lens,
                                           beam=16, context_graph=ctx)
                hyps["beam16_hw"] = sp.decode(ids)[0]
        row = {"id": w.stem, "dur": round(x.shape[1] / 16000, 2), **hyps}
        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        out_f.flush()
        if (i + 1) % 20 == 0:
            print(f"{i+1}/{len(wavs)}", file=sys.stderr)
    out_f.close()
    print(f"✅ {params.out}: {len(wavs)} file", file=sys.stderr)


if __name__ == "__main__":
    main()
