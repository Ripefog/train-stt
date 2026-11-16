#!/usr/bin/env python3
"""
Export Zipformer transducer (epoch-6.pt) → 3 ONNX files.

Outputs (in --outdir):
  encoder-epoch-6.onnx / encoder-epoch-6.int8.onnx
  decoder-epoch-6.onnx / decoder-epoch-6.int8.onnx
  joiner-epoch-6.onnx  / joiner-epoch-6.int8.onnx

Usage:
  python scripts/export_onnx.py [--checkpoint PATH] [--outdir DIR] [--opset N]

Environment:
  ICEFALL_ZIPFORMER  path to icefall zipformer source dir
                     (default: /home/anlnm/.cache/gipformer/icefall/egs/librispeech/ASR/zipformer)
"""

import argparse
import contextlib
import logging
import os
import sys
import types
from pathlib import Path
from typing import Dict, Tuple

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
)

# ── 1. Icefall source path ──────────────────────────────────────────────────────
ICEFALL_ZF = os.environ.get(
    "ICEFALL_ZIPFORMER",
    "/home/anlnm/.cache/gipformer/icefall/egs/librispeech/ASR/zipformer",
)
if not os.path.isdir(ICEFALL_ZF):
    raise RuntimeError(
        f"icefall zipformer dir not found: {ICEFALL_ZF}\n"
        "Set env var ICEFALL_ZIPFORMER to correct path."
    )
if ICEFALL_ZF not in sys.path:
    sys.path.insert(0, ICEFALL_ZF)

# ── 2. k2 stub (k2 không cần cài để export ONNX) ───────────────────────────────
def _make_k2_stub() -> types.ModuleType:
    import torch

    k2 = types.ModuleType("k2")

    def _swoosh_l(x):
        zero = torch.zeros(1, dtype=x.dtype, device=x.device)
        return torch.logaddexp(zero, x - 4.0) - 0.08 * x - 0.035

    def _swoosh_r(x):
        zero = torch.zeros(1, dtype=x.dtype, device=x.device)
        return torch.logaddexp(zero, x - 1.0) - 0.08 * x - 0.313261687

    k2.swoosh_l = _swoosh_l
    k2.swoosh_r = _swoosh_r
    k2.swoosh_l_forward = _swoosh_l
    k2.swoosh_r_forward = _swoosh_r
    k2.swoosh_l_forward_and_deriv = lambda x, d: (_swoosh_l(x), None)
    k2.swoosh_r_forward_and_deriv = lambda x, d: (_swoosh_r(x), None)
    return k2


# ── 3. icefall.utils stub ───────────────────────────────────────────────────────
def _make_icefall_stubs():
    import torch

    icefall_mod = types.ModuleType("icefall")
    icefall_utils = types.ModuleType("icefall.utils")

    @contextlib.contextmanager
    def torch_autocast(enabled=True, dtype=torch.float32, device_type="cuda"):
        with torch.autocast(device_type=device_type, enabled=enabled, dtype=dtype):
            yield

    def make_pad_mask(lengths: "torch.Tensor", max_len: "int | None" = None):
        if max_len is None:
            max_len = int(lengths.max().item())
        indices = torch.arange(max_len, device=lengths.device)
        return indices >= lengths.unsqueeze(1)  # (B, T) True = padded

    def num_tokens(token_table) -> int:
        return max(int(v) for v in token_table._id2sym)

    icefall_utils.torch_autocast = torch_autocast
    icefall_utils.make_pad_mask = make_pad_mask
    icefall_utils.num_tokens = num_tokens
    icefall_mod.utils = icefall_utils

    icefall_checkpoint = types.ModuleType("icefall.checkpoint")
    icefall_checkpoint.load_checkpoint = None  # không dùng
    icefall_mod.checkpoint = icefall_checkpoint

    return icefall_mod, icefall_utils


sys.modules.setdefault("k2", _make_k2_stub())
_icefall_mod, _icefall_utils = _make_icefall_stubs()
sys.modules.setdefault("icefall", _icefall_mod)
sys.modules.setdefault("icefall.utils", _icefall_utils)
sys.modules.setdefault("icefall.checkpoint", _icefall_mod.checkpoint)

# ── 4. Import sau khi inject stubs ─────────────────────────────────────────────
import onnx
import torch
import torch.nn as nn
from scaling import (
    Balancer,
    Dropout3,
    ScaleGrad,
    ScheduledFloat,
    SwooshL,
    SwooshLOnnx,
    SwooshR,
    SwooshROnnx,
    Whiten,
)
from scaling_converter import convert_scaled_to_non_scaled
from subsampling import Conv2dSubsampling
from decoder import Decoder
from joiner import Joiner
from zipformer import CompactRelPositionalEncoding, Zipformer2

# ── 5. Model architecture (khớp với train.py defaults + checkpoint shape) ───────
VOCAB_SIZE     = 2000
BLANK_ID       = 0
CONTEXT_SIZE   = 2
DECODER_DIM    = 512
JOINER_DIM     = 512
ENCODER_DIM    = (192, 256, 384, 512, 384, 256)
ENCODER_DIM_MAX = max(ENCODER_DIM)  # 512


def build_model() -> nn.Module:
    dropout = ScheduledFloat((0.0, 0.3), (20000.0, 0.1))

    encoder_embed = Conv2dSubsampling(
        in_channels=80,
        out_channels=ENCODER_DIM[0],
        dropout=dropout,
    )

    encoder = Zipformer2(
        output_downsampling_factor=2,
        downsampling_factor=(1, 2, 4, 8, 4, 2),
        encoder_dim=ENCODER_DIM,
        num_encoder_layers=(2, 2, 3, 4, 3, 2),
        encoder_unmasked_dim=(192, 192, 256, 256, 256, 192),
        query_head_dim=32,
        pos_head_dim=4,
        value_head_dim=12,
        num_heads=(4, 4, 4, 8, 4, 4),
        feedforward_dim=(512, 768, 1024, 1536, 1024, 768),
        cnn_module_kernel=(31, 31, 15, 15, 15, 31),
        pos_dim=48,
        dropout=dropout,
        warmup_batches=4000.0,
        causal=False,
        chunk_size=(-1,),
        left_context_frames=(-1,),
    )

    decoder = Decoder(
        vocab_size=VOCAB_SIZE,
        decoder_dim=DECODER_DIM,
        blank_id=BLANK_ID,
        context_size=CONTEXT_SIZE,
    )

    joiner = Joiner(
        encoder_dim=ENCODER_DIM_MAX,
        decoder_dim=DECODER_DIM,
        joiner_dim=JOINER_DIM,
        vocab_size=VOCAB_SIZE,
    )

    class AsrModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder_embed = encoder_embed
            self.encoder = encoder
            self.decoder = decoder
            self.joiner = joiner

    return AsrModel()


# ── 6. ONNX wrapper classes ────────────────────────────────────────────────────

class OnnxEncoder(nn.Module):
    def __init__(self, encoder: Zipformer2, encoder_embed: nn.Module, encoder_proj: nn.Linear):
        super().__init__()
        self.encoder = encoder
        self.encoder_embed = encoder_embed
        self.encoder_proj = encoder_proj

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, x_lens = self.encoder_embed(x, x_lens)
        max_len = x.shape[1]
        indices = torch.arange(max_len, device=x.device)
        src_key_padding_mask = indices >= x_lens.unsqueeze(1)
        x = x.permute(1, 0, 2)
        encoder_out, encoder_out_lens = self.encoder(x, x_lens, src_key_padding_mask)
        encoder_out = encoder_out.permute(1, 0, 2)
        encoder_out = self.encoder_proj(encoder_out)
        return encoder_out, encoder_out_lens


class OnnxDecoder(nn.Module):
    def __init__(self, decoder: Decoder, decoder_proj: nn.Linear):
        super().__init__()
        self.decoder = decoder
        self.decoder_proj = decoder_proj

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        decoder_output = self.decoder(y, need_pad=False)
        decoder_output = decoder_output.squeeze(1)
        return self.decoder_proj(decoder_output)


class OnnxJoiner(nn.Module):
    def __init__(self, output_linear: nn.Linear):
        super().__init__()
        self.output_linear = output_linear

    def forward(self, encoder_out: torch.Tensor, decoder_out: torch.Tensor) -> torch.Tensor:
        logit = encoder_out + decoder_out
        return self.output_linear(torch.tanh(logit))


# ── 7. ONNX metadata helper ────────────────────────────────────────────────────

def add_meta_data(filename: str, meta_data: Dict[str, str]):
    model = onnx.load(filename)
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = value
    onnx.save(model, filename)


# ── 8. Export functions ────────────────────────────────────────────────────────

def export_encoder(encoder_model: OnnxEncoder, path: str, opset: int):
    # Pre-warm PE buffers — pe bị đóng băng ở SIZE lúc convert_scaled_to_non_scaled,
    # trace không gọi extend_pe nữa. File dài nhất val 63,4s cần ~6.4k frame;
    # pre-warm 20.000 frame (~200s) để dư cho mọi audio thực tế.
    # (Bundle pre-warm 3000 cũ: crash Reshape trên file >~30s trong sherpa.)
    try:
        dummy_long = torch.zeros(20000)
        n_pw = 0
        for m in encoder_model.modules():
            # Sau convert_scaled_to_non_scaled(is_onnx=True) các module PE đã thành
            # scripted module — isinstance(…, CompactRelPositionalEncoding) KHÔNG khớp
            # nữa (bug bản trước: loop chạy 0 module mà vẫn in "pre-warmed").
            if hasattr(m, "extend_pe") and callable(getattr(m, "extend_pe")):
                m.extend_pe(dummy_long)
                n_pw += 1
        assert n_pw > 0, "không tìm thấy module PE nào để pre-warm — kiểm tra convert"
        logging.info(f"PE buffers pre-warmed to T=20000 ({n_pw} module)")
    except Exception as e:
        logging.warning(f"PE pre-warm skipped: {e}")

    x = torch.zeros(1, 100, 80, dtype=torch.float32)
    x_lens = torch.tensor([100], dtype=torch.int64)

    encoder_model = torch.jit.trace(encoder_model, (x, x_lens), check_trace=False)
    torch.onnx.export(
        encoder_model, (x, x_lens), path,
        # dynamo=False: torch>=2.9 mặc định dynamo exporter (cần onnxscript, không có
        # trong image); legacy TorchScript exporter là đường đã dùng cho mọi bundle cũ.
        dynamo=False,
        verbose=False, opset_version=opset,
        input_names=["x", "x_lens"],
        output_names=["encoder_out", "encoder_out_lens"],
        dynamic_axes={
            "x": {0: "N", 1: "T"},
            "x_lens": {0: "N"},
            "encoder_out": {0: "N", 1: "T"},
            "encoder_out_lens": {0: "N"},
        },
    )
    add_meta_data(path, {
        "model_type": "zipformer2",
        "version": "1",
        "model_author": "icefall",
        "comment": "non-streaming zipformer2 transducer encoder",
    })


def export_decoder(decoder_model: OnnxDecoder, path: str, opset: int):
    context_size = decoder_model.decoder.context_size
    vocab_size = decoder_model.decoder.vocab_size

    y = torch.zeros(10, context_size, dtype=torch.int64)
    decoder_model = torch.jit.script(decoder_model)
    torch.onnx.export(
        decoder_model, y, path,
        dynamo=False,
        verbose=False, opset_version=opset,
        input_names=["y"],
        output_names=["decoder_out"],
        dynamic_axes={"y": {0: "N"}, "decoder_out": {0: "N"}},
    )
    add_meta_data(path, {
        "context_size": str(context_size),
        "vocab_size": str(vocab_size),
    })


def export_joiner(joiner_model: OnnxJoiner, path: str, opset: int):
    joiner_dim = joiner_model.output_linear.weight.shape[1]
    enc = torch.rand(11, joiner_dim, dtype=torch.float32)
    dec = torch.rand(11, joiner_dim, dtype=torch.float32)
    torch.onnx.export(
        joiner_model, (enc, dec), path,
        dynamo=False,
        verbose=False, opset_version=opset,
        input_names=["encoder_out", "decoder_out"],
        output_names=["logit"],
        dynamic_axes={
            "encoder_out": {0: "N"},
            "decoder_out": {0: "N"},
            "logit": {0: "N"},
        },
    )
    add_meta_data(path, {"joiner_dim": str(joiner_dim)})


# ── 9. ONNX-incompatible module replacement ────────────────────────────────────

def prepare_for_onnx(model: nn.Module):
    replacements = {}
    for name, m in model.named_modules():
        if isinstance(m, (Balancer, Whiten, Dropout3, ScaleGrad)):
            replacements[name] = nn.Identity()
        elif isinstance(m, SwooshL):
            replacements[name] = SwooshLOnnx()
        elif isinstance(m, SwooshR):
            replacements[name] = SwooshROnnx()
    for path, new_m in replacements.items():
        if "." in path:
            parent_path, child = path.rsplit(".", 1)
            parent = model.get_submodule(parent_path)
        else:
            parent, child = model, path
        setattr(parent, child, new_m)


# ── 10. Main ───────────────────────────────────────────────────────────────────

def main():
    _script_dir = Path(__file__).resolve().parent.parent  # lmq/train_zipformer/

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        default=str(_script_dir / "callbacks" / "epoch-6.pt"),
        help="Checkpoint file (.pt)",
    )
    ap.add_argument(
        "--outdir",
        default="./onnx",
        help="Output directory for ONNX files",
    )
    ap.add_argument("--opset", type=int, default=13, help="ONNX opset version")
    ap.add_argument("--no-int8", action="store_true", help="Skip INT8 quantization")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ── Build & load ──
    logging.info("Building model...")
    model = build_model()

    logging.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logging.warning(f"Missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        logging.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    logging.info("Checkpoint loaded.")

    model.eval()

    # ── Convert ScaledLinear → Linear ──
    logging.info("Converting scaled modules to non-scaled...")
    convert_scaled_to_non_scaled(model, inplace=True, is_onnx=True)

    # ── Replace ONNX-incompatible ops ──
    prepare_for_onnx(model)

    # ── Wrap for export ──
    encoder = OnnxEncoder(
        encoder=model.encoder,
        encoder_embed=model.encoder_embed,
        encoder_proj=model.joiner.encoder_proj,
    )
    decoder = OnnxDecoder(
        decoder=model.decoder,
        decoder_proj=model.joiner.decoder_proj,
    )
    joiner = OnnxJoiner(output_linear=model.joiner.output_linear)

    enc_params = sum(p.numel() for p in encoder.parameters())
    dec_params = sum(p.numel() for p in decoder.parameters())
    joi_params = sum(p.numel() for p in joiner.parameters())
    logging.info(f"Encoder params: {enc_params:,}")
    logging.info(f"Decoder params: {dec_params:,}")
    logging.info(f"Joiner  params: {joi_params:,}")
    logging.info(f"Total   params: {enc_params + dec_params + joi_params:,}")

    suffix = Path(args.checkpoint).stem  # e.g. "epoch-35"
    out = Path(args.outdir)

    # ── Export FP32 ──
    enc_path = str(out / f"encoder-{suffix}.onnx")
    dec_path = str(out / f"decoder-{suffix}.onnx")
    joi_path = str(out / f"joiner-{suffix}.onnx")

    logging.info(f"Exporting encoder → {enc_path}")
    export_encoder(encoder, enc_path, args.opset)

    logging.info(f"Exporting decoder → {dec_path}")
    export_decoder(decoder, dec_path, args.opset)

    logging.info(f"Exporting joiner  → {joi_path}")
    export_joiner(joiner, joi_path, args.opset)

    logging.info("FP32 export done.")

    # ── INT8 quantization ──
    if not args.no_int8:
        # Import ở ĐÂY, không ở đầu file: onnxruntime chỉ cần cho int8, mà không
        # phải image nào cũng có nó (image worker chỉ có onnx). Import top-level
        # làm --no-int8 vô nghĩa — vẫn chết ngay lúc load module.
        from onnxruntime.quantization import QuantType, quantize_dynamic
        logging.info("Generating INT8 quantized models...")

        quantize_dynamic(
            model_input=enc_path,
            model_output=str(out / f"encoder-{suffix}.int8.onnx"),
            op_types_to_quantize=["MatMul"],
            weight_type=QuantType.QInt8,
        )
        quantize_dynamic(
            model_input=dec_path,
            model_output=str(out / f"decoder-{suffix}.int8.onnx"),
            op_types_to_quantize=["MatMul", "Gather"],
            weight_type=QuantType.QInt8,
        )
        quantize_dynamic(
            model_input=joi_path,
            model_output=str(out / f"joiner-{suffix}.int8.onnx"),
            op_types_to_quantize=["MatMul"],
            weight_type=QuantType.QInt8,
        )
        logging.info("INT8 export done.")

    logging.info(f"All ONNX files saved to: {args.outdir}")


if __name__ == "__main__":
    main()
