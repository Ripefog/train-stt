#!/usr/bin/env python3
# Copyright    2021-2023  Xiaomi Corp.        (authors: Amir Hussein
#                                                       Fangjun Kuang,
#                                                       Wei Kang,
#                                                       Mingshuang Luo,
#                                                       Zengwei Yao,
#                                                       Daniel Povey)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Usage:

export CUDA_VISIBLE_DEVICES="0,1,2,3"

# For non-streaming model training:
./zipformer/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp \
  --full-libri 1 \
  --max-duration 1000

# For streaming model training:
./zipformer/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp \
  --causal 1 \
  --full-libri 1 \
  --max-duration 1000

It supports training with:
  - transducer loss (default)
  - ctc loss
  - attention decoder loss
  - cr-ctc loss (should use half the max-duration compared to regular ctc)
"""


import argparse
import copy
import logging
import warnings
from pathlib import Path
from tqdm import tqdm
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union

import os
import wandb

if __name__ == "__main__" and "WANDB_API_KEY" in os.environ and os.environ["WANDB_API_KEY"]:
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "conformer-training"),
        name=os.environ.get("WANDB_RUN_NAME", "zipformer"),
        id=os.environ.get("WANDB_RUN_ID", None),
        resume="allow",
        sync_tensorboard=True
    )

import k2
import optim
import sentencepiece as spm
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from asr_datamodule import LibriSpeechAsrDataModule
from attention_decoder import AttentionDecoderModel
from decoder import Decoder
from joiner import Joiner
from lhotse.cut import Cut
from lhotse.dataset import SpecAugment
from lhotse.dataset.sampling.base import CutSampler
from lhotse.utils import fix_random_seed
from model import AsrModel
from optim import Eden, ScaledAdam
from scaling import ScheduledFloat
from subsampling import Conv2dSubsampling
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from zipformer import Zipformer2

from icefall import diagnostics
from icefall.checkpoint import load_checkpoint, remove_checkpoints
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.checkpoint import (
    save_checkpoint_with_global_batch_idx,
    update_averaged_model,
)
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.err import raise_grad_scale_is_too_small_error
from icefall.hooks import register_inf_check_hooks
from icefall.utils import (
    AttributeDict,
    MetricsTracker,
    create_grad_scaler,
    get_parameter_groups_with_lrs,
    setup_logger,
    str2bool,
    torch_autocast,
)

LRSchedulerType = Union[torch.optim.lr_scheduler._LRScheduler, optim.LRScheduler]


class _BandClamp:
    """Ép các mel bin trên xuống đúng mức sàn đo được từ val — mô phỏng cắt băng ~3 kHz.

    Vì fbank đã bỏ pha, low-pass trên waveform gần như tương đương với việc ép các mel
    bin trên xuống sàn. Nên đây là bản rẻ của `BAND_LIMIT_SR` trong prepare_data.py:
    không cần extract lại, và khớp trung bình CHÍNH XÁC 0 dB thay vì +6 dB như đường
    resample. Cái giá: sàn phẳng tất định, trong khi val có sàn nhiễu tự nhiên.

    VAL_FLOOR đo được (2026-08-06) trên 549 cut val / 89.537 frame có tiếng nói, đơn vị
    log tự nhiên (chia 4,3429 từ dB), là mức TRUNG BÌNH của từng bin so với quy chiếu.
    Chỉ có số cho bin 54–79 (>3 kHz) — không suy diễn ra ngoài khoảng đó.

    Quy chiếu tính THEO TỪNG FRAME (max của bin 6–23 ≈ 200–800 Hz), không theo từng
    utterance, vì hai lý do: (1) một low-pass thật làm giảm năng lượng dải cao ở mọi
    thời điểm tỉ lệ với tín hiệu tức thời, nên per-frame mới đúng vật lý; (2) batch đã
    được pad nên trung bình theo utterance sẽ bị frame padding làm lệch.

    ⚠️  VAL_FLOOR phải được đo bằng ĐÚNG quy chiếu mà `__call__` dùng. Bản đo đầu tiên
    dùng quy chiếu per-utterance (−13,247 log = −57,5 dB) và lệch **−7,5 dB** so với bản
    per-frame — dùng lẫn thì clamp nằm cao hơn val 7,5 dB và train chỉ *trông như* đã
    khớp. Đổi cách tính quy chiếu ⇒ phải đo lại vector.
    """

    N_MELS = 80
    REF_LO, REF_HI = 6, 24          # bin 6..23 ≈ 200–800 Hz
    FIRST_MEASURED = 54             # bin 54 có tâm ≈ 3000 Hz (mel kaldi, high_freq=7600)
    VAL_FLOOR = (
        -14.294, -15.460, -15.858, -16.111, -16.377, -16.515, -16.193, -15.672,
        -15.017, -14.422, -14.624, -15.697, -15.140, -15.809, -14.737, -13.857,
        -14.701, -13.691, -13.351, -13.870, -14.859, -15.217, -14.091, -14.108,
        -14.884, -15.051,
    )                               # 26 giá trị, cho bin 54..79; trung bình −14,985 log = −65,1 dB

    def __init__(self, first_bin: int):
        if first_bin < self.FIRST_MEASURED:
            raise ValueError(
                f"BAND_CLAMP_BIN={first_bin} < {self.FIRST_MEASURED}: chưa đo sàn của val "
                f"dưới bin {self.FIRST_MEASURED} (~3 kHz) nên không có số để clamp. "
                f"Muốn cắt sâu hơn thì phải đo lại trước."
            )
        if first_bin >= self.N_MELS:
            raise ValueError(f"BAND_CLAMP_BIN={first_bin} >= {self.N_MELS} bin — không clamp gì cả")
        self.n_mels = self.N_MELS
        self.first_bin = first_bin
        self._floor = torch.tensor(
            self.VAL_FLOOR[first_bin - self.FIRST_MEASURED:], dtype=torch.float32
        )

    def __call__(self, inputs: torch.Tensor, supervision_segments=None, **kwargs) -> torch.Tensor:
        # inputs: (batch, time, n_mels) log-mel, đã collate và pad
        if inputs.dim() != 3 or inputs.size(-1) != self.n_mels:
            raise RuntimeError(
                f"_BandClamp cần (batch, time, {self.n_mels}), nhận {tuple(inputs.shape)}"
            )
        floor = self._floor.to(device=inputs.device, dtype=inputs.dtype)
        ref = inputs[:, :, self.REF_LO:self.REF_HI].amax(dim=2, keepdim=True)   # (B, T, 1)
        hi = inputs[:, :, self.first_bin:]
        # in-place: tensor này do input_strategy tạo mới mỗi batch, không dùng lại
        hi.copy_(torch.minimum(hi, ref + floor))
        return inputs


def _check_band_config(clamp_bin: int, fbank_dir: str = "data/fbank") -> Optional[int]:
    """Chặn việc cắt băng HAI LẦN. Trả về BAND_LIMIT_SR của fbank (None nếu không biết).

    `BAND_LIMIT_SR` cắt trên waveform lúc extract, `BAND_CLAMP_BIN` cắt trên feature lúc
    train. Bật cả hai là cắt hai lần và không có lỗi nào tự báo — model chỉ đơn giản học
    kém hơn. Nguồn sự thật là `FEATURE_CONFIG.txt` do prepare_data.py ghi vào thư mục fbank.
    """
    cfg = Path(fbank_dir) / "FEATURE_CONFIG.txt"
    fbank_bl = None
    if cfg.is_file():
        for ln in cfg.read_text(encoding="utf-8").splitlines():
            if ln.startswith("BAND_LIMIT_SR="):
                fbank_bl = int(ln.split("=", 1)[1] or 0)
        logging.info(f"fbank FEATURE_CONFIG: BAND_LIMIT_SR={fbank_bl}")
    else:
        logging.info(
            f"fbank không có {cfg} (extract trước khi có marker) — không tự kiểm được "
            f"là fbank này full-band hay đã band-limit"
        )
    if clamp_bin > 0 and fbank_bl:
        raise ValueError(
            f"BAND_CLAMP_BIN={clamp_bin} nhưng fbank đã được extract với "
            f"BAND_LIMIT_SR={fbank_bl} (low-pass {fbank_bl / 2:.0f} Hz) — bật cả hai là cắt "
            f"băng HAI LẦN. Chọn một: clamp trên fbank full-band, hoặc dùng fbank đã "
            f"band-limit mà không clamp."
        )
    return fbank_bl


def get_adjusted_batch_count(params: AttributeDict) -> float:
    # returns the number of batches we would have used so far if we had used the reference
    # duration.  This is for purposes of set_batch_count().
    return (
        params.batch_idx_train
        * (params.max_duration * params.world_size)
        / params.ref_duration
    )


def set_batch_count(model: Union[nn.Module, DDP], batch_count: float) -> None:
    if isinstance(model, DDP):
        # get underlying nn.Module
        model = model.module
    for name, module in model.named_modules():
        if hasattr(module, "batch_count"):
            module.batch_count = batch_count
        if hasattr(module, "name"):
            module.name = name


def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--num-encoder-layers",
        type=str,
        default="2,2,3,4,3,2",
        help="Number of zipformer encoder layers per stack, comma separated.",
    )

    parser.add_argument(
        "--downsampling-factor",
        type=str,
        default="1,2,4,8,4,2",
        help="Downsampling factor for each stack of encoder layers.",
    )

    parser.add_argument(
        "--feedforward-dim",
        type=str,
        default="512,768,1024,1536,1024,768",
        help="Feedforward dimension of the zipformer encoder layers, per stack, comma separated.",
    )

    parser.add_argument(
        "--num-heads",
        type=str,
        default="4,4,4,8,4,4",
        help="Number of attention heads in the zipformer encoder layers: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--encoder-dim",
        type=str,
        default="192,256,384,512,384,256",
        help="Embedding dimension in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--query-head-dim",
        type=str,
        default="32",
        help="Query/key dimension per head in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--value-head-dim",
        type=str,
        default="12",
        help="Value dimension per head in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--pos-head-dim",
        type=str,
        default="4",
        help="Positional-encoding dimension per head in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--pos-dim",
        type=int,
        default="48",
        help="Positional-encoding embedding dimension",
    )

    parser.add_argument(
        "--encoder-unmasked-dim",
        type=str,
        default="192,192,256,256,256,192",
        help="Unmasked dimensions in the encoders, relates to augmentation during training.  "
        "A single int or comma-separated list.  Must be <= each corresponding encoder_dim.",
    )

    parser.add_argument(
        "--cnn-module-kernel",
        type=str,
        default="31,31,15,15,15,31",
        help="Sizes of convolutional kernels in convolution modules in each encoder stack: "
        "a single int or comma-separated list.",
    )

    parser.add_argument(
        "--decoder-dim",
        type=int,
        default=512,
        help="Embedding dimension in the decoder model.",
    )

    parser.add_argument(
        "--joiner-dim",
        type=int,
        default=512,
        help="""Dimension used in the joiner model.
        Outputs from the encoder and decoder model are projected
        to this dimension before adding.
        """,
    )

    parser.add_argument(
        "--attention-decoder-dim",
        type=int,
        default=512,
        help="""Dimension used in the attention decoder""",
    )

    parser.add_argument(
        "--attention-decoder-num-layers",
        type=int,
        default=6,
        help="""Number of transformer layers used in attention decoder""",
    )

    parser.add_argument(
        "--attention-decoder-attention-dim",
        type=int,
        default=512,
        help="""Attention dimension used in attention decoder""",
    )

    parser.add_argument(
        "--attention-decoder-num-heads",
        type=int,
        default=8,
        help="""Number of attention heads used in attention decoder""",
    )

    parser.add_argument(
        "--attention-decoder-feedforward-dim",
        type=int,
        default=2048,
        help="""Feedforward dimension used in attention decoder""",
    )

    parser.add_argument(
        "--qat",
        type=str2bool,
        default=False,
        help="""Quantization-aware training: convert_scaled_to_non_scaled rồi
        prepare_qat (FakeQuantize per-channel int8 cho Linear/Conv, moving-average
        min-max cho activation). Chạy kèm fp32 (USE_FP16=0).""",
    )

    parser.add_argument(
        "--qat-calib-epochs",
        type=int,
        default=1,
        help="""Số epoch ĐẦU (tính theo epoch TUYỆT ĐỐI) chỉ chạy observer,
        fake-quant tắt — cho scale/zero-point hội tụ trước khi weight phải làm quen lượng tử.""",
    )

    parser.add_argument(
        "--qat-bits",
        type=int,
        default=8,
        choices=[4, 8],
        help="""8: QAT int8 đầy đủ (weight per-channel + activation moving-avg, qconfig
        x86 chuẩn). 4: QAT weight-only int4 — MÔ PHỎNG MatMulNBits (block 128 symmetric)
        bằng FakeQuantize per-channel symmetric quant_min=-8/quant_max=7, KHÔNG
        fake-quant activation. Lưu ý sai khác: per-channel mịn hơn per-block-128 của
        bản deploy — QAT-int4 chịu quant nhẹ hơn thực tế, đọc kết quả với điều đó.""",
    )

    parser.add_argument(
        "--causal",
        type=str2bool,
        default=False,
        help="If True, use causal version of model.",
    )

    parser.add_argument(
        "--chunk-size",
        type=str,
        default="16,32,64,-1",
        help="Chunk sizes (at 50Hz frame rate) will be chosen randomly from this list during training. "
        " Must be just -1 if --causal=False",
    )

    parser.add_argument(
        "--left-context-frames",
        type=str,
        default="64,128,256,-1",
        help="Maximum left-contexts for causal training, measured in frames which will "
        "be converted to a number of chunks.  If splitting into chunks, "
        "chunk left-context frames will be chosen randomly from this list; else not relevant.",
    )

    parser.add_argument(
        "--use-transducer",
        type=str2bool,
        default=True,
        help="If True, use Transducer head.",
    )

    parser.add_argument(
        "--use-ctc",
        type=str2bool,
        default=False,
        help="If True, use CTC head.",
    )

    parser.add_argument(
        "--use-attention-decoder",
        type=str2bool,
        default=False,
        help="If True, use attention-decoder head.",
    )

    parser.add_argument(
        "--use-cr-ctc",
        type=str2bool,
        default=False,
        help="If True, use consistency-regularized CTC.",
    )


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP training.",
    )

    parser.add_argument(
        "--master-port",
        type=int,
        default=12354,
        help="Master port to use for DDP training.",
    )

    parser.add_argument(
        "--tensorboard",
        type=str2bool,
        default=True,
        help="Should various information be logged in tensorboard.",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=30,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="""Resume training from this epoch. It should be positive.
        If larger than 1, it will load checkpoint from
        exp-dir/epoch-{start_epoch-1}.pt
        """,
    )

    parser.add_argument(
        "--start-batch",
        type=int,
        default=0,
        help="""If positive, --start-epoch is ignored and
        it loads the checkpoint from exp-dir/checkpoint-{start_batch}.pt
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="zipformer/exp",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--base-lr", type=float, default=0.045, help="The base learning rate."
    )

    parser.add_argument(
        "--lr-batches",
        type=float,
        default=7500,
        help="""Number of steps that affects how rapidly the learning rate
        decreases. We suggest not to change this.""",
    )

    parser.add_argument(
        "--lr-epochs",
        type=float,
        default=3.5,
        help="""Number of epochs that affects how rapidly the learning rate decreases.
        """,
    )

    parser.add_argument(
        "--ref-duration",
        type=float,
        default=600,
        help="Reference batch duration for purposes of adjusting batch counts for setting various "
        "schedules inside the model",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
    )

    parser.add_argument(
        "--prune-range",
        type=int,
        default=5,
        help="The prune range for rnnt loss, it means how many symbols(context)"
        "we are using to compute the loss",
    )

    parser.add_argument(
        "--lm-scale",
        type=float,
        default=0.25,
        help="The scale to smooth the loss with lm "
        "(output of prediction network) part.",
    )

    parser.add_argument(
        "--am-scale",
        type=float,
        default=0.0,
        help="The scale to smooth the loss with am (output of encoder network) part.",
    )

    parser.add_argument(
        "--simple-loss-scale",
        type=float,
        default=0.5,
        help="To get pruning ranges, we will calculate a simple version"
        "loss(joiner is just addition), this simple loss also uses for"
        "training (as a regularization item). We will scale the simple loss"
        "with this parameter before adding to the final loss.",
    )

    parser.add_argument(
        "--ctc-loss-scale",
        type=float,
        default=0.2,
        help="Scale for CTC loss.",
    )

    parser.add_argument(
        "--cr-loss-scale",
        type=float,
        default=0.2,
        help="Scale for consistency-regularization loss.",
    )

    parser.add_argument(
        "--time-mask-ratio",
        type=float,
        default=2.5,
        help="When using cr-ctc, we increase the amount of time-masking in SpecAugment.",
    )

    parser.add_argument(
        "--attention-decoder-loss-scale",
        type=float,
        default=0.8,
        help="Scale for attention-decoder loss.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )

    parser.add_argument(
        "--print-diagnostics",
        type=str2bool,
        default=False,
        help="Accumulate stats on activations, print them and exit.",
    )

    parser.add_argument(
        "--inf-check",
        type=str2bool,
        default=False,
        help="Add hooks to check for infinite module outputs and gradients.",
    )

    parser.add_argument(
        "--save-every-n",
        type=int,
        default=4000,
        help="""Save checkpoint after processing this number of batches"
        periodically. We save checkpoint to exp-dir/ whenever
        params.batch_idx_train % save_every_n == 0. The checkpoint filename
        has the form: f'exp-dir/checkpoint-{params.batch_idx_train}.pt'
        Note: It also saves checkpoint to `exp-dir/epoch-xxx.pt` at the
        end of each epoch where `xxx` is the epoch number counting from 1.
        """,
    )

    parser.add_argument(
        "--keep-last-k",
        type=int,
        default=30,
        help="""Only keep this number of checkpoints on disk.
        For instance, if it is 3, there are only 3 checkpoints
        in the exp-dir with filenames `checkpoint-xxx.pt`.
        It does not affect checkpoints with name `epoch-xxx.pt`.
        """,
    )

    parser.add_argument(
        "--average-period",
        type=int,
        default=200,
        help="""Update the averaged model, namely `model_avg`, after processing
        this number of batches. `model_avg` is a separate version of model,
        in which each floating-point parameter is the average of all the
        parameters from the start of training. Each time we take the average,
        we do: `model_avg = model * (average_period / batch_idx_train) +
            model_avg * ((batch_idx_train - average_period) / batch_idx_train)`.
        """,
    )

    parser.add_argument(
        "--use-fp16",
        type=str2bool,
        default=False,
        help="Whether to use half precision training.",
    )

    parser.add_argument(
        "--use-bf16",
        type=str2bool,
        default=False,
        help="Whether to use bf16 in AMP.",
    )

    parser.add_argument(
        "--freeze-encoder",
        type=str2bool,
        default=False,
        help="""If True, freeze encoder_embed and encoder weights.
        Equivalent to --trainable-modules ctc_output,joiner,decoder,simple_am_proj,simple_lm_proj,attention_decoder""",
    )

    parser.add_argument(
        "--trainable-modules",
        type=str,
        default="",
        help="""Comma-separated list of module name prefixes that should remain trainable.
        Everything else is frozen. Overrides --freeze-encoder when set.
        Examples:
          'ctc_output'            — freeze all, train full CTC head
          'ctc_output.1'         — freeze all, train only final Linear in CTC head
          'joiner'               — freeze all, train only joiner
        Leave empty to disable (train all params).
        Tiền tố 're:' bật chế độ regex, khớp bằng re.search trên TÊN ĐẦY ĐỦ — cần cho
        BitFit vì tên nằm ở đuôi: 're:(bypass_scale|norm\\.bias|norm\\.log_scale)$'.
        Khớp 0 param là LỖI, script sẽ raise chứ không chạy tiếp.""",
    )

    parser.add_argument(
        "--emb-term-mask",
        type=str,
        default="",
        help="""Đường dẫn word_mapping.json. Khi đặt, CHỈ các dòng của
        decoder.embedding.weight thuộc token BPE của thuật ngữ được cập nhật; mọi dòng
        khác bị ghi lại giá trị gốc sau mỗi optimizer.step().

        ⚠️ Đo 2026-08-13 cho thấy hướng này khó có ích: 87 cụm ra 120/2000 dòng, nhưng
        đó là âm tiết Việt THƯỜNG (▁hai, ▁ba, ▁bốn, ▁năm, ▁em, ▁qua, i, t) vì thuật ngữ
        là phiên âm. Trên 30.000 cut của 200h, các token này xuất hiện 72,4% số lần ở
        ngữ cảnh thường. Tức không tách được gradient thuật ngữ khỏi gradient từ thường.
        Giữ cờ này để chạy được thí nghiệm, không phải vì kỳ vọng nó thắng.""",
    )

    add_model_arguments(parser)

    return parser


def get_params() -> AttributeDict:
    """Return a dict containing training parameters.

    All training related parameters that are not passed from the commandline
    are saved in the variable `params`.

    Commandline options are merged into `params` after they are parsed, so
    you can also access them via `params`.

    Explanation of options saved in `params`:

        - best_train_loss: Best training loss so far. It is used to select
                           the model that has the lowest training loss. It is
                           updated during the training.

        - best_valid_loss: Best validation loss so far. It is used to select
                           the model that has the lowest validation loss. It is
                           updated during the training.

        - best_train_epoch: It is the epoch that has the best training loss.

        - best_valid_epoch: It is the epoch that has the best validation loss.

        - batch_idx_train: Used to writing statistics to tensorboard. It
                           contains number of batches trained so far across
                           epochs.

        - log_interval:  Print training loss if batch_idx % log_interval` is 0

        - reset_interval: Reset statistics if batch_idx % reset_interval is 0

        - valid_interval:  Run validation if batch_idx % valid_interval is 0

        - feature_dim: The model input dim. It has to match the one used
                       in computing features.

        - subsampling_factor:  The subsampling factor for the model.

        - warm_step: The warmup period that dictates the decay of the
              scale on "simple" (un-pruned) loss.
    """
    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            # In loss mỗi N batch. Với 2.340h thì 1 epoch ~14.450 batch, ở 50 batch
            # log ra ~289 dòng/epoch nên rất khó đọc. Đổi qua LOG_INTERVAL nếu cần.
            "log_interval": int(os.environ.get("LOG_INTERVAL", "200") or 200),
            "reset_interval": 200,
            "valid_interval": 3000,  # For the 100h subset, use 800
            # parameters for zipformer
            "feature_dim": 80,
            "subsampling_factor": 4,  # not passed in, this is fixed.
            # parameters for attention-decoder
            "ignore_id": -1,
            "label_smoothing": 0.1,
            "warm_step": 2000,
            "env_info": get_env_info(),
        }
    )

    return params


def _to_int_tuple(s: str):
    return tuple(map(int, s.split(",")))


def get_encoder_embed(params: AttributeDict) -> nn.Module:
    # encoder_embed converts the input of shape (N, T, num_features)
    # to the shape (N, (T - 7) // 2, encoder_dims).
    # That is, it does two things simultaneously:
    #   (1) subsampling: T -> (T - 7) // 2
    #   (2) embedding: num_features -> encoder_dims
    # In the normal configuration, we will downsample once more at the end
    # by a factor of 2, and most of the encoder stacks will run at a lower
    # sampling rate.
    encoder_embed = Conv2dSubsampling(
        in_channels=params.feature_dim,
        out_channels=_to_int_tuple(params.encoder_dim)[0],
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
    )
    return encoder_embed


def get_encoder_model(params: AttributeDict) -> nn.Module:
    encoder = Zipformer2(
        output_downsampling_factor=2,
        downsampling_factor=_to_int_tuple(params.downsampling_factor),
        num_encoder_layers=_to_int_tuple(params.num_encoder_layers),
        encoder_dim=_to_int_tuple(params.encoder_dim),
        encoder_unmasked_dim=_to_int_tuple(params.encoder_unmasked_dim),
        query_head_dim=_to_int_tuple(params.query_head_dim),
        pos_head_dim=_to_int_tuple(params.pos_head_dim),
        value_head_dim=_to_int_tuple(params.value_head_dim),
        pos_dim=params.pos_dim,
        num_heads=_to_int_tuple(params.num_heads),
        feedforward_dim=_to_int_tuple(params.feedforward_dim),
        cnn_module_kernel=_to_int_tuple(params.cnn_module_kernel),
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
        warmup_batches=4000.0,
        causal=params.causal,
        chunk_size=_to_int_tuple(params.chunk_size),
        left_context_frames=_to_int_tuple(params.left_context_frames),
    )
    return encoder


def get_decoder_model(params: AttributeDict) -> nn.Module:
    decoder = Decoder(
        vocab_size=params.vocab_size,
        decoder_dim=params.decoder_dim,
        blank_id=params.blank_id,
        context_size=params.context_size,
    )
    return decoder


def get_joiner_model(params: AttributeDict) -> nn.Module:
    joiner = Joiner(
        encoder_dim=max(_to_int_tuple(params.encoder_dim)),
        decoder_dim=params.decoder_dim,
        joiner_dim=params.joiner_dim,
        vocab_size=params.vocab_size,
    )
    return joiner


def get_attention_decoder_model(params: AttributeDict) -> nn.Module:
    decoder = AttentionDecoderModel(
        vocab_size=params.vocab_size,
        decoder_dim=params.attention_decoder_dim,
        num_decoder_layers=params.attention_decoder_num_layers,
        attention_dim=params.attention_decoder_attention_dim,
        num_heads=params.attention_decoder_num_heads,
        feedforward_dim=params.attention_decoder_feedforward_dim,
        memory_dim=max(_to_int_tuple(params.encoder_dim)),
        sos_id=params.sos_id,
        eos_id=params.eos_id,
        ignore_id=params.ignore_id,
        label_smoothing=params.label_smoothing,
    )
    return decoder


def get_model(params: AttributeDict) -> nn.Module:
    assert params.use_transducer or params.use_ctc, (
        f"At least one of them should be True, "
        f"but got params.use_transducer={params.use_transducer}, "
        f"params.use_ctc={params.use_ctc}"
    )

    encoder_embed = get_encoder_embed(params)
    encoder = get_encoder_model(params)

    if params.use_transducer:
        decoder = get_decoder_model(params)
        joiner = get_joiner_model(params)
    else:
        decoder = None
        joiner = None

    if params.use_attention_decoder:
        attention_decoder = get_attention_decoder_model(params)
    else:
        attention_decoder = None

    model = AsrModel(
        encoder_embed=encoder_embed,
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        attention_decoder=attention_decoder,
        encoder_dim=max(_to_int_tuple(params.encoder_dim)),
        decoder_dim=params.decoder_dim,
        vocab_size=params.vocab_size,
        use_transducer=params.use_transducer,
        use_ctc=params.use_ctc,
        use_attention_decoder=params.use_attention_decoder,
    )
    return model


def get_spec_augment(params: AttributeDict) -> SpecAugment:
    num_frame_masks = int(10 * params.time_mask_ratio)
    max_frames_mask_fraction = 0.15 * params.time_mask_ratio
    logging.info(
        f"num_frame_masks: {num_frame_masks}, "
        f"max_frames_mask_fraction: {max_frames_mask_fraction}"
    )
    spec_augment = SpecAugment(
        time_warp_factor=0,  # Do time warping in model.py
        num_frame_masks=num_frame_masks,  # default: 10
        features_mask_size=27,
        num_feature_masks=2,
        frames_mask_size=100,
        max_frames_mask_fraction=max_frames_mask_fraction,  # default: 0.15
    )
    return spec_augment


def load_checkpoint_if_available(
    params: AttributeDict,
    model: nn.Module,
    model_avg: nn.Module = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
) -> Optional[Dict[str, Any]]:
    """Load checkpoint from file.

    If params.start_batch is positive, it will load the checkpoint from
    `params.exp_dir/checkpoint-{params.start_batch}.pt`. Otherwise, if
    params.start_epoch is larger than 1, it will load the checkpoint from
    `params.start_epoch - 1`.

    Apart from loading state dict for `model` and `optimizer` it also updates
    `best_train_epoch`, `best_train_loss`, `best_valid_epoch`,
    and `best_valid_loss` in `params`.

    Args:
      params:
        The return value of :func:`get_params`.
      model:
        The training model.
      model_avg:
        The stored model averaged from the start of training.
      optimizer:
        The optimizer that we are using.
      scheduler:
        The scheduler that we are using.
    Returns:
      Return a dict containing previously saved training info.
    """
    if params.start_batch > 0:
        filename = params.exp_dir / f"checkpoint-{params.start_batch}.pt"
    elif params.start_epoch > 1:
        filename = params.exp_dir / f"epoch-{params.start_epoch-1}.pt"
    else:
        return None

    assert filename.is_file(), f"{filename} does not exist!"

    saved_params = load_checkpoint(
        filename,
        model=model,
        model_avg=model_avg,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    keys = [
        "best_train_epoch",
        "best_valid_epoch",
        "batch_idx_train",
        "best_train_loss",
        "best_valid_loss",
    ]
    for k in keys:
        # dùng default có sẵn trong params nếu checkpoint không có key này
        # (ví dụ checkpoint đã average — chỉ có 'model', không có state training đầy đủ)
        params[k] = saved_params.get(k, params[k])

    if params.start_batch > 0:
        if "cur_epoch" in saved_params:
            params["start_epoch"] = saved_params["cur_epoch"]

    return saved_params


def save_checkpoint(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    model_avg: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
    sampler: Optional[CutSampler] = None,
    scaler: Optional["GradScaler"] = None,
    rank: int = 0,
) -> None:
    """Save model, optimizer, scheduler and training stats to file.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The training model.
      model_avg:
        The stored model averaged from the start of training.
      optimizer:
        The optimizer used in the training.
      sampler:
       The sampler for the training dataset.
      scaler:
        The scaler used for mix precision training.
    """
    if rank != 0:
        return
    filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
    save_checkpoint_impl(
        filename=filename,
        model=model,
        model_avg=model_avg,
        params=params,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        scaler=scaler,
        rank=rank,
    )

    if params.best_train_epoch == params.cur_epoch:
        best_train_filename = params.exp_dir / "best-train-loss.pt"
        copyfile(src=filename, dst=best_train_filename)

    if params.best_valid_epoch == params.cur_epoch:
        best_valid_filename = params.exp_dir / "best-valid-loss.pt"
        copyfile(src=filename, dst=best_valid_filename)
        
    # Thêm logic xoá các checkpoint epoch cũ, chỉ giữ lại KEEP_LAST_K
    if hasattr(params, "keep_last_k") and params.keep_last_k > 0:
        import glob
        import re
        epoch_files = glob.glob(str(params.exp_dir / "epoch-*.pt"))
        epoch_files = [f for f in epoch_files if re.search(r'epoch-\d+\.pt$', f)]
        
        def epoch_num(f):
            m = re.search(r'epoch-(\d+)\.pt$', f)
            return int(m.group(1)) if m else -1
            
        epoch_files.sort(key=epoch_num)
        
        if len(epoch_files) > params.keep_last_k:
            for f in epoch_files[:-params.keep_last_k]:
                os.remove(f)
                logging.info(f"Removed old epoch checkpoint: {f}")


def compute_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    sp: spm.SentencePieceProcessor,
    batch: dict,
    is_training: bool,
    spec_augment: Optional[SpecAugment] = None,
) -> Tuple[Tensor, MetricsTracker]:
    """
    Compute loss given the model and its inputs.

    Args:
      params:
        Parameters for training. See :func:`get_params`.
      model:
        The model for training. It is an instance of Zipformer in our case.
      batch:
        A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
        for the content in it.
      is_training:
        True for training. False for validation. When it is True, this
        function enables autograd during computation; when it is False, it
        disables autograd.
      spec_augment:
        The SpecAugment instance used only when use_cr_ctc is True.
    """
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    feature = batch["inputs"]
    # at entry, feature is (N, T, C)
    assert feature.ndim == 3
    feature = feature.to(device)

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)

    batch_idx_train = params.batch_idx_train
    warm_step = params.warm_step

    texts = batch["supervisions"]["text"]
    y = sp.encode(texts, out_type=int)
    y = k2.RaggedTensor(y)

    use_cr_ctc = params.use_cr_ctc
    use_spec_aug = use_cr_ctc and is_training
    if use_spec_aug:
        supervision_intervals = batch["supervisions"]
        supervision_segments = torch.stack(
            [
                supervision_intervals["sequence_idx"],
                supervision_intervals["start_frame"],
                supervision_intervals["num_frames"],
            ],
            dim=1,
        )  # shape: (S, 3)
    else:
        supervision_segments = None

    with torch.set_grad_enabled(is_training):
        simple_loss, pruned_loss, ctc_loss, attention_decoder_loss, cr_loss = model(
            x=feature,
            x_lens=feature_lens,
            y=y,
            prune_range=params.prune_range,
            am_scale=params.am_scale,
            lm_scale=params.lm_scale,
            use_cr_ctc=use_cr_ctc,
            use_spec_aug=use_spec_aug,
            spec_augment=spec_augment,
            supervision_segments=supervision_segments,
            time_warp_factor=params.spec_aug_time_warp_factor,
        )

        loss = 0.0

        if params.use_transducer:
            s = params.simple_loss_scale
            # take down the scale on the simple loss from 1.0 at the start
            # to params.simple_loss scale by warm_step.
            simple_loss_scale = (
                s
                if batch_idx_train >= warm_step
                else 1.0 - (batch_idx_train / warm_step) * (1.0 - s)
            )
            pruned_loss_scale = (
                1.0
                if batch_idx_train >= warm_step
                else 0.1 + 0.9 * (batch_idx_train / warm_step)
            )
            loss += simple_loss_scale * simple_loss + pruned_loss_scale * pruned_loss

        if params.use_ctc:
            loss += params.ctc_loss_scale * ctc_loss
            if use_cr_ctc:
                # linear warmup
                cr_loss_scale = min(batch_idx_train / warm_step, 1.0) * params.cr_loss_scale
                loss += cr_loss_scale * cr_loss

        if params.use_attention_decoder:
            loss += params.attention_decoder_loss_scale * attention_decoder_loss

    assert loss.requires_grad == is_training

    info = MetricsTracker()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info["frames"] = (feature_lens // params.subsampling_factor).sum().item()

    # Note: We use reduction=sum while computing the loss.
    info["loss"] = loss.detach().cpu().item()
    if params.use_transducer:
        info["simple_loss"] = simple_loss.detach().cpu().item()
        info["pruned_loss"] = pruned_loss.detach().cpu().item()
    if params.use_ctc:
        info["ctc_loss"] = ctc_loss.detach().cpu().item()
        if params.use_cr_ctc:
            info["cr_loss"] = cr_loss.detach().cpu().item()
    if params.use_attention_decoder:
        info["attn_decoder_loss"] = attention_decoder_loss.detach().cpu().item()

    return loss, info


def wer_decode_modes() -> List[str]:
    """WER_DECODE=ctc|transducer|both — chọn path nào dùng để tính WER.

    QUAN TRỌNG: export_onnx.py chỉ export encoder/decoder/joiner (path transducer),
    KHÔNG export ctc_output — weight CTC bị drop qua load_state_dict(strict=False).
    Nên 'transducer' mới là con số phản ánh chất lượng model thực sự deploy;
    'ctc' chỉ đo CTC head (argmax song song nên nhanh hơn nhiều).
    """
    raw = os.environ.get("WER_DECODE", "ctc").strip().lower()
    if raw == "both":
        return ["ctc", "transducer"]
    if raw in ("ctc", "transducer"):
        return [raw]
    logging.warning(
        f"WER_DECODE={raw!r} không hợp lệ (ctc|transducer|both) — dùng 'ctc'"
    )
    return ["ctc"]


def compute_validation_wer(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    sp: spm.SentencePieceProcessor,
    valid_dl: torch.utils.data.DataLoader,
    max_batches: Optional[int] = 10,
    modes: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute WER bằng CTC greedy và/hoặc transducer greedy.

    Path nào được đo là do WER_DECODE quyết định (xem wer_decode_modes()).
    Encoder chỉ forward MỘT lần mỗi batch, các mode dùng chung output đó — nên
    bật 'both' KHÔNG làm gấp đôi chi phí encoder (chỉ thêm phần decode).

    Returns dict {mode: micro-WER (0..1)}; dict rỗng nếu không đo được gì.

    MICRO, không phải macro: gộp S+D+I và số từ ref của TOÀN corpus rồi chia một lần
    — đúng định nghĩa WER (NIST/sclite/Kaldi). Trước 2026-08-10 hàm này trả trung bình
    WER từng batch (macro), sai ở hai điểm: (1) phụ thuộc cách gom batch nên đổi
    MAX_DURATION là đổi số dù model không đổi một bit; (2) batch câu ngắn được cân
    ngang batch câu dài, mà câu 1-3 từ có WER ~98%. Chênh đo được 0,66-0,71 điểm.
    Macro vẫn được log một dòng riêng để nối với số của các run trước.
    """
    if modes is None:
        modes = wer_decode_modes()

    # chỉ giữ mode mà model thực sự có head tương ứng
    modes = [
        mo
        for mo in modes
        if (mo == "ctc" and params.use_ctc)
        or (mo == "transducer" and params.use_transducer)
    ]
    if not modes:
        logging.info(
            "compute_validation_wer: skipped — không có head nào khớp WER_DECODE "
            f"(use_ctc={params.use_ctc}, use_transducer={params.use_transducer})"
        )
        return {}

    try:
        import jiwer
    except ImportError:
        logging.warning(
            "compute_validation_wer: skipped — jiwer not installed. Run: pip install jiwer"
        )
        return {}

    greedy_search_batch = None
    if "transducer" in modes:
        # dùng greedy_search_batch từ beam_search.py (giống hệt decode.py)
        try:
            from beam_search import greedy_search_batch
        except ImportError:
            logging.warning(
                "compute_validation_wer: bỏ mode 'transducer' — không thấy beam_search.py"
            )
            modes = [mo for mo in modes if mo != "transducer"]
            if not modes:
                return {}

    logging.info(f"compute_validation_wer: decode bằng {modes}")

    # Evaluation mode
    model.eval()

    # Model device
    m = model.module if isinstance(model, DDP) else model
    device = next(m.parameters()).device

    # micro: cộng dồn lỗi và số từ ref toàn corpus, chia MỘT lần ở cuối
    tot_err = {mo: 0 for mo in modes}
    tot_ref = {mo: 0 for mo in modes}
    # macro (cũ): trung bình WER từng batch — chỉ giữ để nối với log các run trước
    total_wer = {mo: 0.0 for mo in modes}
    n_steps = 0
    n_samples = 0

    # tqdm iterator
    eval_iterator = tqdm(valid_dl, total=max_batches, desc="WER", leave=False)

    with torch.no_grad():
        for step, batch in enumerate(eval_iterator):
            if max_batches is not None and step >= max_batches:
                break

            feature = batch["inputs"].to(device)
            feature_lens = batch["supervisions"]["num_frames"].to(device)
            texts = batch["supervisions"]["text"]

            # Lọc các utterance quá ngắn — encoder zipformer subsample ~4x,
            # nếu input < 20 frames thì output có thể = 0 frames → CUDA assert
            min_frames = 20
            valid_mask = (feature_lens >= min_frames).cpu()
            if not valid_mask.all():
                keep = valid_mask.tolist()
                feature = feature[valid_mask.to(device)]
                feature_lens = feature_lens[valid_mask.to(device)]
                texts = [t for t, v in zip(texts, keep) if v]
            if len(texts) == 0:
                continue

            # Encoder forward MỘT lần — mọi mode dùng chung output này
            encoder_out, encoder_out_lens = m.forward_encoder(feature, feature_lens)

            outputs_true = list(texts)
            n_steps += 1
            n_samples += len(outputs_true)

            desc = []
            for mo in modes:
                if mo == "ctc":
                    # ── CTC greedy: argmax → collapse repeated + remove blank ──
                    logits = m.ctc_output(encoder_out)                # (N, T, vocab)
                    tokens_batch = logits.argmax(dim=-1)              # (N, T)

                    outputs_pred = []
                    for toks, length in zip(tokens_batch, encoder_out_lens):
                        toks = toks[:length].tolist()
                        prev = -1
                        filtered = []
                        for t in toks:
                            if t != 0 and t != prev:
                                filtered.append(t)
                            prev = t
                        outputs_pred.append(sp.decode(filtered))
                else:
                    # ── Transducer greedy: giống hệt decode.py ──
                    hyp_tokens = greedy_search_batch(
                        model=m,
                        encoder_out=encoder_out,
                        encoder_out_lens=encoder_out_lens,
                    )
                    outputs_pred = sp.decode(hyp_tokens)

                # process_words cho cả S/D/I/hits — micro cần đếm thô, không cần WER
                out = jiwer.process_words(
                    outputs_true,
                    outputs_pred,
                    reference_transform=jiwer.wer_standardize,
                    hypothesis_transform=jiwer.wer_standardize,
                )
                tot_err[mo] += out.substitutions + out.deletions + out.insertions
                tot_ref[mo] += out.substitutions + out.deletions + out.hits
                total_wer[mo] += out.wer          # macro, chỉ để nối lịch sử
                desc.append(f"{mo} {100 * tot_err[mo] / max(tot_ref[mo], 1):.2f}%")

            eval_iterator.set_description(" | ".join(desc))

    if n_samples == 0:
        logging.warning("compute_validation_wer: skipped — no samples were processed")
        return {}
    micro = {mo: tot_err[mo] / max(tot_ref[mo], 1) for mo in modes}
    # Một dòng để so được với log các run TRƯỚC 2026-08-10 (khi ấy dòng "WER [mo]" là macro)
    logging.info(
        "compute_validation_wer: "
        + " | ".join(
            f"{mo} micro {100 * micro[mo]:.2f}% (macro cũ {100 * total_wer[mo] / n_steps:.2f}%, "
            f"{tot_err[mo]}/{tot_ref[mo]} từ)"
            for mo in modes
        )
    )
    return micro


def log_wer(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    sp: spm.SentencePieceProcessor,
    dl: torch.utils.data.DataLoader,
    tag: str,
    tb_writer=None,
    max_batches: Optional[int] = None,
    prefix: str = "",
) -> Dict[str, float]:
    """Tính WER rồi log + ghi tensorboard. tag: 'TRAIN' hoặc 'VALID'."""
    wers = compute_validation_wer(
        params=params, model=model, sp=sp, valid_dl=dl, max_batches=max_batches
    )
    if not wers:
        logging.warning(f"{prefix}{tag} WER: không tính được")
        return wers
    for mo, w in wers.items():
        logging.info(f"{prefix}{tag} WER [{mo}]: {w * 100:.2f}%")
        if tb_writer is not None:
            tb_writer.add_scalar(
                f"train/{tag.lower()}_wer_{mo}", w, params.batch_idx_train
            )
    return wers


def compute_validation_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    sp: spm.SentencePieceProcessor,
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
) -> MetricsTracker:
    """Run the validation process."""
    model.eval()

    tot_loss = MetricsTracker()

    for batch_idx, batch in enumerate(valid_dl):
        loss, loss_info = compute_loss(
            params=params,
            model=model,
            sp=sp,
            batch=batch,
            is_training=False,
        )
        assert loss.requires_grad is False
        tot_loss = tot_loss + loss_info

    if world_size > 1:
        tot_loss.reduce(loss.device)

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value

    return tot_loss


# Hướng 5. Đặt trong run(), gọi trong train_one_epoch sau mỗi optimizer.step().
# None = tắt (mặc định).
_emb_lock = None


def make_emb_term_lock(model, sp, wm_path: str):
    """Trả về closure ghi lại các dòng embedding KHÔNG thuộc thuật ngữ.

    Vì sao ghi lại chứ không mask gradient: xem chú thích tại chỗ gọi trong
    train_one_epoch — ScaledAdam dịch cả tensor bằng một scalar scale nên dòng có
    grad = 0 vẫn bị dịch.

    Thuật ngữ = các value NHIỀU ÂM TIẾT của word_mapping.json, cùng định nghĩa với
    scripts/metrics.py để con số so được với các bảng trong results.md.
    """
    import json
    import re
    import unicodedata

    # ⚠️ Bảng thật nằm dưới khoá ["words"], không phải cấp ngoài cùng — lặp sai cấp
    # thì ra 6 "cụm" (là các value phụ có dấu cách) thay vì 87.
    with open(wm_path, encoding="utf-8") as f:
        wm = json.load(f)["words"]

    terms = set()
    for k, v in wm.items():
        if k.startswith("_") or " " not in v:   # bỏ value 1 âm tiết: trùng từ thường
            continue
        terms.add(re.sub(r"\s+", " ", unicodedata.normalize("NFC", v).lower()).strip())

    keep_ids = set()
    for t in sorted(terms):
        keep_ids.update(sp.encode(t))

    emb = model.decoder.embedding.weight
    vocab = emb.shape[0]
    keep_ids = {i for i in keep_ids if 0 <= i < vocab}

    mask_frozen = torch.ones(vocab, dtype=torch.bool, device=emb.device)
    mask_frozen[sorted(keep_ids)] = False           # True = đóng băng dòng này
    orig = emb.detach()[mask_frozen].clone()        # giá trị gốc của phần đóng băng

    logging.info(
        f"emb-term-mask: {len(terms)} cụm thuật ngữ → {len(keep_ids)}/{vocab} dòng mở "
        f"({100*len(keep_ids)/vocab:.1f}%, {len(keep_ids)*emb.shape[1]:,} param); "
        f"{int(mask_frozen.sum())} dòng đóng băng"
    )
    logging.info(
        "  token mở: "
        + " ".join(sp.id_to_piece(i) for i in sorted(keep_ids)[:30])
        + (" ..." if len(keep_ids) > 30 else "")
    )

    def _lock():
        with torch.no_grad():
            emb[mask_frozen] = orig.to(emb.dtype)

    return _lock


def train_one_epoch(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    optimizer: torch.optim.Optimizer,
    scheduler: LRSchedulerType,
    sp: spm.SentencePieceProcessor,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    scaler: "GradScaler",
    spec_augment: Optional[SpecAugment] = None,
    model_avg: Optional[nn.Module] = None,
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
    rank: int = 0,
) -> None:
    """Train the model for one epoch.

    The training loss from the mean of all frames is saved in
    `params.train_loss`. It runs the validation process every
    `params.valid_interval` batches.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The model for training.
      optimizer:
        The optimizer we are using.
      scheduler:
        The learning rate scheduler, we call step() every step.
      train_dl:
        Dataloader for the training dataset.
      valid_dl:
        Dataloader for the validation dataset.
      scaler:
        The scaler used for mix precision training.
      spec_augment:
        The SpecAugment instance used only when use_cr_ctc is True.
      model_avg:
        The stored model averaged from the start of training.
      tb_writer:
        Writer to write log messages to tensorboard.
      world_size:
        Number of nodes in DDP training. If it is 1, DDP is disabled.
      rank:
        The rank of the node in DDP training. If no DDP is used, it should
        be set to 0.
    """
    model.train()

    tot_loss = MetricsTracker()

    saved_bad_model = False

    def save_bad_model(suffix: str = ""):
        save_checkpoint_impl(
            filename=params.exp_dir / f"bad-model{suffix}-{rank}.pt",
            model=model,
            model_avg=model_avg,
            params=params,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=0,
        )

    # Dùng số batch đếm được từ epoch trước (nếu có) để hiện ETA ngay từ đầu
    total_batches = getattr(params, "num_batches_per_epoch", None)
    if total_batches is None:
        try:
            total_batches = len(train_dl)
        except TypeError:
            total_batches = None
    train_iter = tqdm(
        train_dl,
        desc=f"Epoch {params.cur_epoch}",
        total=total_batches,
        dynamic_ncols=True,
        leave=True,
    )
    for batch_idx, batch in enumerate(train_iter):
        if batch_idx % 10 == 0:
            set_batch_count(model, get_adjusted_batch_count(params))

        params.batch_idx_train += 1
        batch_size = len(batch["supervisions"]["text"])

        try:
            with torch_autocast(enabled=params.use_autocast, dtype=params.dtype):
                loss, loss_info = compute_loss(
                    params=params,
                    model=model,
                    sp=sp,
                    batch=batch,
                    is_training=True,
                    spec_augment=spec_augment,
                )
            # summary stats
            tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

            # NOTE: We use reduction==sum and loss is computed over utterances
            # in the batch and there is no normalization to it so far.
            scaler.scale(loss).backward()
            scheduler.step_batch(params.batch_idx_train)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Hướng 5: khoá các dòng embedding KHÔNG thuộc thuật ngữ.
            #
            # Phải ghi lại giá trị gốc sau step, KHÔNG chỉ mask gradient — vì
            # ScaledAdam gộp param theo axis 0 (`batched_params`, optim.py:48) nên
            # `decoder.embedding.weight` (2000, 512) thành (1, 2000, 512), khiến
            # `param_rms` là MỘT scalar cho cả bảng và `optim.py:236`
            # `delta.add_(p * scale_step)` dịch MỌI dòng — kể cả dòng đã zero grad.
            # Chỉ mask gradient là để lọt đúng đường đó.
            if _emb_lock is not None:
                _emb_lock()
        except Exception as e:
            logging.info(f"Caught exception: {e}.")
            save_bad_model()
            display_and_save_batch(batch, params=params, sp=sp)
            raise

        if params.print_diagnostics and batch_idx == 5:
            return

        if (
            rank == 0
            and params.batch_idx_train > 0
            and not params.qat  # model QAT có buffer observer → state_dict lệch model_avg
            and params.batch_idx_train % params.average_period == 0
        ):
            update_averaged_model(
                params=params,
                model_cur=model,
                model_avg=model_avg,
            )

        if (
            params.batch_idx_train > 0
            and params.batch_idx_train % params.save_every_n == 0
        ):
            save_checkpoint_with_global_batch_idx(
                out_dir=params.exp_dir,
                global_batch_idx=params.batch_idx_train,
                model=model,
                model_avg=model_avg,
                params=params,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=train_dl.sampler,
                scaler=scaler,
                rank=rank,
            )
            remove_checkpoints(
                out_dir=params.exp_dir,
                topk=params.keep_last_k,
                rank=rank,
            )

        if params.use_autocast:
            cur_grad_scale = scaler._scale.item()

            if cur_grad_scale < 0.01:
                if not saved_bad_model:
                    save_bad_model(suffix="-first-warning")
                    saved_bad_model = True
                    if not params.inf_check:
                        register_inf_check_hooks(model)
                logging.warning(f"Grad scale is small: {cur_grad_scale}")

            if cur_grad_scale < 1.0e-05:
                save_bad_model()
                raise_grad_scale_is_too_small_error(cur_grad_scale)

            # If the grad scale was less than 1, try increasing it.    The _growth_interval
            # of the grad scaler is configurable, but we can't configure it to have different
            # behavior depending on the current grad scale.
            if (
                batch_idx % 25 == 0
                and cur_grad_scale < 2.0
                or batch_idx % 100 == 0
                and cur_grad_scale < 8.0
                or batch_idx % 400 == 0
                and cur_grad_scale < 32.0
            ):
                scaler.update(cur_grad_scale * 2.0)

        if batch_idx % params.log_interval == 0:
            cur_lr = max(scheduler.get_last_lr())
            cur_grad_scale = scaler._scale.item() if params.use_autocast else 1.0

            logging.info(
                f"Epoch {params.cur_epoch}, "
                f"batch {batch_idx}, loss[{loss_info}], "
                f"tot_loss[{tot_loss}], batch size: {batch_size}, "
                f"lr: {cur_lr:.2e}, "
                + (f"grad_scale: {scaler._scale.item()}" if params.use_autocast else "")
            )
            train_iter.set_postfix(
                loss=f"{loss_info['loss'] / loss_info['frames']:.3f}",
                lr=f"{cur_lr:.2e}",
            )

            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train/learning_rate", cur_lr, params.batch_idx_train
                )

                loss_info.write_summary(
                    tb_writer, "train/current_", params.batch_idx_train
                )
                tot_loss.write_summary(tb_writer, "train/tot_", params.batch_idx_train)
                if params.use_autocast:
                    tb_writer.add_scalar(
                        "train/grad_scale", cur_grad_scale, params.batch_idx_train
                    )


    params.num_batches_per_epoch = batch_idx + 1  # lưu để tqdm epoch sau có ETA
    loss_value = tot_loss["loss"] / tot_loss["frames"]
    params.train_loss = loss_value
    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss


def run(rank, world_size, args):
    """
    Args:
      rank:
        It is a value between 0 and `world_size-1`, which is
        passed automatically by `mp.spawn()` in :func:`main`.
        The node with rank 0 is responsible for saving checkpoint.
      world_size:
        Number of GPUs for DDP training.
      args:
        The return value of get_parser().parse_args()
    """
    params = get_params()
    params.update(vars(args))

    fix_random_seed(params.seed)
    if world_size > 1:
        setup_dist(rank, world_size, params.master_port)

    setup_logger(f"{params.exp_dir}/log/log-train")

    class LessSpamFilter(logging.Filter):
        def filter(self, record):
            # Chặn đứng các vòng lặp log rác từ scaling và optim
            if record.filename in ["scaling.py", "optim.py", "asr_datamodule.py", "zipformer.py"]:
                return False
            return True
    
    # Áp dụng bộ lọc cho mọi thiết lập console / file log
    for handler in logging.getLogger().handlers:
        handler.addFilter(LessSpamFilter())

    logging.info("Training started")

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    logging.info(f"Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> is defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.sos_id = params.eos_id = sp.piece_to_id("<sos/eos>")
    params.vocab_size = sp.get_piece_size()

    if not params.use_transducer:
        if not params.use_attention_decoder:
            params.ctc_loss_scale = 1.0
        else:
            assert params.ctc_loss_scale + params.attention_decoder_loss_scale == 1.0, (
                params.ctc_loss_scale,
                params.attention_decoder_loss_scale,
            )

    if params.use_bf16:  # amp + bf16
        assert torch.cuda.is_bf16_supported(), "Your GPU does not support bf16!"
        assert not params.use_fp16, "You can only use either fp16 or bf16"
        params.dtype = torch.bfloat16
        params.use_autocast = True
    elif params.use_fp16:  # amp + fp16
        params.dtype = torch.float16
        params.use_autocast = True
    else:  # fp32
        params.dtype = torch.float32
        params.use_autocast = False

    logging.info(f"Using dtype={params.dtype}")
    logging.info(f"Use AMP={params.use_autocast}")

    # logging.info(params)

    logging.info("About to create model")
    model = get_model(params)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    if params.use_cr_ctc:
        assert params.use_ctc
        assert not params.enable_spec_aug  # we will do spec_augment in model.py
        spec_augment = get_spec_augment(params)
    else:
        spec_augment = None

    assert params.save_every_n >= params.average_period
    model_avg: Optional[nn.Module] = None
    if rank == 0:
        # model_avg is only used with rank 0
        model_avg = copy.deepcopy(model).to(torch.float64)

    assert params.start_epoch > 0, params.start_epoch
    checkpoints = load_checkpoint_if_available(
        params=params, model=model, model_avg=model_avg
    )

    # ── QAT: nướng scale vào weight TRƯỚC, rồi mới gắn FakeQuantize ────────────
    # Thứ tự bắt buộc: load xong checkpoint (còn ScaledLinear, key khớp) →
    # convert_scaled_to_non_scaled (scale thành hệ số trong weight) → prepare_qat
    # (đổi Linear/Conv thành bản QAT + thêm observer). Đảo thứ tự là load lệch key.
    # Lưu ý: checkpoint QAT lưu RA không còn scale param — export/avg sau đó là
    # no-op khi gặp convert_scaled_to_non_scaled lần nữa.
    if params.qat:
        from scaling_converter import convert_scaled_to_non_scaled
        from torch.ao.quantization import (
            disable_fake_quant,
            enable_fake_quant,
            get_default_qat_qconfig,
            prepare_qat,
        )

        convert_scaled_to_non_scaled(model, inplace=True)
        if params.qat_bits == 4:
            # QAT weight-only int4: activation=None (không quant activation, đúng
            # MatMulNBits khi deploy), weight fake-quant 16 mức đối xứng per-channel.
            from torch.ao.quantization import FakeQuantize, MovingAveragePerChannelMinMaxObserver, QConfig
            qconf = QConfig(
                activation=None,
                weight=FakeQuantize.with_args(
                    observer=MovingAveragePerChannelMinMaxObserver,
                    quant_min=-8, quant_max=7,
                    dtype=torch.qint8,
                    qscheme=torch.per_channel_symmetric,
                    reduce_range=False,
                ),
            )
            model.qconfig = qconf
        else:
            model.qconfig = get_default_qat_qconfig("x86")
        prepare_qat(model, inplace=True)
        # model_avg giữ nguyên dạng float thường — KHÔNG chuẩn hoá cùng model,
        # vì vậy update_averaged_model bị chặn khi QAT (state_dict lệch key do
        # buffer của observer). Muốn average thì dùng scripts/avg_ckpt.py sau run.
        if params.start_epoch <= params.qat_calib_epochs:
            disable_fake_quant(model)
            logging.info(
                f"QAT bật: epoch {params.start_epoch} ≤ calib "
                f"{params.qat_calib_epochs} → observer-only, fake-quant TẮT"
            )
        else:
            enable_fake_quant(model)
            logging.info(
                f"QAT bật: epoch {params.start_epoch} > calib "
                f"{params.qat_calib_epochs} → fake-quant BẬT"
            )
        n_obs = sum(
            1
            for m in model.modules()
            if "FakeQuantize" in type(m).__name__ or "Observer" in type(m).__name__
        )
        logging.info(f"QAT: {n_obs} module observer/fake-quant đã gắn")

    # Resolve which modules to keep trainable
    #
    # Cú pháp TRAINABLE_MODULES: danh sách phân tách bằng dấu phẩy, mỗi mục là
    #   - PREFIX  : "decoder", "encoder.encoders.3", "joiner.decoder_proj"
    #               khớp khi name == prefix hoặc name bắt đầu bằng prefix + "."
    #   - "re:..." : biểu thức chính quy, khớp bằng re.search trên TÊN ĐẦY ĐỦ
    #
    # Chế độ regex có mặt vì BitFit cần chọn theo ĐUÔI tên
    # (`...layers.0.bypass.bypass_scale`), mà prefix không diễn đạt được. Ví dụ:
    #   TRAINABLE_MODULES='re:(bypass_scale|norm\.bias|norm\.log_scale)$'
    #
    # ⚠️ Phải bóc dấu nháy bao ngoài. `run_new.sh:42` dùng `source` (bash bóc nháy)
    # nhưng dòng 248 truyền vào container bằng `--env-file`, mà Docker KHÔNG bóc —
    # nó lấy nguyên văn sau dấu `=`. Regex thì bắt buộc phải có nháy trong file env
    # (dấu `(` `|` `$` không thoát được ở dạng trần), nên nếu không bóc ở đây thì
    # chuỗi tới nơi là `'re:...'`, không còn bắt đầu bằng `re:` ⇒ bị coi là prefix
    # ⇒ khớp 0 param. Đã kiểm bằng `docker run --env-file`.
    def _unquote(s: str) -> str:
        s = s.strip()
        if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
            return s[1:-1]
        return s

    _trainable_prefixes = []
    if _unquote(params.trainable_modules).strip():
        _trainable_prefixes = [
            _unquote(m) for m in _unquote(params.trainable_modules).split(",") if m.strip()
        ]
    elif params.freeze_encoder:
        _trainable_prefixes = [
            "ctc_output", "joiner", "decoder",
            "simple_am_proj", "simple_lm_proj", "attention_decoder",
        ]

    if _trainable_prefixes:
        import re as _re
        _pats = []
        for p in _trainable_prefixes:
            if p.startswith("re:"):
                try:
                    _pats.append(("re", _re.compile(p[3:])))
                except _re.error as e:
                    raise ValueError(f"TRAINABLE_MODULES: regex hỏng {p[3:]!r}: {e}")
            else:
                _pats.append(("prefix", p))

        def _keep(name: str) -> bool:
            for kind, pat in _pats:
                if kind == "re":
                    if pat.search(name):
                        return True
                elif name == pat or name.startswith(pat + "."):
                    return True
            return False

        _base = model.module if isinstance(model, DDP) else model
        for name, param in _base.named_parameters():
            param.requires_grad_(_keep(name))
        frozen = sum(p.numel() for p in _base.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in _base.parameters() if p.requires_grad)
        total = frozen + trainable

        # BẮT BUỘC: khớp 0 param là lỗi cấu hình, KHÔNG được chạy tiếp.
        # Trước đây một chuỗi sai (vd "bypass") lọt qua đây và đóng băng toàn bộ
        # model, training chạy 10 epoch mà không cập nhật một trọng số nào.
        if trainable == 0:
            _sample = [n for n, _ in _base.named_parameters()][:5]
            raise ValueError(
                f"TRAINABLE_MODULES={params.trainable_modules!r} khớp 0 param.\n"
                f"  Đã thử {len(_pats)} mẫu: {_trainable_prefixes}\n"
                f"  Tên param bắt đầu bằng, ví dụ: {_sample}\n"
                f"  Tra tên đầy đủ trong architecture.txt (sinh bởi scripts/print_arch.py).\n"
                f"  Chọn theo đuôi tên thì phải dùng tiền tố 're:'."
            )

        _n_tensor = sum(1 for _, p in _base.named_parameters() if p.requires_grad)
        logging.info(
            f"Partial freeze — trainable: {_trainable_prefixes}\n"
            f"  frozen   : {frozen:,} params ({100*frozen/total:.1f}%)\n"
            f"  trainable: {trainable:,} params ({100*trainable/total:.1f}%) "
            f"trong {_n_tensor} tensor"
        )
        # Liệt kê chi tiết khi phạm vi nhỏ (BitFit ra 87 tensor) — với run mở cả
        # encoder thì 745 dòng là rác, chỉ in tóm tắt.
        if _n_tensor <= 100:
            for n, p in _base.named_parameters():
                if p.requires_grad:
                    logging.info(f"    {n:<62} {str(tuple(p.shape)):>12} {p.numel():>9,}")

    model.to(device)

    # Hướng 5: khoá dòng embedding ngoài thuật ngữ. Dựng SAU .to(device) để mask và
    # bản lưu nằm cùng device với weight, và TRƯỚC DDP để lấy được model gốc.
    global _emb_lock
    if params.emb_term_mask.strip():
        _wm = _unquote(params.emb_term_mask)
        if not params.trainable_modules.strip():
            raise ValueError(
                "--emb-term-mask cần đi cùng --trainable-modules chứa 'decoder' "
                "(hoặc 'decoder.embedding'); nếu không thì toàn bộ model vẫn train "
                "và việc khoá dòng embedding là vô nghĩa."
            )
        _emb_lock = make_emb_term_lock(model, sp, _wm)
        _emb_lock()          # áp ngay: đảm bảo điểm khởi đầu đúng bằng ep35

    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # Exclude frozen top-level modules from optimizer.
    #
    # `get_parameter_groups_with_lrs(freeze_modules=...)` chỉ lọc theo TÊN CẤP 1
    # nên đây chỉ bỏ được nguyên khối top-level; phần đóng băng nằm sâu bên trong
    # một khối vẫn ở lại optimizer (ScaledAdam chịu được `p.grad is None`,
    # optim.py:111 thay bằng zeros_like).
    #
    # Suy từ requires_grad THẬT chứ không từ chuỗi prefix: chuỗi có thể là regex
    # (BitFit) và khi đó `p.split(".")[0]` cho ra rác, làm `encoder` bị loại khỏi
    # optimizer trong khi chính nó chứa toàn bộ param cần train ⇒ train 10 epoch
    # mà không có gì thay đổi.
    _frozen_top = []
    if _trainable_prefixes:
        _base = model.module if isinstance(model, DDP) else model
        all_top = {name.split(".")[0] for name, _ in _base.named_parameters()}
        trainable_top = {
            name.split(".")[0] for name, p in _base.named_parameters() if p.requires_grad
        }
        _frozen_top = sorted(all_top - trainable_top)
        logging.info(f"Optimizer bỏ hẳn top-level: {_frozen_top}")

    optimizer = ScaledAdam(
        get_parameter_groups_with_lrs(
            model,
            lr=params.base_lr,
            include_names=True,
            freeze_modules=_frozen_top,
        ),
        lr=params.base_lr,  # should have no effect
        clipping_scale=2.0,
    )

    scheduler = Eden(optimizer, params.lr_batches, params.lr_epochs, warmup_start=0.1)

    if checkpoints and "optimizer" in checkpoints:
        logging.info("Loading optimizer state dict")
        optimizer.load_state_dict(checkpoints["optimizer"])

    if (
        checkpoints
        and "scheduler" in checkpoints
        and checkpoints["scheduler"] is not None
    ):
        logging.info("Loading scheduler state dict")
        scheduler.load_state_dict(checkpoints["scheduler"])

    if params.print_diagnostics:
        opts = diagnostics.TensorDiagnosticOptions(
            512
        )  # allow 4 megabytes per sub-module
        diagnostic = diagnostics.attach_diagnostics(model, opts)

    if params.inf_check:
        register_inf_check_hooks(model)

    librispeech = LibriSpeechAsrDataModule(args)

    if params.full_libri:
        train_cuts = librispeech.train_all_shuf_cuts()

        # previously we used the following code to load all training cuts,
        # strictly speaking, shuffled training cuts should be used instead,
        # but we leave the code here to demonstrate that there is an option
        # like this to combine multiple cutsets

        # train_cuts = librispeech.train_clean_100_cuts()
        # train_cuts += librispeech.train_clean_360_cuts()
        # train_cuts += librispeech.train_other_500_cuts()
    else:
        train_cuts = librispeech.train_clean_100_cuts()

    def remove_short_and_long_utt(c: Cut):
        # Giữ cut có 0.1s <= duration <= 30.0s VÀ nhiều hơn 2 từ.
        # (icefall gốc dùng 1.0–20.0s; nới ra để không mất 37.886 cut trong vùng
        #  0.1–1s và 1.030 cut trong vùng 20–30s của corpus 2.340h này.)
        #
        # Đo trên manifest 2.172.159 cut:
        #   1.0–20.0s            → giữ 2.132.973 (98,20%)
        #   0.1–30.0s + từ > 2   → giữ 2.154.622 (99,19%)
        #   loại: duration > 30s = 270 cut | số từ <= 2 = 17.267 cut | < 0.1s = 0 cut
        #
        # Lọc câu <= 2 từ vì chúng làm WER rất nhiễu (sai 1 từ trên 2 từ = 50%) và
        # phần lớn là filler/lỗi cắt đoạn. Ràng buộc T >= S bên dưới vẫn là chốt cuối
        # cho cut quá ngắn so với số token, nên biên 0.1s chủ yếu để không cấm oan.
        if c.duration < 0.1 or c.duration > 30.0:
            return False

        _sup = c.supervisions[0] if c.supervisions else None
        _text = (_sup.text if _sup is not None else "") or ""
        if len(_text.split()) <= 2:
            return False

        # In pruned RNN-T, we require that T >= S
        # where T is the number of feature frames after subsampling
        # and S is the number of tokens in the utterance

        # In ./zipformer.py, the conv module uses the following expression
        # for subsampling
        T = ((c.num_frames - 7) // 2 + 1) // 2
        tokens = sp.encode(c.supervisions[0].text, out_type=str)

        # For CTC `(T - 2)  < len(tokens)` is needed. otherwise inf. in loss appears.
        # For Transducer `T < len(tokens)` was okay.
        if (T - 2) < len(tokens):
            logging.warning(
                f"Exclude cut with ID {c.id} from training (too many supervision tokens). "
                f"Number of frames (before subsampling): {c.num_frames}. "
                f"Number of frames (after subsampling): {T}. "
                f"Text: {c.supervisions[0].text}. "
                f"Tokens: {tokens}. "
                f"Number of tokens: {len(tokens)}"
            )
            return False

        return True

    train_cuts = train_cuts.filter(remove_short_and_long_utt)

    if params.start_batch > 0 and checkpoints and "sampler" in checkpoints:
        # We only load the sampler's state dict when it loads a checkpoint
        # saved in the middle of an epoch
        sampler_state_dict = checkpoints["sampler"]
    else:
        sampler_state_dict = None

    train_dl = librispeech.train_dataloaders(
        train_cuts, sampler_state_dict=sampler_state_dict
    )

    # ── BAND_CLAMP: mô phỏng băng tần hẹp của val ngay trên feature ────────────
    # Rẻ hơn BAND_LIMIT_SR của prepare_data.py (0 phút vs 6h48m extract) và thực ra
    # khớp CHẶT hơn, vì nó tác động thẳng vào đại lượng cần khớp thay vì đi qua
    # resampler. Cái giá: sàn phẳng nhân tạo, không nguồn audio thật nào có.
    #
    # Chèn vào ĐẦU input_transforms để chạy TRƯỚC SpecAugment. Bắt buộc: SpecAugment
    # mặc định num_feature_masks=2, features_mask_size=27 trên 80 bin, tức nó có thể
    # che trọn dải quy chiếu 6–23 → quy chiếu sai → clamp sai.
    # Chỉ train_dl có input_transforms; valid/train_eval dùng K2SpeechRecognitionDataset
    # không transform, nên val KHÔNG bị chạm và WER vẫn so được với baseline cũ.
    _clamp_bin = int(os.environ.get("BAND_CLAMP_BIN", "0") or 0)
    _check_band_config(_clamp_bin)

    if _clamp_bin > 0:
        _tnfms = getattr(getattr(train_dl, "dataset", None), "input_transforms", None)
        if _tnfms is None:
            raise RuntimeError(
                "BAND_CLAMP_BIN được bật nhưng không tìm thấy train_dl.dataset.input_transforms "
                "— lhotse đã đổi API, phải kiểm lại trước khi tin kết quả"
            )
        _clamp = _BandClamp(_clamp_bin)
        _tnfms.insert(0, _clamp)
        logging.info(
            f"BAND_CLAMP bật: clamp bin {_clamp_bin}–{_clamp.n_mels - 1} xuống sàn của val, "
            f"trước {len(_tnfms) - 1} transform khác ({[type(t).__name__ for t in _tnfms[1:]]}). "
            f"CHỈ áp cho train_dl, val giữ nguyên."
        )
    else:
        logging.info("BAND_CLAMP tắt (BAND_CLAMP_BIN=0)")

    # prepare_data.py ghi CÙNG một val set vào cả 4 manifest (dev-clean, dev-other,
    # test-clean, test-other) để tương thích với icefall recipe. Nên chỉ lấy dev-clean
    # là đã đủ toàn bộ val set — cộng thêm dev-other chỉ làm val bị đếm 2 lần.
    valid_cuts = librispeech.dev_clean_cuts()

    # Chỉ đánh giá trên audio >= EVAL_MIN_DURATION giây. Lý do: remove_short_and_long_utt
    # loại cut < 1.0s khỏi tập TRAIN, nên model chưa bao giờ được train trên câu ngắn hơn
    # thế — đo nó trên val là đo một chế độ chưa hề học. Val có 71/549 câu dưới 1s (13%).
    # LƯU Ý: bật cái này làm val set ĐỔI, nên WER không so trực tiếp được với số cũ
    # (33.20% của epoch-35, 33.42% của epoch-36). Đặt EVAL_MIN_DURATION=0 để tắt.
    _eval_min_dur = float(os.environ.get("EVAL_MIN_DURATION", "1.0") or 0)
    if _eval_min_dur > 0:
        _n_before = len(valid_cuts)
        valid_cuts = valid_cuts.filter(
            lambda c: c.duration >= _eval_min_dur
        ).to_eager()
        logging.info(
            f"VAL filter duration >= {_eval_min_dur}s: {_n_before} → {len(valid_cuts)} cut "
            f"(loại {_n_before - len(valid_cuts)})"
        )

    valid_dl = librispeech.valid_dataloaders(valid_cuts)

    # ── Dataloader để đo WER trên tập TRAIN ───────────────────────────────────
    # Dùng subset cố định (deterministic) + pipeline eval của valid_dataloaders
    # (không shuffle, không SpecAugment) nên số đo so sánh trực tiếp được với val.
    # KHÔNG dùng train_dl vì iterator đó đang được training tiêu thụ.
    # Số cut lấy qua TRAIN_EVAL_CUTS (đặt 0 để tắt hẳn train WER).
    _n_train_eval = int(os.environ.get("TRAIN_EVAL_CUTS", "1000") or 0)
    train_eval_dl = None
    if _n_train_eval > 0:
        # remove_short_and_long_utt cho phép cut từ 0.1s để TRAIN, nhưng khi ĐO WER thì
        # phải dùng cùng ngưỡng với val (EVAL_MIN_DURATION) — nếu không thì TRAIN WER và
        # VALID WER đo trên hai phân phối độ dài khác nhau và cái gap giữa chúng vô nghĩa.
        _train_eval_cuts = train_cuts
        if _eval_min_dur > 0:
            _train_eval_cuts = _train_eval_cuts.filter(
                lambda c: c.duration >= _eval_min_dur
            )
        train_eval_dl = librispeech.valid_dataloaders(
            _train_eval_cuts.subset(first=_n_train_eval)
        )
        logging.info(
            f"Train-set WER sẽ đo trên {_n_train_eval} cut đầu của tập train "
            f"(duration >= {_eval_min_dur}s, cùng ngưỡng với val)"
        )

    # tạm thời k dùng
    # if not params.print_diagnostics:
    #     scan_pessimistic_batches_for_oom(
    #         model=model,
    #         train_dl=train_dl,
    #         optimizer=optimizer,
    #         sp=sp,
    #         params=params,
    #         spec_augment=spec_augment,
    #     )

    scaler = create_grad_scaler(enabled=params.use_autocast, init_scale=1.0)
    if checkpoints and "grad_scaler" in checkpoints:
        logging.info("Loading grad scaler state dict")
        scaler.load_state_dict(checkpoints["grad_scaler"])

    # ── EVAL_FIRST: tính WER trên train + val trước khi train ─────────────
    if os.environ.get("EVAL_FIRST", "false").lower() == "true":
        logging.info("=" * 70)
        logging.info("EVAL_FIRST=true — Computing initial WER (train + val) before training...")
        logging.info("=" * 70)

        _pfx = f"[EVAL_FIRST] epoch {params.start_epoch - 1} — Initial "
        if train_eval_dl is not None:
            log_wer(params, model, sp, train_eval_dl, "TRAIN", tb_writer, prefix=_pfx)
        log_wer(params, model, sp, valid_dl, "VALID", tb_writer, prefix=_pfx)
        logging.info("=" * 70)
        logging.info("Starting training...")
        logging.info("=" * 70)

    for epoch in range(params.start_epoch, params.num_epochs + 1):
        # ── QAT: gạt fake-quant theo giai đoạn epoch ──────────────────────────
        # Epoch ≤ qat_calib_epochs: chỉ observer (rảnh tay học scale/zero-point).
        # Sau đó: fake-quant BẬT — weight phải chịu nhiễu lượng tử khi train.
        if params.qat:
            from torch.ao.quantization import disable_fake_quant, enable_fake_quant

            if epoch <= params.qat_calib_epochs:
                disable_fake_quant(model)
                _qat_mode = "observer-only"
            else:
                enable_fake_quant(model)
                _qat_mode = "fake-quant"
            logging.info(f"QAT epoch {epoch}: {_qat_mode}")
            if tb_writer is not None:
                tb_writer.add_scalar("qat/fake_quant", int(epoch > params.qat_calib_epochs), epoch)

        scheduler.step_epoch(epoch - 1)
        fix_random_seed(params.seed + epoch - 1)
        train_dl.sampler.set_epoch(epoch - 1)

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        params.cur_epoch = epoch

        train_one_epoch(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            sp=sp,
            train_dl=train_dl,
            valid_dl=valid_dl,
            scaler=scaler,
            spec_augment=spec_augment,
            tb_writer=tb_writer,
            world_size=world_size,
            rank=rank,
        )

        if params.print_diagnostics:
            diagnostic.print_diagnostics()
            break

        # ── Validation loss + WER sau mỗi epoch ───────────────────────────
        logging.info("========================= Computing validation loss ===========================")
        valid_info = compute_validation_loss(
            params=params,
            model=model,
            sp=sp,
            valid_dl=valid_dl,
            world_size=world_size,
        )
        logging.info(f"Epoch {params.cur_epoch}, validation: {valid_info}")
        if tb_writer is not None:
            valid_info.write_summary(tb_writer, "train/valid_", params.batch_idx_train)

        logging.info("========================= Computing WER (train + val) =========================")
        _pfx = f"Epoch {params.cur_epoch}, "
        train_wers = {}
        if train_eval_dl is not None:
            train_wers = log_wer(
                params, model, sp, train_eval_dl, "TRAIN", tb_writer, prefix=_pfx
            )
        val_wers = log_wer(params, model, sp, valid_dl, "VALID", tb_writer, prefix=_pfx)

        for mo in val_wers:
            if mo in train_wers:
                logging.info(
                    f"{_pfx}WER gap [{mo}] (val - train): "
                    f"{(val_wers[mo] - train_wers[mo]) * 100:+.2f}% "
                    f"[gap lớn = domain mismatch / overfit; cả hai cao = model chưa học được]"
                )

        save_checkpoint(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=rank,
        )

    logging.info("Done!")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


def display_and_save_batch(
    batch: dict,
    params: AttributeDict,
    sp: spm.SentencePieceProcessor,
) -> None:
    """Display the batch statistics and save the batch into disk.

    Args:
      batch:
        A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
        for the content in it.
      params:
        Parameters for training. See :func:`get_params`.
      sp:
        The BPE model.
    """
    from lhotse.utils import uuid4

    filename = f"{params.exp_dir}/batch-{uuid4()}.pt"
    logging.info(f"Saving batch to {filename}")
    torch.save(batch, filename)

    supervisions = batch["supervisions"]
    features = batch["inputs"]

    logging.info(f"features shape: {features.shape}")

    y = sp.encode(supervisions["text"], out_type=int)
    num_tokens = sum(len(i) for i in y)
    logging.info(f"num tokens: {num_tokens}")


def scan_pessimistic_batches_for_oom(
    model: Union[nn.Module, DDP],
    train_dl: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    sp: spm.SentencePieceProcessor,
    params: AttributeDict,
    spec_augment: Optional[SpecAugment] = None,
):
    from lhotse.dataset import find_pessimistic_batches

    logging.info(
        "Sanity check -- see if any of the batches in epoch 1 would cause OOM."
    )
    batches, crit_values = find_pessimistic_batches(train_dl.sampler)
    for criterion, cuts in batches.items():
        batch = train_dl.dataset[cuts]
        try:
            with torch_autocast(enabled=params.use_autocast, dtype=params.dtype):
                loss, _ = compute_loss(
                    params=params,
                    model=model,
                    sp=sp,
                    batch=batch,
                    is_training=True,
                    spec_augment=spec_augment,
                )
            loss.backward()
            optimizer.zero_grad()
        except Exception as e:
            if "CUDA out of memory" in str(e):
                logging.error(
                    "Your GPU ran out of memory with the current "
                    "max_duration setting. We recommend decreasing "
                    "max_duration and trying again.\n"
                    f"Failing criterion: {criterion} "
                    f"(={crit_values[criterion]}) ..."
                )
            display_and_save_batch(batch, params=params, sp=sp)
            raise
        # logging.info(
        #     f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
        # )


def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()
