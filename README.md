# train-stt — Zipformer ASR tiếng Việt (from-scratch)

Zipformer2 RNN-T + CTC (~69,7M param, BPE 2000 tự train), train **from-scratch 100 epoch**
trên ~430h tiếng Việt công khai, không kế thừa checkpoint ngoài. Checkpoint tốt nhất:
`avg-96-100.pt` (average 5 epoch cuối). Inference qua **ONNX + sherpa-onnx** (CPU).

## Kết quả

### Data train (~430h, 10 dataset công khai)

| dataset | cut | giờ | % |
|---|---:|---:|---:|
| vimd | 18.949 | 102,6 | 23,8% |
| vlsp2020 | 56.427 | 101,4 | 23,5% |
| lsvsc | 56.823 | 100,7 | 23,4% |
| fosd | 25.917 | 30,2 | 7,0% |
| onevoice | 13.076 | 22,2 | 5,2% |
| gigaspeech | 11.035 | 21,3 | 4,9% |
| vietmed | 9.207 | 15,9 | 3,7% |
| multimed | 9.122 | 15,8 | 3,7% |
| vivos | 12.420 | 15,7 | 3,6% |
| vietmdd | 4.266 | 4,9 | 1,1% |

### Benchmark trên local data (val nội bộ 10.830 cut)

Chạy bằng sherpa-onnx 2.0.1. `prod` = decode `beam 16 + hotwords + blank-penalty 1`
(chế độ chạy thật); `greedy` để đối chiếu. **Key-recall** = % từ khóa đặc trưng của ref
(từ xuất hiện ở ≤ 3% số cut val — 121.539 key / 10.830 cut) xuất hiện đúng trong hyp.
Tốc độ đo **A/B công bằng**: 1 process nạp đồng thời 3 model, cùng config (greedy, batch 1,
1 thread), interleave theo block × 3 round lấy median, ghim 1 core — `scripts/bench_ab.py`,
đơn vị × real-time.

| Bundle | greedy WER / CER / Key | prod WER / CER / Key | Δ WER | Size | Tốc độ |
|---|---|---|---:|---:|---:|
| fp32 (avg-96-100) | 8,00 / 5,61 / 92,3 | **7,69 / 5,20 / 92,9** | — | 272 MB | 29,3× RT |
| int8 PTQ | 7,99 / 5,59 / 92,3 | 7,75 / 5,24 / 92,8 | +0,06 | 75 MB | **44,1× RT** |
| int4 PTQ | 8,12 / 5,71 / 92,1 | 7,79 / 5,28 / 92,7 | +0,10 | **52 MB** | 27,2× RT |

⇒ **int8 thắng tuyệt đối** (nhanh gấp 1,5× fp32 nhờ VNNI) và gần như không mất độ chính xác.
**int4 chỉ đáng dùng khi cần size nhỏ** (52 MB, 5,2× so fp32): nó nhỏ nhất nhưng **chậm hơn
cả fp32** trên CPU — kernel MatMulNBits (tra bảng + rescale từng block 128) kém hiệu quả hơn
nhiều so với integer GEMM int8 có VNNI; cái lợi băng thông của weight nhỏ hơn không đủ bù ở
model 69M. QAT đã thử và **thua PTQ** (train tiếp làm trôi weight nhiều hơn phần lỗi quant nó
cứu) — chi tiết trong `results.md`.

## Tải model & chạy thử

Cả 3 bundle nằm ở [Releases](https://github.com/Ripefog/train-stt/releases) (không nhét
binary vào git):

| Bundle | Link | Dùng khi |
|---|---|---|
| int8 (khuyên dùng) | `releases/download/v1.0.0/zipformer-vi-int8.zip` | cân bằng — nhanh nhất |
| int4 | `releases/download/v1.0.0/zipformer-vi-int4.zip` | cần gọn nhất (52 MB) |
| fp32 | `releases/download/v1.0.0/zipformer-vi-fp32.zip` | cần WER thấp nhất |

```bash
pip install sherpa-onnx==2.0.1 soundfile

curl -L -o zipformer-vi-int8.zip \
  https://github.com/Ripefog/train-stt/releases/download/v1.0.0/zipformer-vi-int8.zip
unzip zipformer-vi-int8.zip -d zipformer-vi
```

```python
import soundfile as sf
import sherpa_onnx

rec = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder="zipformer-vi/encoder-avg-96-100.int8.onnx",
    decoder="zipformer-vi/decoder-avg-96-100.int8.onnx",
    joiner="zipformer-vi/joiner-avg-96-100.int8.onnx",
    tokens="zipformer-vi/tokens.txt",
    num_threads=1,
    decoding_method="modified_beam_search",
    beam_size=16,
    blank_penalty=1.0,          # tuỳ chọn — +0,09 pp WER trên benchmark
    hotwords_file="hotwords.txt",  # tuỳ chọn — từ khoá của bạn, ví dụ xem config/hotwords.txt
    hotwords_score=2.0,
)

samples, sr = sf.read("audio.wav", dtype="float32")   # audio 16 kHz
stream = rec.create_stream()
stream.accept_waveform(sr, samples)
rec.decode_stream(stream)
print(stream.result.text)
```

Muốn nhanh nhất mà chấp nhận mất ~0,1 pp: `decoding_method="greedy_search"` và bỏ
`blank_penalty` / `hotwords_*`.

## Cấu trúc repo

```
scripts/          # toàn bộ pipeline — bash + python đều ở đây
  download_datasets.py, convert_*.py   # dựng corpus (~430h)
  train_bpe_scratch.py                 # BPE 2000 tự train
  run*.sh, train.py, model.py          # training (icefall, docker hoặc host venv)
  export_onnx.py, export.sh            # xuất ONNX fp32/int8/int4
  eval_onnx.py, key_recall.py          # benchmark WER/CER/key-recall
  bench_speed.py                       # tốc độ decode
  launch_qat*_host.sh, watch_qat_chain.sh  # QAT chạy thẳng trên host
results.md        # MỌI số đo, cấu hình, hướng đã loại — đọc trước khi hỏi
CLAUDE.md         # bẫy kỹ thuật đã trả giá (vocab, resume, GPU dùng chung…)
```

Tái tạo bảng benchmark: eval per-cut bằng `scripts/eval_onnx.py` (xem argparse trong file),
rồi `python scripts/key_recall.py eval_fp32.jsonl eval_int8.jsonl eval_int4.jsonl`.

## License

[MIT](LICENSE) — code và model đều MIT.
