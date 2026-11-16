# CLAUDE.md

File này cung cấp hướng dẫn cho Claude Code (claude.ai/code) khi làm việc với code trong repo này.

📊 **Mọi số đo train/eval nằm ở [`results.md`](results.md)** — kết quả từng run, bảng WER +
recall thuật ngữ, cấu hình tốt nhất hiện tại, và danh sách hướng đã loại. File này chỉ giữ
cách repo hoạt động và các bẫy kỹ thuật. **Đừng thêm kết quả đo vào đây.**

## Repo này là gì

Một wrapper mỏng bọc quanh recipe Zipformer LibriSpeech của k2-fsa/icefall, đóng gói lại để train model ASR **tiếng Việt** (RNNT + CTC) trong Docker trên GPU server local (RTX 5080). Repo chỉ chứa config (`.env`, `run.sh`, `Dockerfile`) và vài script để override/điều khiển recipe zipformer gốc của icefall lúc container chạy — không vendor icefall vào repo (icefall được `git clone` mới hoàn toàn lúc build image).

Trong project này model được gọi thân mật là "gipformer" (xem path mặc định `ICEFALL_ZIPFORMER` trong `export_onnx.py`, và `WANDB_PROJECT=conformer-training`).

## Bài toán đang giải — đọc trước khi đề xuất bất cứ hướng train nào

**Điểm khởi đầu:** `callbacks/epoch-35.pt` ("gipformer") là **checkpoint tiếng Việt tốt nhất
đang có**, do bên thứ ba train. Nó KHÔNG phải sản phẩm của repo này.

**Data gốc train ra nó: KHÔNG có, và không tái tạo được.** Nó gồm dataset public + dataset
call-center, **số giờ không được công bố**. `BPE_URL` trỏ tới
`hynt/Zipformer-30M-RNNT-6000h` ⇒ rất có thể quy mô gốc ~6.000h (suy luận từ tên, chưa
xác nhận). Đây là ràng buộc chi phối mọi thứ bên dưới.

**Điểm yếu duy nhất cần sửa:** nó **không biết thuật ngữ FPT Telecom** — `ép pi ti`,
`quai phai`, `mô đem`, `in tơ nét`, `gia lô`… Nền tiếng Việt thì đã tốt. Số đo và tiến độ
hướng này: xem `results.md`.

**GIAO THỨC ĐO — mọi lần so checkpoint PHẢI chạy đủ 3 chế độ và báo CẢ HAI chỉ số:**

| | vì sao bắt buộc |
|---|---|
| `greedy` | mốc rẻ nhất, giữ để so với toàn bộ số cũ trong `results.md` — **không còn là chế độ prod** |
| `beam 16` | **thứ tự có thể ĐẢO so với greedy** — đã gặp: hơn 0,31 pp ở greedy nhưng thua 0,27 pp ở beam16 |
| `beam 16 + hotwords` | **chế độ prod đang chạy**; cũng là tầng duy nhất nhích được thuật ngữ (beam một mình cho **đúng +0,0**) |

Chế độ prod không hardcode nữa, đọc từ config (`config.py:366` `GIPFORMER_DECODING_METHOD =
"modified_beam_search"`, `config.py:370` beam 16) và **không** `.env` nào của Core-AI-API /
Core-AI-Worker override → prod = `beam 16 + hotwords`. Trước 2026-08-13 file này ghi prod chạy
`greedy_search`; sai, đã sửa. Nếu ai set `GIPFORMER_DECODING_METHOD=greedy_search` thì hotwords
**tắt im lặng** — sherpa-onnx chỉ biasing với `modified_beam_search` (`model_registry.py:355-370`
có log cảnh báo chỗ này).

Và phải báo **WER micro + recall thuật ngữ**: thuật ngữ chỉ chiếm 145/6.687 từ = 2,2% nên một
cải thiện thật ở đó nằm **dưới** ngưỡng nhiễu 0,45 pp của WER tổng. Đọc WER một mình đã từng
dẫn tới kết luận sai là "finetune vô ích".

⚠️ **RECALL THUẬT NGỮ: dùng `exact` (quyết định 2026-08-13).** Xếp hạng checkpoint đọc cột
`exact` của `scripts/metrics.py` — hyp phải ra đúng chuỗi của `word_mapping.json`. **Không dùng
`equiv` để xếp hạng nữa.**

Vì sao bỏ `equiv` dù nó "đúng với sản phẩm" hơn: nó phụ thuộc hai thứ nằm ngoài model — nội dung
bảng `hotword_entries` và tham số `--min-variant-len` — nên **không so được giữa hai lần đo cách
nhau vài tuần**. Ngày 08-12 tham số đó bị hạ từ 4 xuống 0 (nhận `"ép"`/`"ếp"`, khiến
`"bị ép lại mấy lần"` tính là đọc đúng FPT) và việc đó **lật dấu 3 kết luận**; 08-13 trả về 4 thì
3 kết luận lật ngược lại. `exact` không có tham số nào để chỉnh.

⚠️ **Cái giá, phải nhớ khi báo cáo ra ngoài:** prod CÓ map biến thể → canonical sau ASR
(`NgheChuanAI-worker/app/modules/transcribe/services/transcribe.py:220 _load_hotword_map`, bảng
MySQL `hotword_entries`: 70 canonical / 214 biến thể), nên `ép bi ti` tới tay khách vẫn thành
**FPT**. Do đó `exact` **thấp hơn thứ khách thực nhận ~9÷12 pp** (`B_ep40 beam16+hw+bp1`: exact
45,5% ↔ sau map 57,2%). `exact` là thước để **so checkpoint**, không phải con số sản phẩm.

Một kết luận từ thời `equiv` vẫn còn giá trị và không được quên: **ĐỪNG bỏ biến thể cạnh tranh
khỏi `hotwords.txt`.** Từng nghĩ `QUAY PHAI`/`ÉP BI TI` "cạnh tranh làm loãng boost" nên xoá —
sai: bỏ hotwords thì `quai phai` từ 9 lần đúng về **0**.

```bash
scripts/eval_beam.py  --beams 16 --hotwords config/hotwords.txt --blank-penalties 0,1   # val
scripts/infer_wav.py  --beam 16  --hotwords config/hotwords.txt --blank-penalties 0,1   # audio rời
```

**Data đang có để finetune:** 2.100h public + **200h S2T FPT Telecom** (`clean` + `noise`,
là TTS 14 giọng 8 kHz, không phải thoại thật).

**Quyết định: finetune CHỈ trên 200h FPT.** Lý do: 2.340h của ta **không phải** data gốc
mà là một hỗn hợp KHÁC và KÉM hơn — nó không có call-center thật, ~92% là read speech/TTS.
Train trên nó là kéo model từ phân phối có thoại thật sang phân phối đọc sách.

**Đây là chìa khoá để đọc mọi số trong `results.md`:** finetune trên 2.340h làm val xấu đi
KHÔNG phải vì "lr sai" hay "thiếu epoch" — mà vì **thay data tốt bằng data kém**.

⚠️ **Hai hệ quả của việc không có data gốc, phải nhớ khi chọn kỹ thuật:**
1. `finetune.py --use-mux` theo đúng thiết kế (trộn lại **data gốc**) là **không làm được**.
   2.100h public của ta là hàng thay thế, KHÔNG cùng phân phối — trộn nó vào là đưa lại
   đúng thứ đã gây ra +1,16 pp.
2. Không có data gốc ⇒ không có cách nào "train lại từ đầu cho đúng". Mọi hướng đều là
   **thêm vào một checkpoint không sở hữu**, nên chống quên (đóng băng phạm vi / LoRA / adapter)
   quan trọng hơn là tối ưu lr.

⚠️ 200h FPT là **TTS**, không phải thoại call-center thật — nó cũng chỉ là hàng thay thế
cho domain đích. Đó là lý do trần của hướng này bị chặn, và là lý do 20h thoại thật (nếu
có) đáng giá hơn 200h TTS này.

## Lệnh thường dùng

**Train** (tự build image nếu cần, chạy container ở dạng detached):
```bash
scripts/run.sh
```
Toàn bộ hành vi được điều khiển qua `.env` (copy từ `.env.example` nếu chưa có — `run.sh` sẽ từ chối chạy nếu thiếu file này). Các flag quan trọng:
- `SKIP_PREPROCESSING` — `false` sẽ trích xuất lại fbank từ `/data` qua `prepare_data.py`; `true` dùng lại manifest có sẵn. **Rất chậm và không có chế độ incremental**: nó quét lại toàn bộ, thêm 1 file cũng trả lại đủ chi phí. Đo thực trên corpus 2.340h / 2.172.159 file (2026-08-05): **scan 7h59m + extract 6h48m = 14,84 giờ**, ra 76,9 GB fbank. Scan bị chặn bởi IOPS ổ cơ, không phải CPU — 13,2 ms/file ≈ đúng một lần seek 7200rpm, `iostat` cho `%util 99%` ở chỉ 1,87 MB/s, nên tăng `NUM_WORKERS` không giúp gì. `entrypoint.sh` chỉ kiểm manifest **có tồn tại**, không kiểm số cut, nên `SKIP_PREPROCESSING=true` với manifest rác vẫn chạy và cho ra rác.
- `USE_TRANSDUCER` / `USE_CTC` — quyết định `model.py`'s `AsrModel` build head nào. Key `ctc_output.*` / `joiner.*` trong checkpoint phải khớp với 2 flag này, không thì `load_state_dict` sẽ lệch.
- `START_EPOCH` — **nghiêm ngặt, không auto-detect** (đổi ở commit `889def2`). `START_EPOCH=N` load đúng `callbacks/epoch-{N-1}.pt` và train từ epoch N; không thấy file đó thì `entrypoint.sh` fail luôn chứ không tự dò checkpoint khác. Hệ quả: đang có `epoch-36.pt` mà để `START_EPOCH=36` thì nó load `epoch-35.pt` và **ghi đè** `epoch-36.pt`.
- `TEST_MODE=true` + `TEST_SAMPLES` — giới hạn số file mỗi dataset trong `prepare_data.py` để test nhanh cả pipeline.
- `DETACH=true` (mặc định) → `docker run -d`, xem log bằng `docker logs -f <container_name>`. **Luôn dùng `true`.** `DETACH=false` chạy `docker run --rm -it`: container chết theo terminal bị ngắt, và `--rm` **xoá container cùng toàn bộ `docker logs` và exit code** khi nó thoát. Ngày 2026-08-05 training chết lúc epoch 37 batch 100 và nguyên nhân không truy được vì container đã bị xoá. May là icefall vẫn ghi log ra `callbacks/log/log-train-*` nên còn dấu vết trên đĩa — đó là nguồn log duy nhất đáng tin.
- `SAVE_FBANK` — `true` thì sau khi extract xong, `prepare_data.py` copy `fbank_data/` → `fbank_data_save/`. Phải **copy thật**, không hard link: `to_file()` và `LilcomChunkyWriter` đều mở file bằng `open(path,'wb')` nên truncate **tại chỗ**, giữ nguyên inode — hard link sẽ bị phá theo (đã test xác nhận). Hook từ chối ghi đè bản lưu có **nhiều cut hơn**, để một run `TEST_MODE` không xoá được bản đầy đủ.
- `LOG_INTERVAL` (mặc định `200`) — in loss mỗi N batch. Với 2.340h thì 1 epoch ≈ 14.450 batch, ở mức 50 của icefall gốc sẽ ra ~289 dòng/epoch.
- `EVAL_MIN_DURATION` (mặc định `1.0`) — chỉ đo WER trên audio ≥ N giây, áp cho **cả val và train-eval** để `WER gap` giữa hai cái có nghĩa. Val 549 → 478 cut (loại 71 câu dưới 1s, nhiều câu trong đó là lỗi cắt đoạn: 0,40s mà 5 từ). Đổi giá trị này là **đổi val set**, nên WER không so được với số của lần đo trước.
- `USE_SCREEN=true` → tự chạy trong 1 session `screen` mới thay vì detach qua Docker.

**Export ONNX + inference nhanh:**
```bash
scripts/export.sh [--int8] [--checkpoint other.pt] [-- --n-samples 5]
```
Chạy `scripts/export_onnx.py` rồi `scripts/infer_onnx.py` trên 3 mẫu đầu tập val. Lưu ý: `scripts/infer_onnx.py` **hiện không tồn tại** trong `scripts/` — bước 2 của `export.sh` sẽ fail cho tới khi file này được thêm vào.

**Không có test suite / linter** nào được định nghĩa trong repo; `decode.py` là script decode gốc của icefall, hiện chưa được gọi từ script nào trong repo này.

## Kiến trúc

**Docker là đơn vị thực thi.** `run.sh` chạy `docker run` với các mount host↔container sau (sửa trực tiếp trong `run.sh` nếu muốn đổi path, không phải env var):
- `/data` (ro) — corpus training, xem phần Format dataset bên dưới.
- `data/fbank` — feature shard Lhotse fbank + manifest `librispeech_cuts_*.jsonl.gz`. Được sinh lại mỗi khi `SKIP_PREPROCESSING=false`; việc này sẽ **ghi đè** manifest đang có ở đó.
- `data/lang_bpe_500` — `bpe.model` / `tokens.txt`. **Vocabulary ở đây phải đúng y hệt vocabulary đã dùng để train checkpoint mà bạn resume** — đổi sang `bpe.model` khác (dù cùng `vocab_size`) sẽ làm hỏng ngầm output/embedding layer chứ không báo lỗi, vì token ID không còn trỏ đúng subword nữa. Không giả định 2 file `bpe.model` cùng vocab_size là dùng thay thế được cho nhau; muốn chắc thì diff trực tiếp danh sách piece (qua `sentencepiece_model_pb2`) với `bpe.model.bak` (bản gốc chưa xử lý, được lưu ở bước lowercase bên dưới) của đúng thư mục `lang_bpe_500*` mà run training tạo ra checkpoint đó thực sự dùng.
- `callbacks` → `exp/` trong container — checkpoint và log training.
- `scripts` (ro) — `scripts/` của repo này, mount đè lên bản đã baked sẵn trong image. Sửa file trong `scripts/` có hiệu lực không cần rebuild image, vì bind mount đè lên `COPY scripts/ /app/scripts/` trong `Dockerfile`. **Nhưng nó có hiệu lực với process MỚI, không phải "lần `docker run` tiếp theo"** — xem cảnh báo bên dưới.
- `config/word_mapping.json` → `/app/word_mapping.json` (ro) — bảng map từ/ký hiệu cho `normalize_text`. Repo root **không** được mount, chỉ có `scripts/`, nên file này cần mount riêng; `prepare_data.py` sẽ `raise FileNotFoundError` ngay lúc import nếu không thấy.

**⚠️ ĐỪNG sửa file trong `scripts/` khi container đang chạy.** `prepare_data.py` dùng `Pool(processes=N, maxtasksperchild=1)` với start method **`spawn`**, nên mỗi khi một worker xong task, Pool spawn worker mới và worker đó **import lại `prepare_data.py` từ đĩa** — tức bản vừa sửa. Ngày 2026-08-05 việc thêm `_load_word_mapping()` (raise nếu thiếu file) giữa lúc run 15h đang chạy đã làm mọi worker thay thế chết ngay, 72 traceback trong 60 giây. Process cha không chết (nó giữ module cũ trong bộ nhớ) nên nhìn bề ngoài vẫn chạy. Vá được bằng `docker cp word_mapping.json <container>:/app/word_mapping.json` — `/app` là lớp writable, chỉ `/app/scripts` mới là mount read-only. Lần đó không mất data vì `shard_size = ceil(total/num_workers)` nên chỉ có đúng N task cho N worker, worker thay thế không nhận việc nào.

**`scripts/entrypoint.sh` là pipeline thực sự**, chạy như `ENTRYPOINT` của container:
1. Tải `bpe.model` từ `BPE_URL` nếu chưa có.
2. Chạy 1 lần duy nhất (đánh dấu bằng sự tồn tại của `bpe.model.bak`): lowercase toàn bộ piece của BPE, loại piece lạ/trùng thành `<unused_N>`, rồi sinh `tokens.txt`. Idempotent — bỏ qua nếu `.bak` đã có.
3. Nếu `SKIP_PREPROCESSING=false`: chạy `prepare_data.py` trên `/data`.
4. Copy `scripts/train.py` và `scripts/model.py` đè lên bản gốc của icefall trong checkout icefall (`cp -f ... ./train.py` / `./model.py`) — đây là cách các customization của repo này chạm được vào training loop của icefall mà không cần fork nguyên icefall.
5. Resume theo đúng `START_EPOCH` (xem trên) — load `epoch-{START_EPOCH-1}.pt`, không auto-detect.

**Format dataset mà `prepare_data.py` yêu cầu** (`/data`): mỗi dataset logic là 1 subdirectory, quét đệ quy `**/*.wav`, mỗi `.wav` phải có 1 `.txt` transcript cùng tên (thiếu `.txt` ⇒ tự bỏ qua, không báo lỗi). **Tên subdirectory quyết định việc chia train/val** — dựa theo tên, không dựa nội dung:
- Tên nằm trong `SPLIT_DATASETS` → chia 70/30 train/val.
- Tên nằm trong `VAL_DATASETS` → 100% val.
- Còn lại → 100% train.

Cả 2 set này đang nằm ở đầu file `prepare_data.py`, cần sửa trực tiếp ở đó nếu tên/vai trò dataset thay đổi.

`normalize_text` chạy theo thứ tự: **NFC → gộp dấu phân cách nghìn dạng khoảng trắng → `word_mapping.json` → đọc số thành chữ → lowercase → xóa dấu câu**. Phải khớp với cách token của BPE đã normalize (bước 2 ở trên), không thì coverage giảm ngầm. Bốn điều quan trọng:

- **NFC bắt buộc đi đầu.** Chữ Việt mã hoá tổ hợp (NFD) có dấu là combining mark (category Mn), mà `\w` không khớp Mn — nên bước xóa dấu câu sẽ biến `kết nối` thành `ke t no i`. Không exception, không cảnh báo.
- **`lang_bpe_500_gip/tokens.txt` có 2000 token và KHÔNG có token chữ số nào** (dù tên thư mục là `500`). Mọi chữ số còn sót sẽ thành `<unk>`. Đo trên manifest: **12,74% cut của data public có chữ số** — FPT `audio_text` và val thì 0. Nên bước đọc số không phải phòng thủ, nó sửa lỗi thật.
- **`word_mapping.json` chỉ chạy MỘT lượt.** Nếu thêm `"a": "b"` mà `"b": "c"` đã tồn tại thì `a` dừng ở `b`, không đi tới `c`. Kiểm bằng: không target nào được đồng thời là key của entry khác.
- **Bẫy khớp sai với từ tiếng Việt.** Đừng thêm key là âm tiết tiếng Việt hợp lệ. Hai ca thật: `lan` xuất hiện 846 dòng nhưng là `lan đao` = "line down", không phải LAN mạng; `phi` 88 dòng là phiên âm "free" nhưng `\bphi\b` sẽ ăn cả `phi công`, `châu Phi` trong 2 triệu file public (và `▁phi` nằm trong 2000 token BPE, tức nó phổ biến). Luôn `grep` ví dụ thật trước khi thêm key ngắn.

## 🔜 HƯỚNG SẼ THỬ TIẾP: chỉnh `tokens.txt` (2000 dòng) theo 200h FPT rồi finetune lại như `B`

Ý tưởng: bảng BPE hiện tại đến từ `hynt/Zipformer-30M-RNNT-6000h`, tối ưu cho tiếng Việt phổ
thông. Dựng lại / chỉnh nó theo text của 200h FPT để thuật ngữ được cắt thành mảnh gọn hơn, rồi
chạy lại đúng công thức `B` (`TRAINABLE_MODULES=decoder`, lr 3e-5, ep36→45).

**Số đo hiện trạng** (tokenize toàn bộ text 200h + val bằng `lang_bpe_500_gip`):

| | |
|---|---|
| loại token thật sự xuất hiện trong 200h train | **627 / 2000** |
| loại token xuất hiện trong val | 693 / 2000 |
| **không xuất hiện ở cả hai** | **1.044** |
| xuất hiện ≤ 20 lần | 352 |
| ⇒ có thể thay mà không đụng ID đang dùng | **1.396 / 2000 = 69,8%** |
| slot `<unused_N>` sẵn có | **0** (bảng đầy) |

**Có headroom thật, đã đo**: BPE hiện tại bằm nát thuật ngữ hơn từ thường **47%**.

| | mảnh / âm tiết |
|---|---|
| từ thường (20.000 cut, đã bỏ cụm thuật ngữ) | **1,22** |
| 87 cụm thuật ngữ (379 mảnh / 211 âm tiết) | **1,80** |

Tệ nhất là mức ký tự: `pách két` → `▁p á ch ▁k é t` (7 mảnh / 2 âm tiết), `phây búc` →
`▁ph â y ▁b ú c`, `úp lốt` → `▁ ú p ▁l ố t`. Trong khi cụm may mắn đã gọn sẵn: `mô đem` →
`▁mô ▁đem`, `ca me ra` → `▁ca ▁me ▁ra` (1,00).

Vì sao số mảnh quan trọng — nối với probe âm học ở `results.md`: transducer phải **thắng blank ở
từng mảnh**, nên cụm 7 mảnh có 7 cơ hội rớt. `quai phai` = `▁qua i ▁pha i`, và đo được là mảnh
`▁qua` chỉ đạt p = 0,003. Cụm càng nhiều mảnh thì càng dễ tuột khỏi `ContextGraph` giữa chừng,
mà rời graph là **bị đòi lại toàn bộ bonus** (`context_graph.py:358`).

### Cách làm: thay BPE + train lại đúng 3 lớp gắn ID, tái dùng phần còn lại

Đây là cách gỡ được ràng buộc "đổi BPE là hỏng checkpoint": 3 lớp gắn vào token ID thì **train
lại**, 91% còn lại (encoder) độc lập với ID nên **giữ nguyên**.

| lớp | param | vì sao |
|---|---|---|
| `decoder.embedding` | 1.024.000 | tra theo token ID |
| `joiner.output_linear` | 1.026.000 | xuất ra 2000 chiều vocab |
| `ctc_output` | 1.026.000 | như trên |
| **tổng trainable** | **3.076.000 = 4,42%** | |
| `encoder` + `encoder_embed` + `joiner.{encoder,decoder}_proj` + `decoder.conv` | 64,3M | không dính ID ⇒ tái dùng |

```
TRAINABLE_MODULES=decoder.embedding,joiner.output_linear,ctc_output
```

✅ **KHÔNG cần extract lại fbank.** `train.py:1081` gọi `sp.encode(texts)` **lúc train**, đọc
text thẳng từ manifest — fbank chỉ chứa feature audio, không chứa token. Nên `SKIP_PREPROCESSING=true`
vẫn dùng được, tiết kiệm ~15h.

⚠️ **Khởi tạo 3 lớp đó thế nào mới là chỗ quyết định thành bại:**
- **Đừng random toàn bộ.** 3,08M param học lại từ đầu trên 200h TTS với encoder đóng băng — đó
  đúng là công thức đã đo là hỏng (`results.md`: train trên TTS ⇒ nhớ giọng tổng hợp, không
  chuyển sang thoại thật).
- **Nên transplant theo TÊN PIECE**: piece nào có trong cả bảng cũ và mới thì copy nguyên hàng
  (dù ID đổi); piece mới thì init bằng mean của bảng. Giữ lại phần lớn kiến thức 6.000h, chỉ
  phần thật sự mới mới phải học. Với 604 token đang dùng thì phần lớn sẽ khớp tên.

⚠️ **RỦI RO — và nó đúng là thứ hiện KHÔNG đo được.** 1.044 token "chết" chỉ chết *trong domain
FPT*: `▁những`, `▁nước`, `▁điều`, `▁xây`, `▁dựng`, `▁luật`, `▁pháp`, `▁bệnh`, `▁thương`… là từ
vựng tiếng Việt phổ thông mà `epoch-35.pt` học từ ~6.000h. Thay chúng = đổi năng lực nền lấy
thuật ngữ FPT. Val hiện 100% thoại FPT nên **không nhìn thấy cái mất đi**.

⇒ **Nên cắt val public trước** (vài trăm cut ViVoice + bud500, `results.md` hướng #2). Không có
nó thì run này ra con số recall đẹp hơn mà không ai biết đã đánh đổi cái gì. Nếu chỉ muốn thử
nhanh để xem có tín hiệu không thì chạy được ngay — nhưng đọc kết quả với đúng cảnh báo này.

**Trích xuất fbank** (`_worker_extract_shard` trong `prepare_data.py`) dùng `kaldifeat` trên GPU nếu import được, fallback về `torchaudio.compliance.kaldi.fbank` trên CPU — cả 2 đều config khớp convention của icefall (`high_freq=-400`). Cần giữ đồng bộ với cách feature extraction mà checkpoint bạn đang resume đã được train — trộn lẫn 2 kiểu feature (ví dụ MelSpectrogram cũ với Kaldi-style mới) giữa chừng sẽ không báo lỗi nhưng làm hại convergence vì phân phối input của encoder bị lệch.

**Lưu ý khi resume checkpoint**: `load_checkpoint_if_available` trong `train.py` (đã patch so với bản gốc icefall) chấp nhận checkpoint chỉ có key `model` (ví dụ checkpoint đã average qua `--avg` của icefall, không có `optimizer`/`scheduler`/`batch_idx_train`/...) bằng cách dùng default có sẵn trong `params` thay vì raise `KeyError`. Resume từ loại checkpoint này là **warm-start weight**, không phải resume thật — optimizer momentum và LR schedule sẽ bắt đầu lại từ đầu dù model weight vẫn được giữ.

## Data ở đâu, và checkpoint nào train trên data nào

Toàn bộ data nằm trên ổ cơ: `/mnt/hdd/shared/quanlm46/zipformer_data/ASR_dataset/`
(317G audio gốc + 2 × 72G fbank). Repo không chứa data.

### Audio gốc — `formated_dataset/`

```
formated_dataset/train/        9 subdir = 9 dataset (format prepare_data.py: **/*.wav + .txt cùng tên)
formated_dataset/val/audio/    + ground_truth_new.csv
```

**"Data 200h" = FPT TTS = `formated_dataset/train/clean` + `formated_dataset/train/noise`.**
Đây **không phải thoại call-center thật** — nó là TTS. Đo từ `train/metadata.csv` (176.401 dòng,
**200,0h**): source `fpt_s2t_100k` (100.000) + `fpt_s2t_100k_v2` (70.401) + `fpt_train_6000` (6.000),
**14 giọng TTS đặt tên** (`thanh_dat`, `mai_linh`, `ngoc_huyen`, …), 8.000 Hz. `clean` và `noise` là
cùng bộ câu — bản sạch (87.734 cut) và bản trộn nhiễu (88.667 cut). ⚠️ `metadata.csv` **chỉ mô tả
clean+noise**, 7 dataset public không có mặt trong đó.

**"Data full 2.340h"** = cả 9 subdir. Đo trực tiếp từ manifest đang dùng:

| dataset | giờ | cut | vai trò |
|---|---:|---:|---|
| ViVoice | 1.017,0 | 887.772 | public |
| viet_bud500 | 461,9 | 649.158 | public |
| infore2_audiobooks | 415,8 | 315.449 | public |
| VLSP2020 | 101,4 | 56.427 | public |
| LSVSC | 100,7 | 56.823 | public |
| **clean** | **100,1** | 87.734 | **FPT TTS** |
| **noise** | **100,1** | 88.667 | **FPT TTS** |
| FOSD | 30,2 | 25.917 | public |
| fleurs | 13,3 | 4.212 | public |
| **tổng** | **2.340,3** | **2.172.159** | FPT = **8,6%** thời lượng |

Val = **549 cut / 0,50h** thoại call-center thật, 11.025 Hz (`EVAL_MIN_DURATION=1.0` → còn 478).
Val là nguồn *duy nhất* trong project có thoại thật; mọi thứ trong train đều là TTS hoặc read speech.

### fbank

- **`fbank_data/`** (72G) — **đang dùng**. `librispeech_cuts_train-clean-100.jsonl.gz` = 2.172.159 cut
  / 2.340,3h, **đã shuffle**; `librispeech_cuts_dev-clean.jsonl.gz` = 549 cut. Full-band: không có
  `FEATURE_CONFIG.txt` nên `train.py` tự báo là không kiểm được, tức nó được extract **trước** khi có
  patch band-limit ⇒ `sr gốc → 16k` thẳng, không cắt băng.
- `fbank_data_save_no_normalize/` (72G) — snapshot **cùng 2.172.159 cut / 2.340,3h**, chỉ khác
  transcript (bản trước `normalize_text` mới). **Không phải** bản 200h.
- **`fbank_fpt200/`** (6,2G) — **200h FPT thuần**, extract 2026-08-11 từ `fpt200_dataset`.
  Val ở đây được extract lại cùng lượt nên dùng `word_mapping` MỚI (`ép pi ti`) → **khung đo
  khác** `fbank_data`, không so chéo được. Đây là fbank của mọi run 200h.
- `fbank_data_band3k/` — **rỗng**. `.env_band3k` trỏ vào đây nhưng run đó chưa từng chạy.

**`fpt200_dataset/` là cây HARD LINK** (`train/{clean,noise}` + `val/`) trỏ vào
`formated_dataset`, tốn ~0 byte. ⚠️ Phải là hard link, KHÔNG symlink: symlink trỏ ra ngoài
`DATA_DIR` không resolve trong container, và `prepare_data.py:690` dùng `p.is_dir()` nên
subdir bị **bỏ im lặng**, không lỗi.

## `train.py` log macro-WER, không phải WER chuẩn

`compute_validation_wer` trả *mean batch WER* (docstring của chính nó), tức trung bình WER từng
batch. WER chuẩn là micro — một phép chia `(S+D+I)/N` trên toàn corpus. Chênh thực đo 0,66–0,71 pp,
cùng cỡ với hiệu ứng đang truy. Macro còn **phụ thuộc `MAX_DURATION`** (đổi cách gom batch là đổi
số: cùng `epoch-35.pt` ra 32,25% ở log train, 32,17% ở dump với `--max-duration 250`, micro 32,88%).
Mọi so sánh chính thức dùng micro của `dump_val.py` / `eval_beam.py`; số trong log train chỉ để
đối chiếu nội bộ một run.

## Lệch phân phối feature giữa train và val (đo 2026-08-05)

Ba nguồn audio trong project này có đặc tính phổ **rất khác nhau**, và vì `prepare_data.py`
resample tất cả về 16 kHz rồi lấy fbank 80 mel bin phủ 0–7600 Hz, sự khác nhau đó đi thẳng
vào input của encoder.

Chỉ số dùng để đo (script tạm, không có trong repo): lấy trung bình mỗi mel bin qua mọi
frame của 1 file, rồi đếm số bin thấp hơn bin cao nhất **quá 6 dB**.

| nguồn | sample rate gốc | mel bin "yếu" |
|---|---|---|
| public (LSVSC, VLSP2020, fleurs, FOSD, infore2_audiobooks, viet_bud500, ViVoice) | 16k–48k | **0–1 / 80** |
| train FPT (`clean`, `noise` — TTS 14 giọng) | 8.000 Hz | 18–24 / 80 |
| **val (549 câu thoại call-center thật)** | **11.025 Hz** | **~35 / 80** |

Hai điều cần nhớ:

1. **Val là nguồn có phổ hẹp nhất, không phải rộng nhất.** Suy từ sample rate thì val
   (11.025 Hz) trông như rộng băng hơn train FPT (8.000 Hz), nhưng đo thật thì ngược lại:
   val là thoại điện thoại đã qua codec narrowband rồi transcode, còn FPT là TTS tổng hợp
   nên sạch và đầy đặn trong dải của nó. Đừng suy đặc tính phổ từ sample rate.

2. **Chỉ số này thô, đừng đọc quá.** Bin thứ 45/80 ứng với ~2,2 kHz, nên "35 bin yếu" nghĩa
   là yếu từ ~2,2 kHz trở lên — đó phần lớn là **độ dốc phổ tự nhiên của giọng người**, không
   phải cắt băng ở 5,5 kHz. Ngưỡng 6 dB là tùy ý và có tính cả frame im lặng. Muốn số cắt
   băng thật thì phải đo phổ FFT trực tiếp.

Kết luận vẫn đứng: phân phối feature đầu vào của val lệch nhiều so với data public mà
`epoch-35.pt` (gipformer gốc) đã học. Lệch này nằm ở tầng **acoustic**, tức trong **encoder** —
nên mọi cấu hình `FREEZE_ENCODER=true` / `TRAINABLE_MODULES=joiner` đều **không thể chạm tới
nó**. Đây là lý do có thể giải thích tại sao VALID WER dừng ở ~33% qua mọi lần finetune output
head, và tại sao trộn thêm data public full-band vào trong khi encoder đóng băng gần như chắc
chắn không giúp được gì cho val.

## Manifest xếp theo khối dataset — phải shuffle, nếu không mỗi batch chỉ có 1 dataset

`prepare_data.py` ghi cut theo thứ tự scan (`sorted(sub_dirs)` rồi `extend` lần lượt), nên manifest
là **các khối liền mạch theo dataset**. Còn `DynamicBucketingSampler` trong `asr_datamodule.py` dùng
`buffer_size = num_buckets * 5000` = **150.000 cut**, chỉ 6,9% của 2,17 triệu. Khối ViVoice một mình
đã 678.800 cut — gấp 4,5 lần cửa sổ. Kết quả: mỗi batch lấy từ gần như một dataset duy nhất.

Thành phần corpus 2.340h (đo 2026-08-05, lấy dataset từ `recording.sources[0].source`):

| dataset | cut | giờ | %thời lượng | dài TB |
|---|---|---|---|---|
| ViVoice | 887.772 | 1017,0 | 43,5% | 4,12s |
| viet_bud500 | 649.158 | 461,9 | 19,7% | 2,56s |
| infore2_audiobooks | 315.449 | 415,8 | 17,8% | 4,75s |
| noise + clean (**FPT, in-domain**) | 176.401 | 200,2 | **8,6%** | 4,1s |
| LSVSC + VLSP2020 | 113.250 | 202,1 | 8,6% | 6,4s |
| FOSD + fleurs | 30.129 | 43,5 | 1,9% | 4,2s / 11,4s |

**Hệ quả nguy hiểm: đường loss không đo việc học.** Với `TRAINABLE_MODULES=joiner`, head CTC bị đóng
băng nên WER CTC ra giống hệt tới 2 chữ số — vậy mà `ctc_loss` vẫn nhảy 0,73 → 0,11 → 0,71 → 0,22
trong một epoch. Các bậc thang đó rơi **đúng vào biên khối**: hai lần loss vọt lên (batch 8100 và
10800, kèm `grad_scale` sụp 32→16→8) trùng biên vào `infore2_audiobooks`. Đọc "loss giảm nên training
tốt" là sai — nó là bản đồ độ khó của từng dataset.

Cách sửa đã dùng: **shuffle chính manifest một lần** (`random.Random(seed).shuffle(lines)` trên jsonl,
verify multiset dòng không đổi rồi `os.replace` nguyên tử). Sau đó cửa sổ 150.000 cut chứa **9/9
dataset**, lệch tối đa 0,23% so với tỉ lệ toàn cục. Đổi `buffer_size` thì phải sửa `asr_datamodule.py`
nằm **trong image** (không được mount) nên phải rebuild — shuffle manifest rẻ hơn nhiều.

Cái giá: đọc `.lca` chuyển từ semi-sequential sang random trên cả 16 shard. Ước lượng 2,17M cut /
9.360s một epoch = 232 seek/s, trong khi ổ chỉ cho ~180 IOPS lúc extraction. **Batch trộn đều ⟺ đọc
đĩa random**, không có cách nào vừa trộn vừa giữ tuần tự.

Và "mỗi batch có đủ 9 dataset" là **không đạt được**: batch = 600s ≈ 155 cut, `fleurs` chiếm 0,19% số
cut → 0,3 cut/batch, tức chỉ ~26% batch có nó. Muốn mọi batch đều có thì phải oversample, tức không
còn "đúng tỉ lệ".

## Eden LR dùng số epoch TUYỆT ĐỐI — `BASE_LR` không phải lr bạn nhận

```
lr = BASE_LR × ((batch² + LR_BATCHES²)/LR_BATCHES²)^-0.25
             × ((epoch² + LR_EPOCHS²)/LR_EPOCHS²)^-0.25 × warmup
```

`epoch` là số epoch **tuyệt đối**, không phải số epoch đã train trong run này. Finetune từ epoch 36
với `LR_EPOCHS=3` cho hệ số epoch = **0,2882**, tức `BASE_LR` bị **chia ngầm cho 3,47**:
`BASE_LR=0.000015` → lr thật **4,3e-6**. Đã kiểm khớp với log trong 2% từ batch 600 trở đi.

`LR_EPOCHS` là epoch mà hệ số đó tụt xuống `2^-0.25 = 0,841`. Với `LE=3` thì mốc đó là epoch 3, nên ở
epoch 36 ta đã ở sâu trong đuôi phẳng: 0,2882 (e36) → 0,2735 (e40), chỉ **-5,1% trên cả run**. Tức
`LR_EPOCHS` **không còn là scheduler**, nó chỉ là hằng số chia. Nó mới có tác dụng thật nếu train xa
hơn (epoch 60: 0,2235 = -22%).

Toàn bộ phần giảm lr đến từ `LR_BATCHES` qua bộ đếm batch **tích luỹ**. Lưu ý bộ đếm này reset về 0
khi resume từ checkpoint đã average (như `epoch-35.pt`, chỉ có key `model`) nhưng **tiếp tục** khi
resume từ checkpoint do `save_checkpoint` ghi (như `epoch-36.pt`, có `batch_idx_train`).

Với 5 epoch = 72.250 batch, `LR_BATCHES=7500` làm lr tụt **68%** (cuối/đầu = 0,34). Muốn lr phẳng để
dò lr mà không bị decay làm nhiễu phép đo thì cần `LR_BATCHES ≈ 300.000` (tụt 1,4%). Nhưng phẳng là
cấu hình **thăm dò**, không phải để train thật — chốt được lr rồi thì trả decay về.

Cảnh báo: số batch/epoch phụ thuộc `MAX_DURATION`, nên **đổi `MAX_DURATION` là đổi luôn lịch lr**.
72.250 ở trên tính theo `MAX_DURATION=600`; ở `250` (bắt buộc khi full finetune, xem section 08-06)
một epoch là **35.600 batch** → 5 epoch = 178.000 batch, và `LR_BATCHES=300.000` tụt **7,2%** chứ
không phải 1,4%.

## Gotcha: `TEST_MODE` + `MAX_DURATION` làm dataloader trả 0 batch

Triệu chứng: `UnboundLocalError: local variable 'batch_idx' referenced before assignment` ở
`train_one_epoch` (dòng `params.num_batches_per_epoch = batch_idx + 1`). Message này che mất
nguyên nhân — nó chỉ có nghĩa là vòng `for batch_idx, batch in enumerate(train_dl)` **không
chạy lần nào**.

Nguyên nhân: `DynamicBucketingSampler` chia cut thành `num_buckets=30` nhóm theo độ dài và gom
batch **trong từng bucket riêng biệt**, còn `--drop-last` default là **`True`** (icefall
`asr_datamodule.py`). Nếu không bucket nào tích đủ `MAX_DURATION` giây thì mọi batch đều dở dang
và bị bỏ hết.

Điều kiện cần: **tổng thời lượng tập train ≥ `num_buckets` × `MAX_DURATION`**. Với
`MAX_DURATION=600` thì cần ≥ 30 × 600 = 18.000s = **5 giờ** audio, bất kể có bao nhiêu file.
`TEST_SAMPLES=3000` (≈2,9h) vẫn fail. Khi test nhanh thì hạ `MAX_DURATION=100` (cần 50 phút
audio) rồi trả về `600` cho run thật.

Kèm theo: `TEST_SAMPLES` áp cho **cả `train` và `val`**, nên nó ghi đè luôn manifest val bằng
một subset nhỏ. Vì vậy WER đo trong lúc `TEST_MODE=true` **không so sánh được** với WER của run
thật (đo trên 50 câu thay vì 549 làm VALID WER dịch tới ~3 điểm dù model không đổi một bit).
Và `prepare_data.py` **ghi đè `fbank_data/` tại chỗ** — chạy `TEST_MODE` là mất manifest của
dataset đầy đủ, phải extract lại.

## Training bị chặn bởi ổ cơ, không phải GPU (đo 2026-08-06, lúc epoch 37 đang chạy)

```
GPU0: 40 mẫu / 40s → util trung bình 23,0%, power 100W / 360W
sda : %util 96,02 | 280 r/s | 9,0 MB/s | r_await 6,27ms | aqu-sz 1,76
```

`fbank_data` 72 GB nằm trên `/dev/sda` (HDD 5,5T). Ổ bão hoà, **GPU rảnh 77% thời gian**. Epoch =
**5h57m** cho 35.600 batch (0,55–0,59 s/batch). Đây đúng là cái giá đã dự đoán ở section shuffle:
batch trộn đều ⟺ đọc `.lca` random.

Muốn sửa thì chuyển `fbank_data` sang NVMe: `/` còn 100 GB / 916 GB — vừa đủ 72 GB nhưng chỉ để lại
28 GB trên server dùng chung, nên prune docker build cache (~26 GB) trước. Phải sửa mount trong
`run.sh` và restart training, nên chỉ đáng làm nếu đã dừng vì lý do khác.

## Log của icefall ghi giờ UTC, không phải giờ host

Cả nội dung `callbacks/log/log-train-*` và tên file đều dùng giờ trong container (**UTC**), còn
`ls`/`stat` trên host là **+07**. Dòng cuối `07:57:57` với `mtime 14:57:57` là **cùng một thời điểm** —
không phải log đứng 7 tiếng. Kiểm nhanh bằng `docker inspect --format '{{.State.StartedAt}}'` (cũng
UTC) so với tên file log.

## `compute_validation_wer` không trả model về `train()`

`compute_validation_wer` gọi `model.eval()` và **không bao giờ gọi lại `model.train()`**. Ở cuối epoch
thì vô hại vì `train_one_epoch` tự set `model.train()` ở đầu epoch sau. Nhưng nếu thêm lời gọi
`log_wer` **vào giữa vòng batch** thì model sẽ kẹt ở eval mode suốt phần còn lại của epoch — tắt
dropout, dùng running stats — và training bị hỏng âm thầm. Phải tự gọi `model.train()` ngay sau đó.

Ghi chú thêm: `compute_validation_wer` có `max_batches=10` ở signature, nhưng `log_wer` truyền
`max_batches=None` nên thực tế nó quét **hết** dataloader. Đó là lý do TRAIN WER trên 1.000 cut mất
61s còn VALID trên 549 cut chỉ mất 2s (val vừa được đọc cho validation loss nên feature còn nóng).
Với `TRAIN_EVAL_CUTS=10000` thì tính TRAIN WER mất khoảng 10 phút mỗi epoch.

## Filter cut khi train (`remove_short_and_long_utt`)

Đã nới so với icefall gốc (1,0–20,0s) thành **0,1–30,0s và bỏ câu ≤ 2 từ**. Đo trên 2.172.159 cut:

```
1,0–20,0s              → giữ 2.132.973 (98,20%)
0,1–30,0s + từ > 2     → giữ 2.154.622 (99,19%)
loại: từ ≤ 2 = 17.267 (0,79%) | duration > 30s = 270 | duration < 0,1s = 0
nhận thêm: vùng 0,1–1s = 37.886 cut | vùng 20–30s = 1.030 cut
```

Biên 0,1s chủ yếu để không cấm oan — chốt thật cho cut quá ngắn vẫn là ràng buộc `T >= S` ngay bên
dưới (`(T-2) < len(tokens)` → loại), nên cut 0,4s mà 5 từ vẫn không vào được. Lọc `từ ≤ 2` là thay
đổi có tác động lớn nhất trong ba cái, và nó loại đúng loại câu làm WER nhiễu nhất.

Nới lên 30s làm peak memory mỗi batch tăng (bucket 30s có ~20 cut thay vì ~30 cut ở bucket 20s, tổng
giây không đổi nên về lý thuyết không OOM). Nếu run sau thấy `Grad scale is small` xuất hiện dày hơn
thì đây là nghi phạm đầu tiên.
