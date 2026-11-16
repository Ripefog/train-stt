# results.md — kết quả đo của các run finetune

Mọi số đo train/eval nằm ở đây. `CLAUDE.md` chỉ giữ cách repo hoạt động và các bẫy kỹ thuật.

**Điểm khởi đầu của mọi run:** `callbacks/epoch-35.pt` = **gipformer gốc**, checkpoint tiếng
Việt tốt nhất đang có, do bên thứ ba train. Data gốc train ra nó (public + call-center, số
giờ không công bố) **không có và không tái tạo được**. Mọi thứ dưới đây là *finetune từ nó*.

**Điều kiện đo chung** (trừ khi ghi khác): val = **478 cut** (549 lọc `duration >= 1.0s`),
**6.687 từ** ref, **145 lần xuất hiện thuật ngữ** (87 cụm nhiều âm tiết trong
`word_mapping.json`). WER là **micro** (`(S+D+I)/N` toàn corpus). Bootstrap là **paired**,
5.000 lần, cùng tập cut.

⚠️ **Bootstrap paired phải join theo `id`, KHÔNG theo thứ tự dòng.** Thứ tự cut trong dump của
`eval_beam.py` **không ổn định giữa các lần chạy**: hai lần đo cách nhau chỉ trùng vị trí
**7/478**. Ghép theo dòng làm CI phình từ ±0,4 lên ±20 pp (thuật ngữ) / ±3 pp (WER) — trông như
"mọi thứ đều trong nhiễu". Đã kiểm: `valbp1_ep35` và `valbp1_B_ep40` may cùng thứ tự 478/478 nên
các số bootstrap ghi trước 08-13 không bị ảnh hưởng.

### Hai cách tính recall thuật ngữ — đọc số nào

| | định nghĩa | trả lời câu hỏi |
|---|---|---|
| **exact** | hyp phải ra ĐÚNG chuỗi chuẩn của `word_mapping.json` | model tự học được dạng chuẩn chưa |
| **equiv** | hyp ra BẤT KỲ biến thể `map_safe` cùng canonical cũng tính đúng | khách nhận được text đúng tên không |

## 🔻 QUYẾT ĐỊNH 2026-08-13: **chỉ đo `exact`**

Từ nay **`exact` là cột duy nhất dùng để xếp hạng checkpoint.** `equiv` ngừng dùng.

Lý do không phải vì `equiv` sai về mặt sản phẩm — nó vẫn đúng rằng prod map biến thể về canonical
(`transcribe.py:220 _load_hotword_map()` đọc bảng MySQL `hotword_entries`, nên `ép bi ti` và
`ép pi ti` cho ra cùng kết quả cuối). Lý do là **`equiv` không phải một phép đo ổn định**: nó phụ
thuộc hai thứ nằm ngoài model — nội dung bảng `hotword_entries` và tham số `min_variant_len` — và
trong một ngày nó đã **lật dấu 3 kết luận** khi tham số đó đổi từ 4 sang 0 rồi ngược lại. `exact`
so hyp với đúng chuỗi trong `word_mapping.json`, không có tham số nào để chỉnh, nên hai lần đo
cách nhau vài tuần vẫn so được với nhau.

Cái giá phải chấp nhận, ghi ra để không quên: `exact` **thấp hơn thứ khách thực nhận ~9–12 pp**
(`B_ep40 beam16+hw+bp1`: exact 45,5% trong khi sau map của prod là 57,2%). Nên đừng báo số
`exact` ra ngoài như "recall thuật ngữ của sản phẩm" — nó là **sàn dưới**, dùng để **so
checkpoint với nhau**.

Kiểm chứng trước khi đổi: không kết luận nào trong file này đảo khi chuyển sang `exact`. Mọi
ứng viên vẫn hoà với `B_ep40` (CI ôm 0 ở cả hai chế độ): `dense_ep45` +1,4 [−1,4, +4,0],
`emb_ep41` +0,7 [−1,9, +4,1], `dproj_ep38` +0,7 [−2,9, +4,4], `C_deep_ep45` +0,7 [−4,9, +6,3],
`avg-36-45` +0,0, `decay_ep41` +0,0. Và `B_ep40 − ep35` vẫn thật: **+7,6 [+3,6, +11,9] ✅**.

⚠️ Các bảng bên dưới còn giữ cột `equiv` như **số liệu lịch sử** — đọc cột `exact`. Mọi số
`equiv` trong file này ở thang `min_variant_len=4`; nếu gặp số cũ (66,9% cho `B_ep40 +bp1`,
64,1% cho `beam16+hw`) thì đó là thang `0`, không so được với gì cả.

⚠️ **HAI KHUNG ĐO, không so chéo được.** Val được extract lại 2026-08-11 với
`word_mapping.json` mới (`fpt` → `ép pi ti`), làm đổi reference của 37/549 cut:

| khung | fbank | ep35 micro-WER | thuật ngữ |
|---|---|---|---|
| cũ (≤ 08-10) | `fbank_data` | 32,88% | /108 |
| **mới (≥ 08-11)** | `fbank_fpt200` | **32,85%** | **/145** |

---

## ✅ Cấu hình tốt nhất hiện tại

**`callbacks_fpt200_B/epoch-40.pt` + beam 16 + hotwords + `blank_penalty=1`**

### Train
| | |
|---|---|
| base | `callbacks/epoch-35.pt` (gipformer gốc) |
| **data** | **200h FPT** — `fpt200_dataset/train/{clean,noise}`, fbank `fbank_fpt200` |
| | ⚠️ là **TTS 14 giọng 8 kHz**, không phải thoại thật; clean+noise cùng bộ câu → text độc nhất chỉ ~100h |
| `TRAINABLE_MODULES` | **`decoder`** = 1.028.096 param (**1,48%**) — encoder, joiner, `ctc_output` đóng băng |
| `BASE_LR` | **3e-5** (`LR_BATCHES=300000`, `LR_EPOCHS=200` → lr phẳng) |
| `MAX_DURATION` | 600 → **1.200 batch/epoch** |
| epoch | `START_EPOCH=36`, `NUM_EPOCHS=45` (10 epoch) → **lấy epoch 40** |
| khác | `USE_TRANSDUCER/USE_CTC=true`, `BAND_LIMIT_SR=0`, `BAND_CLAMP_BIN=0` |
| chạy bằng | `./run_fpt200_B.sh` ⟵ `.env_fpt200_B`, GPU 0, ~4–5 phút/epoch |

Xác minh freeze: **`VALID ctc = 33,18%` giống hệt cả 11 lần đo** (EVAL_FIRST + epoch 36–45).

### Decode
`modified_beam_search`, `beam=16`, `--hotwords hotwords.txt` (cụm wifi ở `:3`),
`--blank-penalty 1`. Lệnh: `scripts/eval_beam.py` (val) hoặc `scripts/infer_wav.py` (audio rời).

### Số đo
| | WER | CER | thuật ngữ *exact* | thuật ngữ *equiv* |
|---|---|---|---|---|
| gốc `ep35` greedy | 32,85% | 27,42% | 33,8% (49/145) | 38,6% (56/145) |
| **`B_ep40` beam16+hw+bp1** | **32,53%** | 26,93% | **45,5% (66/145)** | **57,2% (83/145)** |
| `B_ep40` beam16+hw (bp0) | **31,79%** | **26,27%** | 44,1% (64/145) | 54,5% (79/145) |

`equiv` là con số đúng với sản phẩm giao cho khách (xem *Hai cách tính recall* ở đầu file).
`CER` bỏ khoảng trắng, và nó **xếp hạng y hệt WER** ở mọi cấu hình đã đo — chỉ dùng để xác nhận
không config nào ăn gian WER bằng cách đoán ký tự.

`bp1` đổi **+0,74 pp WER** lấy **+1,4 pp thuật ngữ** so với `bp0`. Chọn theo ưu tiên:
**bp0 nếu ưu tiên WER, bp1 nếu ưu tiên thuật ngữ**. `bp2` bị loại — cùng 66/145 như bp1
nhưng WER +2,26 pp.

### Bảng xếp hạng cuối (2026-08-13) — `exact` @ `beam16+hw+bp1`, KÈM cột kiểm chứng audio dài

**Cột "160s" là cột lọc, không phải cột phụ.** Xếp hạng chỉ theo val sẽ đưa 6 checkpoint hỏng
lên đầu bảng.

| hạng | ckpt | exact | WER | #từ `beam16` / 160s | |
|---|---|---|---|---|---|
| 1 | dproj_ep45 | 57,9% | 35,70% | **21** | ❌ sụp |
| 2 | dproj_ep43 | 57,9% | 34,20% | **73** | ❌ sụp |
| 3 | dproj_ep42 | 55,2% | 33,17% | **178** | ❌ sụp |
| 4 | dproj_ep41 | 53,8% | 32,93% | **178** | ❌ sụp |
| 5 | dproj_ep40 | 52,4% | 32,27% | **242** | ❌ sụp |
| 6 | dproj_ep39 | 50,3% | 32,32% | **449** | ❌ rạn |
| 7 | dense_ep45 | 46,9% | 32,53% | *chưa đo* | ⚠️ |
| 8 | dproj_ep38 | 46,2% | 32,38% | 523 | ✅ |
| 8 | emb_ep41 | 46,2% | **32,24%** | **535** | ✅ |
| 8 | C_deep_ep45 | 46,2% | 32,57% | *chưa đo* | ⚠️ |
| 8 | h5_ep45 | 46,2% | 32,72% | **547** | ✅ |
| 12 | **B_ep40** | 45,5% | 32,53% | **545** | ✅ |
| 12 | avg-36-45 | 45,5% | 32,44% | 546 | ✅ |
| 14 | bitfit_ep45 | 42,8% | 32,96% | 523 | ✅ |
| — | *ep35 gốc* | *37,9%* | *32,23%* | *536–608* | ✅ |

Trong nhóm đã kiểm chứng, bốn bản đầu chênh nhau **1 lần xuất hiện trên 145** — bootstrap ôm 0.
`h5_ep45` (đo 2026-08-13) leo tới đây bằng **0,094% param** nhưng bootstrap vs `B_ep40` ôm 0 ở
cả 3 chế độ, nên nó **hoà**, không phải thắng. `bitfit_ep45` là bản duy nhất ngoài `ep35`
**thua có ý nghĩa** ở `exact`.
`emb_ep41` (đo 2026-08-13, ngay trước khi xoá) **bằng hoặc hơn `dproj_ep38` ở mọi trục**: cùng
67/145, WER thấp hơn 0,14 pp, giữ nhiều hơn 12 từ trên audio dài, cùng bắt 3 `quai phai` +
1 `quay phai` ở `beam16+hw`. ⇒ Không có lý do đo được để xếp `dproj` trên `emb` hay trên `B`.

### Checkpoint còn trên đĩa (2026-08-13)

| | |
|---|---|
| `callbacks/epoch-35.pt` | 269M — **gipformer gốc, KHÔNG BAO GIỜ xoá.** Bên thứ ba train, data không công bố ⇒ **không tái tạo được**. Là seed của cả 12 run và là baseline `ep35` trong mọi bảng. |
| `callbacks_fpt200_B/` | 6,9G, 14 file — bản đang dùng |
| `callbacks_fpt200_B_dproj/epoch-38.pt` | 553M |
| `callbacks_fpt200_h5/` | 6,4G, 11 file — **xoá được** (hướng đóng, tái tạo từ `.env_fpt200_h5`) |
| `callbacks_fpt200_bitfit/` | 12G, 13 file — **xoá được** (hướng đóng, tái tạo từ `.env_fpt200_bitfit`) |

Đã xoá `.pt` của `conv`, `decay`, `emb`, `dense`, `dproj` (11/12), `C_deep`, `C_front` — tất cả
đều **tái tạo được** từ `.env` tương ứng + `epoch-35.pt`, ~50 phút/run. `log/` + `tensorboard/`
và 15 file dump `jsonl` giữ nguyên, nên mọi số trong file này tính lại được không cần GPU.

---

## Bảng val đầy đủ (2026-08-12) — 3 checkpoint × 3 chế độ × 3 mức bp

WER / recall thuật ngữ.

| | bp0 | bp1 | bp2 |
|---|---|---|---|
| **ep35** greedy | 32,85 / 33,8 | 32,62 / 33,8 | 33,41 / 34,5 |
| **ep35** beam16 | 32,21 / 33,8 | 32,59 / 34,5 | 33,42 / 34,5 |
| **ep35** beam16+hw | **31,94** / 36,6 | 32,23 / 37,9 | 33,11 / 40,0 |
| **B_ep40** greedy | 32,62 / 33,8 | 32,77 / 35,2 | 33,95 / 35,2 |
| **B_ep40** beam16 | 32,30 / 35,2 | 32,88 / 35,9 | 34,29 / 36,6 |
| **B_ep40** beam16+hw | **31,79 / 44,1** | **32,53 / 45,5** | 34,05 / 45,5 |
| **B_ep45** greedy | 33,78 / 33,1 | 33,32 / 34,5 | 34,23 / 35,2 |
| **B_ep45** beam16 | 32,99 / 33,1 | 33,53 / 35,2 | 34,77 / 37,2 |
| **B_ep45** beam16+hw | 32,47 / 42,1 | 33,03 / 44,8 | 34,23 / 46,2 |

## Ba tầng đóng góp vào hai trục KHÁC NHAU (paired bootstrap)

| tầng | ΔWER | Δ thuật ngữ |
|---|---|---|
| beam16 | **−0,64** [−1,13, −0,15] ✅ | **+0,0** [+0,0, +0,0] ✗ |
| + hotwords | **−0,27** [−0,54, −0,03] ✅ | **+2,8** [+0,6, +5,7] ✅ |
| + finetune (`decoder`, 200h FPT) | −0,15 [−0,65, +0,35] ✗ | **+7,6** [+3,6, +12,0] ✅ |
| **tổng vs `ep35` greedy** | **−1,06** [−1,73, −0,40] ✅ | **+10,3** [+5,6, +15,6] ✅ |
| *(thêm bp2 thay bp0)* | +2,26 [+1,44, +3,18] ✅ hại | +1,4 [+0,0, +3,5] ✗ |
| *(ep45 thay ep40)* | +0,67 [−0,10, +1,61] ✗ | −2,1 [−7,2, +2,5] ✗ |

**Đây là kết quả cốt lõi.** beam lấy gần hết WER nhưng **đúng bằng không** cho thuật ngữ.
Finetune thì ngược: **không đo được gì trên WER** nhưng **+7,6 pp thuật ngữ** — đóng góp lớn
nhất. ⇒ **Đánh giá hướng domain-term bằng WER một mình sẽ kết luận SAI là finetune vô ích.**

## Sweep lr trên `decoder`-only (2026-08-12) — trần thuật ngữ ~45% ĐỘC LẬP với lr

Giả thuyết đem đi kiểm: *lr cao hơn → trôi mạnh hơn về domain FPT → nhiều thuật ngữ hơn,
dù WER xấu đi*. **Bị phủ định.**

Cùng `decoder`-only 1,5%, cùng 200h FPT, cùng 10 epoch, chỉ đổi `BASE_LR`:

| lr | VALID trans qua 10 epoch | beam16+hw WER / thuật ngữ | +bp1 |
|---|---|---|---|
| **3e-5** | 32,78 → đáy **32,62** (ep40) → 33,78 | **31,79% / 44,1%** | **32,53% / 45,5%** |
| 1,5e-4 (×5) | 32,77 → sụp → 43,70 | 32,00% / 43,4% | 32,72% / 44,1% |
| 6e-4 (×20) | **37,36 ngay ep36** → 66,11 | 35,32% / 43,4% | 35,31% / 44,8% |

**`B_ep40` (3e-5) thắng CẢ HAI cột so với mọi checkpoint lr cao hơn** — không ô nào của
lr5x/lr20x vượt được nó ở bất kỳ chỉ số nào. Thuật ngữ **bão hoà 43–45% dù lr chênh 20 lần**,
trong khi WER xấu đi đơn điệu theo lr.

⇒ **Trần ~45% KHÔNG do decoder trôi chưa đủ.** Đẩy decoder mạnh hơn chỉ phá WER. Ghép với
hai kết quả trước — beam cho **đúng +0,0** thuật ngữ, hotwords chỉ +2,8 — thì ~55% thuật ngữ
còn lại **không nằm trong tầm với của tầng ngôn ngữ**: chúng thuộc phía **âm học** (model
không nghe ra âm) hoặc do **danh sách hotword thiếu biến thể** (ví dụ vị trí `bộ con quai phụ`
chưa cụm nào trong `hotwords.txt` khớp được).

⇒ Fork này **đã phân giải bằng phép đo trực tiếp: chủ yếu là âm học** — xem
*Probe âm học (2026-08-12)* bên dưới.

**`BASE_LR=3e-5` là điểm tối ưu, sweep đóng lại.** Biên an toàn nằm giữa 3e-5 và 1,5e-4.

### Train THÊM epoch ở 3e-5 KHÔNG giúp — cả hai chỉ số đều xấu đi sau ep40

| ep | 36 | 37 | 38 | 39 | **40** | 41 | 42 | 43 | 44 | 45 |
|---|---|---|---|---|---|---|---|---|---|---|
| WER | 32,78 | 32,82 | 32,79 | 32,69 | **32,62** | 32,87 | 33,00 | 33,26 | 33,42 | 33,78 |

Sau ep40 tăng **đơn điệu**, +1,16 pp trong 5 epoch (~0,23 pp/epoch và đang tăng dần). Thuật
ngữ cũng giảm: **ep40 44,1% → ep45 42,1%**. Nên chạy 20/30 epoch chỉ làm xấu cả hai.

Và **tăng độ phân giải cũng vô ích**: ep36–ep40 chỉ trải 0,16 pp (32,78 → 32,62), **dưới**
ngưỡng phân giải paired ~0,45 pp. "Đáy ở ep40" không phân biệt được với ep36–39 về mặt thống
kê — thêm checkpoint giữa epoch là đo nhiễu.

## Probe âm học (2026-08-12) — trần ~45% là ÂM HỌC, không phải danh sách hotword

Fork còn để ngỏ ở mục trên (~55% sai là do model không nghe ra âm, HAY do `hotwords.txt`
thiếu biến thể) **đã phân giải: chủ yếu là âm học.**

Phép đo — `scripts/probe_acoustic.py`, **không train, không decode lại**: căn ref vào audio
bằng **Viterbi trên lattice RNN-T đầy đủ** (không prune), rồi tại đúng ô `(t,u)` phát ra token
của thuật ngữ đọc hai thứ độc lập nhau:
- **phân phối joiner** đầy đủ trên vocab → `gap` = `logP(token tốt nhất khác blank) − logP(token ĐÚNG)`, cộng trên mọi token của cụm. Đây là số nats cần bù để cụm thắng.
- **posterior head CTC** trong cửa sổ ±N frame quanh frame đã căn. CTC **chưa từng được
  finetune** ở bất kỳ run nào và không nhìn ngữ cảnh nhãn ⇒ đây là bằng chứng **âm học thuần**.

Chạy trên `B_ep40`, mode `beam16+hw`, **cả 145 lần xuất hiện, 0 bỏ sót**.
Dump: `dump_val/probe_{B_ep40,B_ep40_w8,ep35}.jsonl`.

Hai kiểm định trước khi đọc số: (1) 79/145 hit khớp **đúng** con số tương đương 54,5% trong
`CLAUDE.md`; (2) nhóm đối chứng — các lần thuật ngữ ĐÚNG — có gap trung vị **0,00**.

| | MISS (n=66) | HIT đối chứng (n=79) |
|---|---|---|
| gap toàn cụm (nats) | 6,12 | **0,00** |
| rank token đúng ≤ 10 | 43/66 | 74/79 |
| **CTC logP, cửa sổ ±2 frame** | **−8,39** | **−0,18** |
| CTC logP, nới ±8 frame (±320 ms) | **−7,44** | −0,18 |

**Dòng CTC là bằng chứng quyết định: uniform trên 2.000 token = −7,60.** Ở vị trí sai, head
âm học nằm **ngang mức ngẫu nhiên**, và nới cửa sổ căn frame gấp 4 lần không cứu được (−8,39 →
−7,44, vẫn ≈ uniform) — trong khi ở vị trí đúng nó là −0,18 ≈ 84% xác suất. Cách nhau ~7,3 nats.

Token đúng vẫn đứng rank 3–10 trong joiner là **do decoder prior kéo lên**, không phải do bằng
chứng âm học. Ví dụ `quai phai` (đúng vị trí đã nêu ở mục averaging):

```
▁qua  rank 4,  gap  6,1   ctc −10,1
i     rank 1,  gap  0,0   ctc −14,6
▁pha  rank 56, gap 10,5   ctc −17,1   ← không có gì ở đây cả
i     rank 1,  gap  0,0   ctc −14,6
```

### Finetune đã đóng phần ngôn ngữ, phần còn lại không thuộc về nó

Ghép cặp `ep35 → B_ep40` trên cùng vị trí:

| | ep35 | B_ep40 |
|---|---|---|
| Δgap trung vị | | **−0,46 nats** (64 tốt lên / 10 xấu đi / 54 không đổi) |
| rank trung vị | 6 | **3** |
| chuyển trạng thái | | **12 sai→đúng, 0 đúng→sai** |
| 65 vị trí sai ở CẢ HAI (lõi cứng) | gap 9,30 | → 5,90, CTC **−8,66** |

Finetune đóng gần một nửa khoảng cách phía **ngôn ngữ** và không phá vị trí nào — đúng dấu với
+7,6 pp thuật ngữ đo bằng bootstrap. Phần còn lại là 65 vị trí có CTC **dưới** mức ngẫu nhiên.

### Boost hotword lật được — nhưng bằng cách đè lên âm học không tồn tại

`context_score` là tham số **DECODE**, không có trong train (không gradient, checkpoint không
lưu gì về nó) ⇒ mọi số dưới đây đổi được không cần train lại. `hotwords.txt` thực tế chỉ có
**hai mức**: 15 dòng `:3` (nhóm ÉP …TÊ + QUAI/QUAY/GOAI/WAI PHAI), 94 dòng còn lại ở 2,0
(36 dòng ghi `:2`, 58 dòng không ghi `:N` nên rơi về `--context-score` mặc định **2,0**).

Đếm số vị trí sai có `gap < ngân sách boost` (= score × số token):

| ngân sách | lật được |
|---|---|
| 3,0 đều | 52/66 |
| 2,0 đều | 43/66 |
| **THẬT theo `hotwords.txt`** | **37/48** — 18 vị trí còn lại không có dòng nào cho dạng chuẩn |

Nhưng 18 vị trí "thiếu trong danh sách" đó (`ô kê` ×10, `ô tê pê` ×3, `bốn gờ` ×2, `diu túp`,
`banh kinh`, `gờ i xê`) có **CTC(±8) trung vị −7,49 — y hệt nhóm đã có trong danh sách (−7,57)**.

⇒ **Thêm biến thể vào `hotwords.txt` không cứu được chúng.** Boost lật được vị trí không phải
vì làm lộ ra bằng chứng âm học, mà vì **cộng thẳng điểm đè lên chỗ không có bằng chứng nào** —
nên cái giá là dương tính giả ở nơi khác. Đây chính là lý do hotwords chỉ cho +2,8 pp, và lý do
mọi lần vặn `bp`/boost đều đổi WER lấy thuật ngữ.

⚠️ **Ba giới hạn của phép đo này:**
1. Nó **không** phân biệt "model không nghe ra" với "**audio thật sự không chứa cụm đó**"
   (ref sai, hoặc người nói phát âm khác hẳn). `ô kê` sai 10/66 rất đáng nghi thuộc loại thứ
   hai. Cần nghe tay ~10 mẫu để chốt — **chưa làm**.
2. Bảng ghép cặp `ep35↔ep40` khoá theo `(cut, thuật ngữ)` nên gộp các lần xuất hiện trùng
   trong cùng một cut: **128/145** vị trí, không phải toàn bộ.
3. `gap` đo cục bộ tại một ô `(t,u)` dưới ràng buộc đi theo ref; nó **không** bằng chi phí thật
   mà beam search phải trả để giữ cụm đó sống qua cả path.

## Averaging checkpoint (2026-08-12) — hoà, nhưng bền hơn trên audio dài

`icefall.checkpoint.average_checkpoints` trên `callbacks_fpt200_B/`:

| | beam16+hw | +bp1 |
|---|---|---|
| B_ep40 (đơn lẻ) | **31,79% / 44,1%** | 32,53% / 45,5% |
| avg-36-40 | 31,91% / **42,1%** ↓ | 32,56% / 42,8% ↓ |
| **avg-36-45** | 31,91% / 44,1% | **32,44%** / 45,5% |

Val **không chọn được** giữa `B_ep40` và `avg-36-45` (chênh ≤0,12 pp, dưới ngưỡng 0,45 pp).
Nghịch lý đáng ghi: average **10** epoch (gồm cả ep41–45 đã xấu) **tốt hơn** average 5 epoch
tốt — averaging rộng regularize mạnh hơn.

Phân biệt được bằng **audio dài 160s** (phép đã bắt lỗi ep45):

| | #từ | cụm wifi | lặp | vị trí `bộ con quai ___ phụ` |
|---|---:|---:|---:|---|
| B_ep40 beam16+hw | 548 | 6 | 0 | `bộ con quai phụ` ✗ |
| **avg-36-45 beam16+hw** | 546 | 6 | 0 | **`bộ con quai phai phụ`** ✅ |

`avg-36-45` là bản **duy nhất** chạm được vị trí thứ ba — chỗ mà mọi checkpoint và mọi mức
bp/điểm hotword khác đều sai. ⚠️ Cỡ mẫu = 1 file, không có ground truth ⇒ tín hiệu, chưa
phải bằng chứng. Với `avg-36-45` thì **bp=0 đúng** (bp1 làm rớt vị trí này).

## Lợi ích thuật ngữ nằm ở EMBEDDING, không ở conv (2026-08-12)

Hai run song song từ `epoch-35.pt`, đều decoder-side, 10 epoch, 200h FPT:
- **decay**: `TRAINABLE_MODULES=decoder`, `LR_BATCHES=3000` → lr 3e-5 giảm còn 1,45e-5
- **conv**: `TRAINABLE_MODULES=decoder.conv` → **4.096 param (0,006% model)**.
  Prefix khớp được nhờ `train.py:1723`, không sửa code.

Phân rã decoder: `embedding.weight` **1.024.000 (99,6%)** + `conv.weight` **4.096** (shape
`[512, 4, 2]`, kernel 2 = `context_size=2` → trigram trên token BPE).

| checkpoint | param | beam16+hw WER / thuật ngữ | +bp1 |
|---|---:|---|---|
| *ep35 (gốc, chưa finetune)* | — | *31,94% / 36,6%* | *32,23% / 37,9%* |
| **B_ep40** (decoder, flat) | 1.028.096 | **31,79% / 44,1%** | 32,53% / 45,5% |
| decay_ep41 | 1.028.096 | 31,84% / 43,4% | 32,60% / 45,5% |
| **conv_ep38** | **4.096** | 32,24% / **37,9%** | 32,75% / 40,7% |
| conv_ep45 | 4.096 | 32,39% / 41,4% | 33,17% / 42,8% |
| emb_ep39 (embedding một mình) | 1.024.000 | 31,96% / 40,0% | 32,35% / 42,1% |
| **emb_ep41** (embedding một mình) | 1.024.000 | 32,11% / 42,8% | 32,24% / **46,2%** |

**`conv` chỉ mang lại ~17% lợi ích thuật ngữ** (36,6 → 37,9 = +1,3 pp, so với +7,5 pp của
decoder đầy đủ). ⇒ **83% lợi ích đến từ bảng embedding 1.024.000 param.** Kiến thức thuật ngữ
KHÔNG phải quy tắc chuỗi mà `conv` học, mà là **dịch biểu diễn từng token** trong cụm — hợp lý
vì trigram 2 token không nhìn hết `quai phai` (BPE: `▁qua i ▁pha i` = 4 token).

Và nó giải thích nghịch lý: `conv_ep38` **bằng đáy WER** (32,60% greedy) nhưng thuật ngữ kém xa
⇒ **WER và thuật ngữ đến từ hai nguồn khác nhau ngay trong cùng module.**

### Kiểm chứng trực tiếp: chạy `decoder.embedding` MỘT MÌNH (2026-08-12) — 83% xác nhận, nhưng KHÔNG thay được B

Con số 83% ở trên là **suy ra** (100% − 17% của conv). Run `.env_fpt200_B_emb`
(`TRAINABLE_MODULES=decoder.embedding` = 1.024.000 param, bỏ đúng 4.096 param `conv` so với B,
mọi thứ còn lại y hệt B) đo thẳng nó. `VALID ctc = 33,18%` đứng im cả 10 epoch ⇒ freeze sạch.

| ep | 36 | 37 | 38 | **39** | 40 | 41 | 42 | 43 | 44 | 45 | trôi từ đáy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **emb** (1,02M) | 32,78 | 32,63 | 32,68 | **32,54** | 32,56 | 32,57 | 32,88 | 32,99 | 33,14 | 33,23 | **+0,69** |
| B (1,03M) | 32,78 | 32,82 | 32,79 | 32,69 | **32,62** | 32,87 | 33,00 | 33,26 | 33,42 | 33,78 | +1,16 |

**Trên thang khớp chính xác, 83% được xác nhận đúng y:** `(42,8 − 36,6) / (44,1 − 36,6)` =
**82,7%**. Nhưng trên thang **tương đương** (thang đúng với prod) thì recall như sau:

| | beam16+hw WER / thuật ngữ *equiv* | +bp1 |
|---|---|---|
| ep35 | 31,94% / 45,5% | 32,23% / 49,7% |
| **B_ep40** | **31,79% / 54,5%** | 32,53% / **57,2%** |
| emb_ep39 | 31,96% / 50,3% | 32,35% / 52,4% |
| emb_ep40 | 32,02% / 51,7% | **32,20%** / 55,9% |
| emb_ep41 | 32,11% / 52,4% | 32,24% / 55,2% |
| emb_ep45 | 32,30% / 51,7% | 32,51% / 53,1% |
| conv_ep38 | 32,24% / 47,6% | 32,75% / 53,1% |

Paired bootstrap 10.000, cùng 478 cut, **join theo `id`**:

| so sánh | ΔWER | Δ thuật ngữ (equiv) |
|---|---|---|
| emb_ep41 − ep35 (beam16+hw) | +0,17 [−0,20, +0,55] ✗ | **+6,9 [+2,8, +11,5]** ✅ |
| **emb_ep41 − B_ep40** (beam16+hw) | +0,31 [−0,05, +0,67] ✗ | −2,1 [−5,8, +1,4] ✗ |
| emb_ep41 − B_ep40 (+bp1) | −0,28 [−0,66, +0,08] ✗ | −2,1 [−5,4, +0,7] ✗ |
| emb_ep45 − B_ep40 (+bp1) | −0,02 [−0,65, +0,72] ✗ | **−4,1 [−8,2, −0,8]** ✅ xấu hơn |
| conv_ep38 − B_ep40 (beam16+hw) | +0,45 [−0,02, +0,96] ✗ | **−6,9 [−11,9, −2,3]** ✅ xấu hơn |

⇒ **`emb_ep41` không phân biệt được với `B_ep40`** ở cả hai chế độ (80/145 vs 83/145 ở `+bp1`,
CI ôm 0). ⇒ **Không có lý do thay `B_ep40`, cũng không có lý do coi `emb` là kém hơn.** Hai bản
coi như thay thế được cho nhau; `emb` **ổn định hơn** (trôi +0,69 thay vì +1,16 pp) nên nếu phải
chọn epoch cuối mà không đo tay thì `emb` an toàn hơn. Nhưng `emb_ep45` **thua thật** (−4,1 pp,
CI không ôm 0) ⇒ lợi thế "ổn định" đó không kéo dài tới cuối run.

⚠️ **Ngày 08-12 tôi đã ghi đè hai kết luận này thành "không có ý nghĩa thống kê"; đo lại trên
thang đúng thì CẢ HAI kết luận GỐC đứng vững, nguyên số:**
- *"embedding một mình cho lợi ích thuật ngữ THẬT so với ep35"* → **+6,9 [+2,8, +11,5] ✅**.
  Bản 08-12 ghi `+2,8 [−0,8, +6,7] ✗`, đó là số của thang `min_variant_len=0`.
- `emb_ep45 − B_ep40 (+bp1)` → **−4,1 [−8,2, −0,8] ✅ xấu hơn**. Bản 08-12 ghi `−3,4 [−7,8, +0,0] ✗`.

Bài học: **đổi `min_variant_len` là đổi định nghĩa của metric**, và ở đây nó lật dấu kết luận
theo cả hai chiều. Mọi số `equiv` phải ghi kèm thang.

### ⚠️ 2026-08-13: trả cột `equiv` về `min_variant_len=4` — thang 0 là dương tính giả

Ngày 08-12 tôi đã đổi cả file sang `min_variant_len=0` với lý do "tập biến thể lúc đó thiếu
`ép`/`ếp`". **Đó là bước lùi, nay đã trả lại.** 18 biến thể chỉ tồn tại ở thang 0 —
`ép`, `ếp`, `fpt`, `gic`, `ok`, `pro`, `ios`, `otp`, `qos`, `4g`, `5g`, `ghz`, `dbm`… — và
riêng `ép` sinh ra toàn bộ chênh lệch. Liệt kê thẳng 14 cut mà thang 0 tính đúng còn thang 4
tính sai (`B_ep40 +bp1`), **13/14 là do chuỗi `ép` đứng một mình**:

| ref | hyp | thang 0 chấm |
|---|---|---|
| …bên **ép pi ti** á mình báo quai phai… | mình **ép** mình báo quai phai… | ✅ đúng |
| …mình có đi **ép pi ti** lại mấy lần rồi… | …có bị **ép** lại mấy lần rồi… | ✅ đúng |
| …nhà mạng **ép pi ti** ạ… | …nhà mạng **ép tám** ạ… | ✅ đúng |
| …em bên **ép pi ti** nhá | …em bên **ép min ép i đi** nha | ✅ đúng |
| …của **ép pi ti** ở khu vực mình… | …của **ép** ở nơi khu vực mình… | ✅ đúng |

`ép` là **động từ tiếng Việt thường dùng** — "bị ép lại mấy lần" không hề chứa tên thương hiệu.
Đây đúng là ca mà docstring `metrics.py` cảnh báo. Nếu prod thật sự để `ép → FPT` ở `map_safe=1`
thì đó là **bug của bảng `hotword_entries`**, không phải lý do để thổi con số recall lên.

Công bằng mà nói: **1/14 ca là hụt thật** (`ép pi tê`, `ép i tí` — biến thể đọc hợp lệ mà bảng
biến thể dài ≥ 4 ký tự chưa phủ). Cách sửa đúng cho những ca đó là **thêm biến thể dài vào
`hotword_variants.tsv`**, không phải hạ `min_variant_len` xuống 0.

Hệ quả: mọi số `equiv` trong file này thấp hơn bản 08-12 khoảng **10÷14 pp**, và **một kết luận
bị lật** — xem điểm 2 mục ENCODER (`C_front` không chứng minh được là phá kiến thức thuật ngữ).

### Sàn 32,6% là thật — 5 cấu hình khác nhau đều dừng ở đó

| run | phạm vi | lr | đáy | ep45 | trôi từ đáy |
|---|---|---|---|---|---|
| B (flat) | decoder 1,03M | 3e-5 phẳng | ep40 **32,62** | 33,78 | +1,16 pp |
| decay | decoder 1,03M | 3e-5 → 1,45e-5 | ep41 **32,60** | 33,06 | **+0,46 pp** |
| conv | decoder.conv 4,1K | 3e-5 phẳng | ep38 **32,60** | **32,85** | **+0,25 pp** |
| emb | decoder.embedding 1,02M | 3e-5 phẳng | ep39 **32,54** | 33,23 | +0,69 pp |
| **C_deep** | **encoder.encoders.3 30,34M** | 1e-5 phẳng | ep38 **32,57** | 32,87 | **+0,30 pp** |

Năm đáy trùng trong **0,08 pp** ⇒ ~32,6% (greedy) / 31,8% (beam16+hw) là **sàn thật**, không phải
điểm may. Và nó **không còn là "sàn của hướng decoder"**: `C_deep` mở 30,34M param **bên trong
encoder**, gấp ba mươi lần `B`, vẫn dừng đúng chỗ đó. ⇒ Sàn này là **sàn của data**, không phải
của phạm vi param. Khác biệt giữa các run chỉ ở **độ ổn định**:
- `decay` làm đúng việc: giảm trôi từ +1,16 xuống +0,46 pp. Chậm một nửa, **không chặn được**.
- `conv` gần như không trôi nổi: ep45 = 32,85% = **đúng baseline**. 4.096 param tự giới hạn.
- `emb` (bỏ đúng 4.096 param `conv` khỏi B): trôi +0,69 thay vì +1,16 pp, và **thấp hơn B ở cả
  10 epoch** — chênh nằm dưới ngưỡng nhiễu, và trên trục thuật ngữ nó **hoà** với B (không
  "mất" như từng ghi — xem phần sửa lại ở mục `emb` bên trên).
- `C_deep` trôi chỉ +0,30 pp dù mở 30,34M param — vì lr 1e-5, thấp hơn ba run kia ba lần.

⇒ `decay` không cho đáy tốt hơn, nó chỉ khiến epoch cuối dùng được mà không phải chọn tay.

## Hướng ENCODER: đã đo, đã đóng (2026-08-12)

Hai run mở thẳng vào encoder — thứ mà mọi cấu hình trước đều đóng băng. Cùng 200h FPT, cùng
`MAX_DURATION=600`, lr **1e-5** (thấp hơn 3e-5 của hướng decoder vì encoder ở dưới, mọi dịch
chuyển lan qua toàn bộ phần còn lại).

| run | `TRAINABLE_MODULES` | param | % model |
|---|---|---:|---:|
| **C_front** | `encoder_embed,encoder.encoders.0,encoder.encoders.1` | 6.496.023 | 9,3% |
| **C_deep** | `encoder.encoders.3` | 30.343.052 | **43,6%** |

VALID transducer WER qua 10 epoch (`START_EPOCH=36`, baseline ep35 = 32,85%):

| ep | 36 | 37 | 38 | 39 | 40 | 41 | 42 | 43 | 44 | 45 |
|---|---|---|---|---|---|---|---|---|---|---|
| **C_deep** | 32,85 | 32,71 | **32,57** | 32,77 | 32,85 | 33,02 | 32,99 | 32,93 | 32,91 | 32,87 |
| **C_front** | 33,45 | 33,65 | 33,71 | 33,63 | 33,80 | 33,81 | 33,92 | 34,23 | 34,44 | **34,72** |

`C_front` **xấu hơn baseline ngay từ epoch đầu và tăng đơn điệu, không có đáy**. Trong khi đó WER
trên tập train tụt **8,27% → 4,42%** và CTC xấu đi **33,18% → 35,04%**. Đó là overfit vào TTS ở
dạng sách vở: học rất tốt cách nghe của 14 giọng tổng hợp, mất dần khả năng nghe thoại thật.
`C_deep` cũng đổi CTC (33,18 → 33,83) ⇒ cả hai run **thật sự** đã mở encoder, không phải lỗi
khai báo scope.

### So với `B_ep40` — không ai vượt, một cái hại thật

⚠️ **Bảng dưới đã tính lại ngày 2026-08-13 ở `min_variant_len=4`** (mặc định của `metrics.py`,
cũng là thang của CLAUDE.md và của mọi mục khác trong file này). Bản trước dùng `0`, tức nhận cả
biến thể 2 ký tự `"ép"`/`"ếp"` — đúng thứ docstring `metrics.py` gọi là dương tính giả vì chúng
khớp bừa vào âm tiết Việt thường. Thang lỏng đó thổi cột `equiv` lên **+10÷14 pp** và, quan
trọng hơn, **đổi hẳn một kết luận** (xem điểm 2). Số cũ: ep35 59,3 / B_ep40 64,1 / emb 62,1 /
C_deep 62,8 / C_front 54,5.

| checkpoint | scope | param | WER | CER | exact | **equiv** |
|---|---|---:|---|---|---|---|
| *ep35* | — | — | *31,94%* | *26,53%* | *36,6%* | *45,5%* |
| **B_ep40** | decoder | 1.028.096 | **31,79%** | **26,27%** | **44,1%** | **54,5%** |
| emb_ep41 | decoder.embedding | 1.024.000 | 32,11% | 26,62% | 42,8% | 53,1% |
| C_deep_ep38 | encoder.encoders.3 | 30.343.052 | 32,24% | 26,74% | 39,3% | 50,3% |
| C_front_ep45 | encoder_embed + stack 0,1 | 6.496.023 | 33,93% | 28,37% | 35,9% | 44,8% |

*(chế độ `beam16+hw`; ở `+bp1` thứ tự không đổi: B_ep40 57,2% / emb 57,2% / C_deep 51,7% /
C_front 49,7%)*

Paired bootstrap 10.000, join theo `id`, 478 cut, cùng thang `min_variant_len=4`:

| so sánh | chế độ | ΔWER | Δ thuật ngữ (equiv) |
|---|---|---|---|
| B_ep40 − ep35 | beam16+hw | −0,15 [−0,62, +0,34] ✗ | **+8,9 [+3,8, +14,9]** ✅ |
| C_deep_ep38 − ep35 | beam16+hw | +0,30 [−0,17, +0,81] ✗ | **+4,8 [+1,8, +8,3]** ✅ |
| C_deep_ep38 − B_ep40 | beam16+hw | +0,45 [−0,13, +1,07] ✗ | −4,2 [−8,9, +0,6] ✗ |
| C_deep_ep38 − B_ep40 | +bp1 | −0,08 [−0,60, +0,45] ✗ | **−5,5 [−9,9, −1,5]** ✅ hại |
| C_front_ep45 − ep35 | beam16+hw | **+1,98 [+1,15, +2,82]** ✅ hại | −0,7 [−5,1, +3,1] ✗ |
| C_front_ep45 − B_ep40 | beam16+hw | **+2,13 [+1,18, +3,08]** ✅ hại | **−9,6 [−14,9, −5,0]** ✅ hại |
| C_front_ep45 − B_ep40 | +bp1 | +0,61 [−0,38, +1,57] ✗ | **−7,6 [−12,8, −2,7]** ✅ hại |

**Ba kết luận:**

1. **`C_deep` hoà với `B_ep40` ở cả hai trục** dù mở `30.343.052` param — gấp **30 lần** `B`.
   Nó có lợi ích thuật ngữ thật so với ep35 (+4,8 pp, CI không ôm 0) nhưng không hơn được một
   run 1M param, và ở `+bp1` còn thua `B` có ý nghĩa (−5,5 pp). ⇒ **Nút thắt không phải thiếu
   dung lượng để học.**

2. **`C_front` phá WER có ý nghĩa thống kê (+1,98 pp so ep35), nhưng KHÔNG chứng minh được là
   nó phá kiến thức thuật ngữ.** ⚠️ Đây là chỗ thang cũ dẫn tới kết luận sai: ở
   `min_variant_len=0` thì `C_front_ep45 − ep35` ra −4,8 pp với CI không ôm 0, và file này từng
   viết "phá luôn kiến thức thuật ngữ đã có". Tính lại ở thang đúng thì Δ = **−0,7 [−5,1, +3,1]**,
   CI ôm 0 — tức phần "mất" đó nằm gần hết trong các biến thể 2 ký tự khớp bừa. Cái đo được chắc
   chắn là: `C_front` **không lấy được** khoản +8,9 pp mà `B` lấy được (−9,6 so với `B`), và nó
   **làm hỏng WER**. Kết luận thực tế không đổi — train acoustic trên TTS vẫn là hại — nhưng lý
   do là hại WER, không phải xoá thuật ngữ.

3. `C_front_ep45` là epoch tệ nhất của run đó, nhưng epoch tốt nhất (`33,45%` ở ep36) cũng đã
   xấu hơn baseline và đường val tăng đơn điệu suốt 10 epoch ⇒ **không checkpoint nào cứu được**.

4. ⚠️ **Khuyến nghị kiến trúc trước đó đã bị chính thí nghiệm này bác bỏ.** Tôi từng xếp front
   end (`encoder_embed` + stack 0,1) là hướng số 1 và khuyên **không** thử `encoders.3`, lập
   luận rằng frame ở stack sâu dài 160 ms nên "dài hơn một âm tiết". Lập luận đó sai: stack 3 ăn
   **output đã mã hoá** của stack 0–2 chứ không ăn 160 ms sóng thô, nên độ dài frame không nói
   gì về việc nó biểu diễn cái gì. Run kiểm chứng (do người dùng yêu cầu chạy đối chứng) cho kết
   quả **ngược hẳn**: front end xấu đi đơn điệu, `encoders.3` phẳng.

⚠️ Hạn chế: `emb` chỉ còn `epoch-41.pt` trên đĩa (mất `epoch-39.pt` là đáy, lệch 0,03 pp nên
`ep41` vẫn đại diện được), `C_front` chỉ còn `epoch-45.pt`. `C_deep` còn đủ ep36–45.

## Ghi nhớ đọc log: `val-lớn` trong log train KHÔNG đo generalization (2026-08-13)

Log train của mọi run 200h in **hai** tập valid xen kẽ nhau, phân biệt bằng số từ:

| tên trong log | số từ | thực chất |
|---|---:|---|
| `val-lớn` | 151.274 | **30% split của chính data TTS đang train** (`SPLIT_DATASETS` 70/30) |
| `FPT-val` | 6.687 | **tập giữ riêng thật** — 478 cut thoại call-center, là tập mọi bảng ở trên dùng |

Tách hai đường ra cho `C_deep` / `C_front` (transducer WER):

| ep | C_deep `val-lớn` | C_deep `FPT-val` | C_front `val-lớn` | C_front `FPT-val` |
|---|---|---|---|---|
| 35 | 8,27 | 32,85 | 8,27 | 32,85 |
| 38 | 5,35 | 32,57 | 7,68 | 33,71 |
| 41 | 3,38 | 33,02 | 6,17 | 33,81 |
| 45 | **2,39** | **32,87** | **4,42** | **34,72** |

`C_deep` **nuốt trọn tập TTS** — 8,27% → 2,39%, giảm 3,5 lần trong 10 epoch — trong khi
`FPT-val` **đứng im tuyệt đối** (32,85 → 32,87). Hai hệ quả:

1. **Encoder HOÀN TOÀN học được ở lr 1e-5.** Nếu ai đó nhìn `FPT-val` phẳng rồi kết luận "lr
   quá thấp, model không học gì" thì sai — nó vừa chứng minh ngược lại trên chính tập kia. Cái
   nó học được **không chuyển sang thoại thật**, vì trong data không có thoại thật để học.
2. **`val-lớn` là chỉ số vô dụng cho mọi run finetune trên FPT200** và rất dễ đọc nhầm thành
   "training tiến bộ tốt". Chỉ đọc `FPT-val` (6.687 từ).

⇒ Đây là lý do sâu hơn cả cho việc mọi hướng chọn module đều dừng ở ~32,6%: **không phải chọn
sai module, mà là data không chứa thứ cần học.** 20h thoại thật đáng giá hơn 200h TTS này.

## `B_dense` — null result sạch: trần ep40 không phải do lr, cũng không do thiếu data (2026-08-13)

`callbacks_fpt200_B_dense` nối **thẳng từ `B_ep40`** (`START_EPOCH=41`, lr 1e-5) trên
`fbank_fpt200_dense` — thêm ~15% dữ liệu so với B. Nếu "train quá ep40 thì xấu đi" là do lr còn
cao hoặc do hết data thì run này phải khá hơn.

⚠️ `callbacks_fpt200_B_dense/epoch-40.pt` **md5 trùng khít** `callbacks_fpt200_B/epoch-40.pt`
(`e770f5587ad7b145cfacda3a76db4472`) — nó là **seed**, không phải kết quả. Đừng chấm nó rồi
tưởng là một điểm dữ liệu.

| ckpt | beam16+hw exact / equiv / WER | +bp1 exact / equiv / WER |
|---|---|---|
| **B_ep40** | 44,1% / **54,5%** / **31,79%** | 45,5% / 57,2% / 32,53% |
| dense_ep41 | 43,4% / 53,1% / 31,87% | 45,5% / **57,2%** / 32,48% |
| dense_ep42 | 43,4% / 53,1% / 31,94% | 45,5% / **57,2%** / 32,54% |

Ở `+bp1`, `dense_ep41` và `dense_ep42` ra **đúng 66/145 exact và 83/145 equiv** — trùng từng con
số với `B_ep40`, WER lệch 0,05 pp. Bootstrap: ΔWER `−0,04 [−0,21, +0,10] ✗`, Δequiv
**`+0,0 [+0,0, +0,0]`** — không một cut nào đổi.

⇒ **Trần ở ep40 là trần thật, không phải bài toán lr hay khối lượng data.** Đây là bằng chứng
mạnh nhất hiện có cho "sàn 32,6%".

## `B_dproj` — recall thuật ngữ tăng là GIẢ, mua bằng cách nuốt nội dung (2026-08-13)

`.env_fpt200_B_dproj` khác `.env_fpt200_B` **đúng một dòng**: `TRAINABLE_MODULES=decoder,joiner.decoder_proj`
(1.290.752 param thay vì 1.028.096). Cùng seed `epoch-35.pt`, cùng lr 3e-5, cùng data, cùng
ep36→45 ⇒ so ep-với-ep là A/B sạch.

Ghép đường val với phép thử audio dài 160s (`audio_test/call_16k.wav`):

| ep | equiv `+bp1` | WER | #từ `beam16` | #từ `beam16+hw` |
|---|---|---|---|---|
| *B_ep40 (đối chứng)* | *57,2%* | *32,53%* | *545* | *548* |
| dproj 36 | 55,2% | 32,85% | 541 | 558 |
| dproj 37 | 56,6% | 32,32% | 537 | 538 |
| **dproj 38** | **57,9%** | 32,38% | 523 | 530 |
| dproj 39 | 60,0% | 32,32% | **449** ↓ | **439** ↓ |
| dproj 40 | 59,3% | 32,27% | **242** ↓↓ | 379 |
| dproj 41 | 60,0% | 32,93% | **178** | 263 |
| dproj 42 | 60,0% | 33,17% | **178** | 300 |
| dproj 43 | 62,8% | 34,20% | **73** | 135 |
| dproj 45 | **62,8%** | 35,70% | **21** ↓↓↓ | 131 |

**Hai đường trùng khít.** Recall chỉ vượt `B_ep40` từ ep39 — đúng epoch model bắt đầu nuốt nội
dung. Ở mọi epoch còn an toàn (36–38), bootstrap so `B_ep40` ôm 0 ở cả hai trục:

| so sánh (`+bp1`) | ΔWER | Δequiv |
|---|---|---|
| dproj_ep37 − B_ep40 | −0,21 [−0,72, +0,28] ✗ | −0,7 [−4,4, +2,9] ✗ |
| dproj_ep38 − B_ep40 | −0,15 [−0,73, +0,40] ✗ | +0,7 [−2,3, +3,9] ✗ |
| dproj_ep39 − B_ep40 | −0,21 [−0,84, +0,40] ✗ | +2,8 [−1,4, +7,1] ✗ |
| dproj_ep38 − ep35 | +0,15 [−0,40, +0,71] ✗ | **+8,3 [+3,0, +14,3]** ✅ |

⇒ **Thêm `joiner.decoder_proj` không mua được gì đo được.** `dproj_ep38` là bản duy nhất vừa
nhỉnh trên val vừa chưa sụp, nhưng không phân biệt được với `B_ep40`, và trên audio dài còn kém
hơn chút (530 từ / 3 lần `quai phai` so với 548 / 5).

### Xác minh phạm vi freeze bằng diff trọng số (2026-08-13) — không suy từ `.env`

Diff từng tensor với `epoch-35.pt` (745 tensor). Đây là cách kiểm `TRAINABLE_MODULES` **thật sự**
có hiệu lực hay không, thay vì tin vào log:

| run | tensor đổi | param | dòng trong `architecture.txt` |
|---|---|---|---|
| **B** | **2**/745 | 1.028.096 (1,48%) | 1950/**2696** `decoder.embedding.weight`, 1952/**2697** `decoder.conv.weight` |
| **dproj** | **4**/745 | 1.290.752 (1,85%) | như B + 1956/**2700** `joiner.decoder_proj.weight`, 1956/**2701** `.bias` |

*(dòng thứ nhất = cây module; dòng in đậm = bảng liệt kê từng tensor cuối file)*

743 tensor còn lại **y hệt tới từng bit**, gồm `encoder` (dòng 67, 63.382.150 param),
`encoder_embed` (35), `joiner.encoder_proj` (1955), `joiner.output_linear` (1957),
`ctc_output` (1960), `simple_am_proj` (1958), `simple_lm_proj` (1959). ⇒ xác nhận độc lập cho
`VALID ctc = 33,18%` đứng im suốt 11 lần đo.

Biên độ thay đổi (`|Δ|max` so ep35):

| tensor | B_ep40 | dproj_ep38 |
|---|---|---|
| `decoder.embedding.weight` | **1,90e−01** | **1,00e−01** |
| `decoder.conv.weight` | 6,11e−02 | 2,77e−02 |
| `joiner.decoder_proj.weight` | — | 2,34e−02 |

⚠️ `dproj` **phân bổ lại** lượng thay đổi chứ không cộng thêm: có `decoder_proj` để xoay thì
`embedding` chỉ phải dịch **một nửa** so với B, cùng 10 epoch cùng lr. Đây là lý giải cơ học cho
việc hai run ra kết quả gần như nhau dù `dproj` mở thêm 25% param.

### ⚠️ Quy tắc mới: recall TĂNG kèm WER TĂNG ⇒ phải kiểm audio dài trước khi tin

`dproj_ep45` có `equiv = 62,8%` — **cao nhất từng đo trong toàn bộ project** — mà là con số rác.
Nó không nhận ra thuật ngữ nhiều hơn; nó **nói ít đi** rồi để hotword chèn cụm vào chỗ trống.
Bằng chứng: `beam16` trên 160s còn **21/548 từ**, nhưng `beam16+hw` vẫn giữ 131 từ và 5 lần
`quai phai` — tức phần sống sót gần như chỉ còn hotword.

Val **không phát hiện được** chế độ này vì cut val chỉ 1–20s, quá ngắn để phần "nói ít đi" lộ ra.
Đây là **lần thứ hai** val bỏ lọt đúng chế độ này (lần đầu: `B_ep45`, mục dưới). ⇒ Bất kỳ
checkpoint nào có `Δrecall > 0` **và** `ΔWER > 0` đều phải chạy `infer_wav.py` trên
`audio_test/call_16k.wav` và đếm số từ trước khi được đưa vào bảng xếp hạng.

## Chữ `quai` (Wi-Fi) trượt vì ÂM HỌC, không phải vì conf thấp (2026-08-13)

Dùng `scripts/greedy_detail.py` (viết mới trong session này — `greedy_search_batch` trả **logit
thô** chứ không phải xác suất, nên không so được giữa các frame; script này decode lại đúng vòng
lặp đó nhưng giữ softmax đầy đủ) trên `audio_test/seg_120_130.wav`, 248 frame encoder × 40 ms:

| mảnh BPE | ep35 p đỉnh | B_ep40 p đỉnh | số frame p > 1% |
|---|---|---|---|
| `▁qua` (của **quai**) | 0,0029 (hạng 3/2000) | 0,0035 | **0/248** ở cả hai |
| `▁quay` | 0,0094 | 0,0098 | **0/248** ở cả hai |
| `▁pha` / `phai` | 0,403 → **0,767** | ↑ rõ | nhiều |

Model **không** phân vân giữa `quai` và cái nó chọn — ở những frame đó nó phân vân giữa
`▁pha` (0,403) và `▁sai` (0,292). `quai` nằm ngoài vùng xác suất đáng kể, `p_blank` = 0,988.

Finetune 200h **có** sửa được nửa sau: `phai` sắc lại rõ (`▁phụ` 0,548→0,767, `▁sáu`
0,424→0,838), nhưng **không nhích được nửa đầu một chút nào** (0,0029 → 0,0035).

Giải thích: `quai phai` là phiên âm Việt của **Wi-Fi** /waɪ faɪ/. Âm /faɪ/ có chỗ đứng trong âm
vị học tiếng Việt nên học được; **/waɪ/ thì không**, và 200h TTS không tạo ra nổi xác suất đo
được cho nó.

⇒ Hai hệ quả thực dụng:
1. **Hotword phải để ở dạng CỤM 2 tiếng** (`QUAI PHAI`), không phải một tiếng. Âm tiết sau vững
   về mặt âm học nên nó **kéo** âm tiết trước đi qua; mà `ContextGraph` lại **đòi lại toàn bộ
   bonus** khi beam rời graph giữa chừng (`context_graph.py:358`, `score = -state.node_score`),
   nên khớp lẻ "quai" ròng đúng bằng **0**.
2. Đây là lý do `context_score` cho nhóm wifi phải để `:3` chứ không dùng mặc định `2.0`
   (`hotwords.txt` dòng 14–17). Ngân sách boost = score × số token BPE = 3 × 4 = **12,0**.

## `B_ep45` loại — val không thấy, audio dài mới thấy

Trên val, ep45 không phân biệt được với ep40 (cả hai CI ôm 0). Nhưng trên file 160s:

| config | #từ output |
|---|---:|
| ep35 / B_ep40 (mọi chế độ) | 536–608 |
| **B_ep45 beam16** | **182** |
| B_ep45 beam16+hw | 327 |

Mất 2/3 nội dung. Val chỉ có cut ngắn (1–20s) nên **không phát hiện được sụp trên audio dài**
— cần test file dài trước khi chốt bất kỳ checkpoint nào.

---

## Hướng 5 — `decoder` + mask 120 dòng embedding (2026-08-13): **null, và ở epoch tốt nhất là ÂM**

Cấu hình: `TRAINABLE_MODULES=decoder` + `EMB_TERM_MASK=/app/word_mapping.json` ⇒ chỉ mở đúng
**120 hàng** của `decoder.embedding` thuộc 87 cụm thuật ngữ, cộng `decoder.conv.weight`.
Trainable thực = **65.536 param = 0,094%** — hẹp nhất trong mọi run của project.

### Cơ chế đã xác minh (không suy từ `.env`)

```
dong embedding bi doi : 120/2000       dong duoc phep doi : 120
doi NGOAI danh sach   : 0              |d|max dong khoa   : 0.000e+00
tensor khac bi doi    : ['decoder.conv.weight']
```

Bit-exact sau các bước `ScaledAdam` thật. Đây là chỗ cần thiết vì `batched_params`
(`optim.py:48`) xếp `decoder.embedding.weight` (2000,512) thành (1,2000,512) nên `param_rms` là
**một scalar duy nhất** — `optim.py:236 delta.add_(p * scale_step)` đẩy **mọi** hàng kể cả hàng
grad = 0. ⇒ **mask gradient một mình là KHÔNG đủ**, phải phục hồi hàng đóng băng *sau*
`optimizer.step()`.

Bằng chứng độc lập thứ hai: **`CTC WER = 33,18%` giống nhau tới 2 chữ số ở cả 10 epoch** (36→45).
Encoder và `ctc_output` đứng tuyệt đối.

### Val: đường đẹp nhất trong mọi run — và nó lừa

| ep | 36 | 37 | 38 | 39 | 40 | 41 | 42 | 43 | 44 | 45 |
|---|---|---|---|---|---|---|---|---|---|---|
| transducer | 32,71 | 32,71 | **32,62** | **32,60** | 32,66 | 32,72 | 32,69 | 32,69 | 32,72 | 32,72 |
| ctc | 33,18 | 33,18 | 33,18 | 33,18 | 33,18 | 33,18 | 33,18 | 33,18 | 33,18 | 33,18 |

Đáy val ở **ep38–39**, mà recall `exact` ở đúng hai epoch đó lại **thấp nhất**:

| ckpt | `beam16+hw` | `beam16+hw+bp1` | WER (hw) |
|---|---|---|---|
| h5_ep38 | 61/145 = 42,1% | 62/145 = 42,8% | 32,05% |
| h5_ep39 | 62/145 = 42,8% | 64/145 = 44,1% | 31,99% |
| h5_ep45 | 64/145 = 44,1% | **67/145 = 46,2%** | 31,87% |
| **B_ep40** | **64/145 = 44,1%** | 66/145 = 45,5% | — |

### Bootstrap vs `B_ep40` (10.000 resample, per-cut)

| | ΔWER | Δexact |
|---|---|---|
| h5_ep38 − B_ep40 `bp1` | +0,22 [−0,17,+0,62] ✗ | **−2,8 [−5,6,−0,6] ✅ thua thật** |
| h5_ep39 − B_ep40 `bp1` | +0,10 [−0,27,+0,49] ✗ | −1,4 [−3,6,+0,0] ✗ |
| h5_ep45 − B_ep40 `bp1` | +0,19 [−0,20,+0,58] ✗ | +0,7 [−1,6,+3,0] ✗ |
| h5_ep38 − B_ep40 `hw` | +0,25 ✗ | −2,0 [−5,2,+0,7] ✗ |
| h5_ep39 − B_ep40 `hw` | +0,19 ✗ | −1,4 [−4,7,+1,9] ✗ |
| h5_ep45 − B_ep40 `hw` | +0,07 ✗ | −0,0 [−3,4,+3,2] ✗ |
| h5_ep38/39/45 − B_ep40 `beam16` | +0,12 / +0,06 / −0,03 ✗ | −0,0 / −0,0 / +2,0 ✗ |

**8/9 ô ôm 0, ô thứ 9 là h5_ep38 thua có ý nghĩa.** Audio dài 160s không sụp
(`beam16` = 547 từ, `B_ep40` = 545) nên đây là null thật, không phải null che sụp.

⇒ **Hướng 5 đóng.** Nó không có đường nào để tốt hơn `B`: 65K param là **tập con** của
1,03M param mà `B` đã train, cùng lr, cùng data. Bó hẹp hơn chỉ mua được ít phương sai hơn,
không mua được kiến thức nào `B` chưa có. Hình dạng "val đẹp ở ep38–39 nhưng recall tệ ở đúng
đó, rồi ep45 bò về ngang `B`" đọc đúng là: 120 hàng đi một vòng nhiễu rồi quay lại gần chỗ cũ.

📌 Cảnh báo này đã được ghi sẵn trong docstring `--emb-term-mask` **trước khi chạy**; giờ có số.

---

## BitFit encoder (2026-08-13): **chạm được encoder thật — và làm hại thật**

Cấu hình: `TRAINABLE_MODULES='re:(bypass_scale|norm\.bias|norm\.log_scale)'`, lr 3e-5, ep36→45.

| nhóm | tensor | param |
|---|---:|---:|
| `bypass_scale` | 53 | 19.072 |
| `norm.bias` | 17 | 5.952 |
| `norm.log_scale` | 17 | 17 |
| **tổng** | **87** | **25.041 = 0,036%** |

### Cơ chế đã xác minh — diff trọng số ep36 vs ep35

```
bypass_scale   53 tensor  19072 param  |d|max 2.755e-02
norm.bias      17 tensor   5952 param  |d|max 8.424e-02
norm.log_scale 17 tensor     17 param  |d|max 2.803e-03
=> khong co nhom "KHAC"  (0 tensor ngoai pham vi bi doi)
```

63,4M param encoder vẫn nằm trong optimizer mà **không một tensor nào ngoài phạm vi dịch** —
xác nhận `re:` filter đúng, và lr đủ để có tín hiệu (`|d|max` = 8,4e-02).

### Đây là run ĐẦU TIÊN trong project làm `CTC WER` dịch

| ep | 36 | 37 | 38 | 39 | 40 | 41 | 42 | 43 | 44 |
|---|---|---|---|---|---|---|---|---|---|
| ctc | 33,33 | 33,65 | 33,71 | 33,68 | 33,72 | 33,72 | 33,80 | 33,90 | **34,02** |
| transducer | **32,75** | 32,81 | 32,96 | 32,87 | **32,74** | 32,99 | 33,14 | 33,21 | 33,21 |

`ctc` **đơn điệu tăng** 33,18 → 34,02 = **+0,84 pp**, gấp gần 2× ngưỡng nhiễu 0,45 pp. Mọi run
trước (`joiner`, `decoder`, `emb`, `dproj`, hướng 5) đều đóng băng `ctc_output` **hoặc** encoder
nên `ctc` bất động ở đúng 33,18. Ở đây `ctc_output` **cũng bị đóng băng** — nên toàn bộ 0,84 pp
là do **biểu diễn của encoder xấu đi**. Đó là bằng chứng trực tiếp: 25K param bias thật sự
điều khiển được encoder, và ở lr này chúng điều khiển sai hướng.

### Recall `exact` — không đổi được gì so với `ep35`, và thua `B` rõ

| ckpt | `beam16+hw` | `beam16+hw+bp1` | WER (hw) | #từ `beam16`/160s |
|---|---|---|---|---|
| bitfit_ep36 | 54/145 = 37,2% | 55/145 = 37,9% | 31,96% | — |
| bitfit_ep40 | 56/145 = 38,6% | 57/145 = 39,3% | 32,53% | — |
| bitfit_ep45 | 58/145 = 40,0% | 62/145 = 42,8% | 32,81% | 523 ✅ |
| *ep35 gốc* | *53/145 = 36,6%* | *55/145 = 37,9%* | — | *536–608* |
| **B_ep40** | **64/145 = 44,1%** | 66/145 = 45,5% | — | 545 |

Bootstrap:

| | ΔWER | Δexact |
|---|---|---|
| bf_ep36 − B_ep40 `bp1` | −0,18 ✗ | **−7,6 [−12,6,−3,1] ✅ thua** |
| bf_ep40 − B_ep40 `bp1` | +0,03 ✗ | **−6,2 [−11,1,−1,5] ✅ thua** |
| bf_ep45 − B_ep40 `bp1` | +0,43 ✗ | −2,8 [−7,6,+1,7] ✗ |
| bf_ep36 − B_ep40 `hw` | +0,16 ✗ | **−6,9 [−11,6,−2,6] ✅ thua** |
| bf_ep40 − B_ep40 `hw` | **+0,73 [+0,09,+1,39] ✅ hại** | **−5,6 [−10,7,−0,7] ✅ thua** |
| bf_ep45 − B_ep40 `hw` | **+1,02 [+0,36,+1,69] ✅ hại** | −4,2 [−9,0,+0,6] ✗ |
| **bf_ep36 − ep35** `hw` | +0,01 ✗ | **+0,7 [+0,0,+2,1] ✗** |
| **bf_ep40 − ep35** `hw` | **+0,58 [+0,02,+1,17] ✅ hại** | +2,0 [−0,8,+5,1] ✗ |
| **bf_ep45 − ep35** `hw` | **+0,87 [+0,31,+1,46] ✅ hại** | +3,4 [+0,0,+7,0] ✗ |

Đọc hai khối cuối cạnh nhau là đủ kết luận: **so với `ep35` chưa finetune, BitFit không
thắng được ô `Δexact` nào (CI ôm 0 hết, biên dưới chạm đúng +0,0), nhưng thắng 2 ô `ΔWER`
theo chiều XẤU.** Tức nó trả WER mà không nhận lại recall.

⇒ **Hướng BitFit đóng ở lr 3e-5.** Đây là một null result **có thông tin**: câu hỏi thiết kế
("0,036% param có chạm nổi encoder không?") được trả lời **CÓ** — hai lần, bằng diff trọng số và
bằng `ctc` dịch. Câu trả lời cho câu hỏi thứ hai ("chạm thì có lợi không?") là **không**.
25K param quá ít để học đặc tính phổ của val nhưng vừa đủ để overfit 200h TTS 8 kHz: đường
`ctc` đơn điệu tăng, `transducer` tốt nhất ngay ở ep36/ep40 rồi rã.

Nếu còn muốn thử lại thì biến duy nhất đáng đổi là **hạ `BASE_LR` xuống dưới 3e-5**
(1e-5 hoặc 3e-6) — vì `|d|max` = 8,4e-02 cho thấy bước đi hiện tại lớn hơn nhu cầu. Nhưng
ưu tiên nên thấp: hai hướng encoder trước (`C_deep` 30M param, `C_front` 6,5M) cũng đã ra
hoà và có hại.

---

## Export ONNX `B_40_onnx` + đối chiếu PyTorch ↔ ONNX (2026-08-13)

Export `callbacks_fpt200_B/epoch-40.pt` → ONNX transducer fp32, đẩy lên
`huggingface.co/lmq1909/zipformer-vi-fpt200-finetune/tree/main/B_40_onnx` (repo private).
Bản HF khớp bản local: 4 file LFS khớp **sha256** (gồm `encoder` 261 MB), 3 file nhỏ khớp size.

Folder gồm 7 file: `encoder/decoder/joiner-epoch-40.onnx`, `tokens.txt`, `bpe.model`,
`bpe.vocab`, `hotwords.txt`. Nội bộ nhất quán: `tokens.txt` và `bpe.vocab` khớp **từng ID**
của `bpe.model` (lệch 0/2000), `vocab_size` metadata ONNX = 2000 = số token, 109 cụm hotwords
encode **0 lần** ra `<unk>`.

### Bốn model, cùng 478 cut val

WER micro / CER (đã bỏ space) / recall thuật ngữ trên 145 lần xuất hiện.

| mode | model | WER | CER | exact | *equiv* |
|---|---|---|---|---|---|
| greedy | PT-35 | 32,85% | 27,42% | 49 = 33,8% | *67 = 46,2%* |
| greedy | ONNX-35 | 32,81% | 27,39% | 49 = 33,8% | *67 = 46,2%* |
| greedy | PT-40 | 32,62% | 27,00% | 49 = 33,8% | *67 = 46,2%* |
| greedy | **ONNX-40** | **32,56%** | **26,99%** | 49 = 33,8% | *67 = 46,2%* |
| beam16 | PT-35 | 32,21% | 26,86% | 49 = 33,8% | *68 = 46,9%* |
| beam16 | ONNX-35 | 32,23% | 26,83% | 49 = 33,8% | *69 = 47,6%* |
| beam16 | PT-40 | 32,30% | 26,78% | 51 = 35,2% | *68 = 46,9%* |
| beam16 | ONNX-40 | 32,26% | **26,75%** | 51 = 35,2% | *68 = 46,9%* |
| beam16+hw | PT-35 | 31,94% | 26,53% | 53 = 36,6% | *86 = 59,3%* |
| beam16+hw | ONNX-35 | 31,93% | 26,54% | 53 = 36,6% | *86 = 59,3%* |
| beam16+hw | PT-40 | 31,79% | 26,27% | 64 = 44,1% | *93 = 64,1%* |
| beam16+hw | **ONNX-40** | **31,75%** | **26,21%** | **65 = 44,8%** | *94 = 64,8%* |
| beam16+hw+bp1 | PT-35 | 32,23% | 26,84% | 55 = 37,9% | *90 = 62,1%* |
| beam16+hw+bp1 | ONNX-35 | 32,18% | 26,77% | 55 = 37,9% | *91 = 62,8%* |
| beam16+hw+bp1 | PT-40 | 32,53% | 26,93% | **66 = 45,5%** | *97 = 66,9%* |
| beam16+hw+bp1 | ONNX-40 | 32,60% | 26,98% | **66 = 45,5%** | *97 = 66,9%* |

`PT-35` = `callbacks/epoch-35.pt` · `ONNX-35` = bundle prod `gipformer/*-epoch-35-avg-6.onnx`
`PT-40` = `callbacks_fpt200_B/epoch-40.pt` · `ONNX-40` = `B_40_onnx/*-epoch-40.onnx`

**`epoch-35.pt` và `decoder-epoch-35-avg-6.onnx` là CÙNG trọng số** — so trực tiếp
`decoder.embedding.weight` ra `|d|max = 0,000e+00` (trùng bit-per-bit). Tên `avg-6` chỉ là cách
đặt tên, không phải model khác. ⇒ cặp `PT-35`↔`ONNX-35` là so **thuần pipeline**.

**Đọc bảng:** lệch PT↔ONNX tối đa **0,07 pp** WER, dưới ngưỡng nhiễu 0,45 pp; recall `exact`
trùng khít **15/16 ô** (ô duy nhất lệch là `ONNX-40 beam16+hw`, 65 vs 64 = đúng MỘT lần xuất
hiện). ⇒ ONNX thay thế được PyTorch. Và ở greedy **cả bốn model cho ĐÚNG cùng recall**
(49 exact) — finetune decoder không tự đọc ra thuật ngữ, nó chỉ làm hotword boost bám tốt hơn;
chênh thật chỉ xuất hiện ở `beam16+hw` (53 → 65).

### Vì sao PT ≠ ONNX: feature của PyTorch bị NÉN MẤT MÁT

```
PyTorch : wav -> resample -> kaldifeat -> NÉN lilcom -> .lca -> giải nén -> encoder
ONNX    : wav -> resample -> kaldi-native-fbank -> encoder            (không nén)
```

`lhotse.LilcomChunkyWriter` mặc định `tick_power = -5` ⇒ bước lượng tử `2^-5 = 0,03125`,
**sai số tối đa nửa bước = 0,015625**. Đo trên 6 cut val, so feature đã lưu với bản tính lại
từ wav (cùng config, cùng resampler):

| | |
|---|---|
| `\|d\|mean` | 0,0078 |
| `\|d\|max` | **0,016 — GIỐNG HỆT ở mọi cut** |
| offset trung bình | ±0,0001 (không thiên vị) |
| biên độ feature | ~[−16, +4,5] |

`|d|max` chạm cùng một trần ở mọi cut là dấu hiệu quyết định: nếu do hai cài đặt fbank khác
nhau thì sai số phải **biến thiên theo nội dung**. Đây là trần lượng tử, và `0,015625` khớp
chính xác con số đo được.

Hệ quả: ở greedy **16/478 cut (3,3%)** ra khác nhau, toàn là đổi MỘT từ ở chỗ mơ hồ
(`có phải`↔`không phải`, `làm gì`↔`giấy`). Vì nhiễu lượng tử đối xứng nên **không thiên vị**:
trên riêng 16 cut đó ONNX tốt hơn 1,54 pp, còn toàn corpus chỉ còn 0,06 pp — các lần lật
triệt tiêu nhau.

Hai nghi phạm **đã loại**:
- **Cấu hình fbank**: sherpa mặc định `dither=0`, `high_freq=-400`, `snip_edges=False`,
  `low_freq=20`, 80 mel, 16 kHz — trùng khít `prepare_data.py`. Không lệch knob nào.
- **Resampler**: toàn bộ 478 cut val là 11025 Hz nên cut nào cũng phải resample. Đổi RIÊNG
  resampler (sherpa nội bộ ↔ `torchaudio.functional.resample`) làm 6/478 cut đổi kết quả,
  nhưng khoảng cách PT↔ONNX **vẫn giữ 16 cut** ở cả hai cách ⇒ không phải nguyên nhân.
- Nhiễu fp32 giữa PyTorch và ONNX Runtime cỡ 1e-6, **nhỏ hơn nhiễu lilcom bốn bậc** ⇒ bỏ qua.

⚠️ **Áp cho MỌI số PyTorch trong file này**: tất cả đều đo trên fbank đã nén lilcom, tức mang
thêm ±0,0156 nhiễu feature. 0,06 pp nằm sâu dưới ngưỡng 0,45 pp nên không lật kết luận nào,
nhưng đây là tầng nhiễu **cố hữu**, không phải sai số của một lần đo. Muốn hai đường sát nhau
hơn thì hạ `tick_power` (vd `-8` → sai số 0,002) lúc extract, đổi lại fbank phình ra.

### ⚠️ HAI BUNDLE CÓ KIỂU CHỮ NGƯỢC NHAU — hotwords lệch kiểu chữ là vô hiệu IM LẶNG

| bundle | `bpe.model` | `'QUAI PHAI'` | `'quai phai'` |
|---|---|---|---|
| `gipformer/` (prod, ep35-avg-6) | **1996/2000 piece CHỮ HOA** | hợp lệ | `<unk>` |
| `B_40_onnx/` (finetune ep40) | **0/2000 piece chữ hoa** | `<unk>` | hợp lệ |

Cùng 2000 token, **cùng token ID** (`▁TÂM 199` ↔ `▁tâm 199`), chỉ khác cách viết — bản train
lowercase toàn bộ piece ở `entrypoint.sh` bước 2, bundle prod thì không.

Sherpa-onnx **không** tự chuẩn hoá hoa/thường hotword (icefall `eval_beam.py` thì tự lowercase
trong `build_context_graph`). Lệch kiểu chữ ⇒ mọi cụm ra `<unk>` ⇒ context graph rỗng ⇒ biasing
vô hiệu **hoàn toàn** mà **không có log nào**. Đo trên `B_40_onnx`, chỉ đổi hoa/thường file
hotwords:

| file hotwords | WER | exact | equiv |
|---|---|---|---|
| chữ HOA | 32,24% | 51/145 | 68/145 |
| **chữ thường** | **31,75%** | **65/145** | **94/145** |
| *(mốc: beam16 KHÔNG hotwords)* | *32,26%* | *51/145* | *68/145* |

Bản chữ HOA ra **y hệt dòng không hotwords** — đó là chữ ký của "vô hiệu", không phải "yếu".

Đã sửa `_build_hotwords_file()` ở cả `Core-AI-API` và `Core-AI-Worker`: thêm
`_kieu_chu_vocab(model_dir)` dò `bpe.vocab` (fallback `tokens.txt`) trả `lower`/`upper`/`mixed`,
rồi `_doi_kieu_chu()` áp **ngay trước khi ghi file** — phải ở đó chứ không phải sửa file tĩnh,
vì hàm đó gộp cả dòng từ DB (`canonical` kiểu `FPT`, `Wifi` — cũng viết HOA). Kiểm: bundle prod
→ `upper` (giữ nguyên, **hành vi prod không đổi**), `B_40_onnx` → `lower`.

### Còn chặn gì để deploy `B_40_onnx`

| việc | trạng thái |
|---|---|
| `decoding_method = modified_beam_search`, `blank_penalty = 1.0` | ✅ đã bật ở prod |
| khối hotwords trong `model_registry.py` | ✅ đã bỏ comment |
| `gipformer/bpe.vocab` | ❌ **thiếu** ⇒ code đi vào nhánh tắt biasing (CÓ log warning) |
| tên file | ❌ `model_registry.py:234-236` hardcode `*-epoch-35-avg-6.onnx`, bộ mới là `*-epoch-40.onnx` |
| bản int8 | ❌ không có (image thay thế thiếu `onnxruntime`); prod nạp fp32 nên không cản |

⇒ **Prod hiện tại đang ở gần dòng `ONNX-35 beam16` (32,23% / 33,8%), KHÔNG phải dòng `+hw`.**

⚠️ `bpe.vocab` cho prod phải sinh từ **`bpe.model` của prod** (chữ HOA). Copy
`B_40_onnx/bpe.vocab` (chữ thường) vào `gipformer/` sẽ làm hotwords hỏng — đúng lỗi trên, đổi chiều.

### Cách tái lập

`scripts/export_onnx.py` (đã sửa: import `onnxruntime` thành lazy để `--no-int8` có nghĩa) và
`scripts/eval_onnx.py` (mới — decode val bằng sherpa-onnx, ghi JSONL cùng schema `eval_beam.py`
nên `scripts/metrics.py` dùng lại được).

Hai bẫy môi trường gặp phải:
- Image `zipformer-training:latest` **đã bị xoá khỏi máy** giữa phiên (30,9 GB). Dùng
  `luongnguyenminhan/speech_to_text_api:latest` thay thế — có torch 2.8 CPU + onnx, nhưng
  **không có `pip`, `jiwer`, `onnxruntime`, `kaldifeat`**. Nên: container chỉ decode, WER tính
  ở host bằng venv `uv` riêng.
- Phải mount **cả cây icefall**, không chỉ thư mục `zipformer`: `encoder_interface.py` và 2 file
  khác là **symlink trỏ ra ngoài** (`../transducer_stateless/` …), mount hẹp làm chúng dangling
  và báo `ModuleNotFoundError` rất khó đoán.

## RUN FROM-SCRATCH 100 EPOCH (2026-11-09 → 2026-11-11): mô hình tự train hoàn toàn — KHÔNG dùng epoch-35.pt

Run đầu tiên của repo **không kế thừa gipformer**: khởi tạo random, BPE tự train, data tự tải.
Chạy trên máy khác mọi run cũ: **RTX PRO 6000 96GB + NVMe** (không phải 5080 + HDD).

### Data (430,5h / 217.232 cut / 10 bộ, split 95/5 hash)

| dataset | cut | giờ | % | dataset | cut | giờ | % |
|---|---:|---:|---:|---|---:|---:|---:|
| vimd | 18.949 | 102,6 | 23,8% | gigaspeech | 11.035 | 21,3 | 4,9% |
| vlsp2020 | 56.427 | 101,4 | 23,5% | vietmed | 9.207 | 15,9 | 3,7% |
| lsvsc | 56.823 | 100,7 | 23,4% | multimed | 9.122 | 15,8 | 3,7% |
| fosd | 25.917 | 30,2 | 7,0% | vivos | 12.420 | 15,7 | 3,6% |
| **onevoice** | 13.076 | 22,2 | 5,2% | vietmdd | 4.266 | 4,9 | 1,1% |

Tele/sale + y tế (domain đích) ≈ 82h = 19%; còn lại nền âm học. Val = 10.830 cut (lọc ≥1s),
**khác hẳn val 478 cut của mọi run cũ** — mọi số dưới đây KHÔNG so được với số của section finetune.

### Config khác biệt so với các run finetune

- `MAX_DURATION=1200` (96GB cho phép), fp16 AMP, `NUM_WORKERS` truyền hẳn vào train.py
  (entrypoint cũ không truyền ⇒ icefall mặc định 2 worker ⇒ GPU bỏ đói 1,44 s/batch — phát hiện
  bằng py-spy: main process kẹt 1 nhân CPU, GPU 0–5%)
- BPE **unigram 2000 tự train** trên 217k transcript đã normalize, đúng convention icefall
  (`<blk>`=0, `<sos/eos>`=1, `<unk>`=2 — sai thứ tự này là hỏng ngầm export_onnx/sherpa vì cả
  hai hardcode blank=0)
- Eden: BASE_LR 0.045, lr_batches 7500, lr_epochs 3.5 — giữ nguyên, epoch tuyệt đối 1→100

### Đường hội tụ (VALID transducer, greedy trong log)

| mốc | best đơn | avg cửa sổ | giờ GPU |
|---|---|---|---|
| 40 | 9,25 (ep37) | 8,22 (avg-37-40) | ~22h |
| 60 | 8,60 (ep59) | 7,99 (avg-54-60) | +14h |
| 100 | 8,17 (ep100) | **7,84 (avg-96-100)** | +25h |

Từ random (632%) → 7,84%. Mỗi khối +40 epoch chỉ mua ~0,4 pp đơn / 0,2 pp avg — hội tụ trên
430h; TRAIN WER ~1% từ ep40 (gap +8,3pp và nới dần = thuộc tập train, phần val còn lại là
khoảng cách phân phối chứ không thiếu epoch).

### Kết quả cuối: `callbacks_scratch/avg-96-100.pt` — đủ 6 chế độ (giao thức mới áp dụng)

| chế độ | WER micro | CER | thuật ngữ exact | equiv |
|---|---:|---:|---:|---:|
| greedy | 7,84% | 5,82% | 112/142 = 78,9% | 81,7% |
| greedy+bp1 | 7,66% | — | — | — |
| beam16 | 7,60% | 5,56% | 80,3% | 83,8% |
| beam16+bp1 | 7,53% | — | — | — |
| beam16+hw | 7,60% | 5,56% | 82,4% | 87,3% |
| **beam16+hw+bp1** | **7,54%** | 5,46% | **118/142 = 83,1%** | 87,3% |

Lần đầu trên model tự train xác nhận cơ chế hotwords: **+4,2 pp thuật ngữ exact (78,9→83,1)
không tốn WER tổng** (7,60→7,60). bp1 đổi deletion lấy substitution đúng lý thuyết
(del 2,20→1,79, sub 3,88→4,04).

### Gold PNC (thoại call-center thật, 20 đoạn đã duyệt ~66s) — KHÔNG dùng xếp hạng

`gold_data_s2t.zip` của bên ngoài: WER đo được 63–68% mọi chế độ mọi checkpoint, NHƯNG phát
hiện **label JSON lệch timestamp với audio mp3** (JSON khai sai độ dài cuộc gọi; nhiều cặp
ref/hyp khác hẳn nội dung; annotator tự ghi "đoạn này không có trong hội ghi âm"). Decode
**full-call** (không cắt) cho transcript mạch lạc có cả thuật ngữ (`in tơ nét`) — model nghe
được, phép đo bị hỏng bởi alignment. Bai transcript: `gold_eval/full_call_transcripts.jsonl`.
Cần bản JSON+audio khớp + DVKH 5,76h để đo thật — khi đó mới so được với gipformer.

### Bẫy kỹ thuật mới (chưa có trong CLAUDE.md)

1. **`docker commit` giữ nguyên Cmd của container đang chạy** — commit từ container build
   kaldifeat làm image có ENTRYPOINT=`bash -c "git clone..."` thay vì entrypoint.sh. Phải
   `docker commit --change 'ENTRYPOINT ["/app/scripts/entrypoint.sh"]' --change 'CMD []'`.
2. **`k2-fsa/kaldifeat` repo đã 404** — dùng fork `csukuangfj/kaldifeat` (tác giả chính).
   Import kaldifeat MỘT mình dính oneMKL error; import torch TRƯỚC thì OK.
3. **eval_beam 6 chế độ trên 10.830 cut = ~11h** khi chia GPU (beam16+hw là vòng python từng
   bước). Cần sửa script ghi output từng batch nếu muốn xem tiến độ; hiện không có cách check
   % (strace bị cap chặn, .lca không giữ FD mở).
4. **Máy dùng chung**: 2 lần OOM CUDA + 1 SIGKILL (exit 137, host OOM-killer) khi dự án khác
   (`lmq`) chiếm VRAM/RAM — relaunch resume từ checkpoint là sạch, chỉ mất epoch dở.
5. Extract fbank kaldifeat GPU trên NVMe = **374 cut/s** (train 206k cut trong 9,2 phút) —
   so với 14,8 giờ của prepare_data trên HDD cũ.
6. `entrypoint.sh` đã thêm `--num-workers` cho train.py (fix bẫy 1 ở config).
7. `scripts/avg_ckpt.py`, `scripts/infer_batch.py`, `scripts/train_bpe_scratch.py` là sản phẩm
   phụ dùng lại được; janitor checkpoint (`/tmp/ckpt_janitor.sh`) cần khi đĩa <10GB.

### Kết luận run

- **Sản phẩm: `avg-96-100.pt` = 7,54% WER / 83,1% thuật ngữ exact (beam16+hw+bp1) trên khung
  val mới.** Đã vắt kiệt 430h — thêm epoch chỉ ~0,005 pp/giờ GPU.
- Bước tiếp theo có ý nghĩa duy nhất: **thoại call-center thật** (gold align đúng + DVKH 5,76h
  + mọi nguồn thật khác) để finetune từ avg-96-100 — theo mọi bằng chứng của repo, 5h thật
  đáng hơn 200h TTS/read speech lặp lại.

## Lịch sử các run

### Trên 200h FPT (finetune từ ep35)

| ngày | phạm vi | lr | kết quả |
|---|---|---|---|
| 08-04 Run A | `FREEZE_ENCODER` 8,1% | 1,3e-4 | **sụp** 43,23 → 73,93% |
| 08-04 Run B | joiner 2,2% | 1,3e-4 | **sụp** 42,10 → 67,74% |
| 08-04 Run C | joiner 2,2% | 4,4e-6 | 32,97 / **32,91** / 32,97 *(khung cũ, baseline 33,20)* |
| 08-11 | joiner+decoder 3,7% | 3e-6 | đáy ep37 **32,53** (−0,32) |
| 08-11 | joiner+decoder 3,7% | **3e-5** | **SỤP** 33,35 → **70,69%** (10 epoch, đơn điệu) |
| **08-11 (B)** | **decoder 1,5%** | **3e-5** | đáy **ep40 32,62** (−0,23), sống sót ← **TỐT NHẤT** |
| 08-12 (B_lr5x) | decoder 1,5% | 1,5e-4 (×5) | ep36 32,77 rồi **sụp** → 43,70% |
| 08-12 (B_lr20x) | decoder 1,5% | 6e-4 (×20) | **sụp ngay ep36** 37,36 → 66,11% |
| 08-12 (B_emb) | `decoder.embedding` 1,5% | 3e-5 | đáy **ep39 32,54**; thuật ngữ **hoà** với B ở cả hai trục (equiv vs ep35 chỉ +2,8 pp, CI ôm 0) ⇒ **không thay B, cũng không kém B** |
| 08-12 (C_deep) | **`encoder.encoders.3` 43,6%** | 1e-5 | đáy **ep38 32,57** — 30,34M param mà **hoà** với B (1M param) ⇒ vô ích |
| 08-12 (C_front) | **`encoder_embed`+stack 0,1 9,3%** | 1e-5 | **xấu ngay ep36** 33,45 → 34,72, WER +2,13 pp và thuật ngữ −9,6 pp vs B (CI không ôm 0) ⇒ **CÓ HẠI** |
| 08-13 (h5) | `decoder` + mask 120 dòng emb **0,094%** | 3e-5 | đáy val **ep38 32,62** nhưng exact ở đó **thua B −2,8 pp** (CI không ôm 0); ep45 hoà ⇒ **NULL/ÂM** |
| 08-13 (BitFit) | `bypass_scale`+`norm.bias`+`norm.log_scale` **0,036%** | 3e-5 | chạm được encoder (**ctc dịch 33,18 → 34,02, đơn điệu**) nhưng exact thua B −5,6 pp và WER hại +0,87 pp vs ep35 ⇒ **CÓ HẠI** |

**Cùng lr 3e-5, cùng data, cùng số bước: joiner+decoder → 70,69%, decoder-only → 33,78%.
Chênh 37 điểm.** ⇒ **`joiner` là chỗ sụp đổ, không phải `decoder`.** Joiner sinh logit trên
vocab kể cả blank nên là đường trực tiếp tới nghiệm thiên blank; decoder chỉ cấp biểu diễn
lịch sử, một mình không đẩy được blank thắng. ⇒ **decoder là tầng an toàn để đẩy lr.**

### Trên 2.340h (public + FPT) — mọi hướng đều thất bại

| ngày | cấu hình | kết quả |
|---|---|---|
| 08-05 | joiner 2,2%, lr 4,4e-6→3e-6, 1 epoch | 33,20 → 33,42 *(khung cũ)* = trong nhiễu |
| 08-06 | **full finetune**, lr 6e-5, md 250 | **+1,16 pp (HẠI)**, CI95 [+0,26, +2,08] |
| 08-06 | full finetune + `BAND_CLAMP_BIN=54` | +1,47 pp, tức clamp vô tác dụng (+0,30 so main, CI ôm 0) |
| 08-07 | joiner 2,2% + manifest **đã shuffle**, lr 3e-6 | ep36 **+0,05 pp** = không đo được gì |

**`+1,16 pp` không phải lỗi lr mà là thay data tốt bằng data kém.** 2.340h của ta KHÔNG phải
data gốc — nó không có call-center thật, ~92% là read speech/TTS. Đó là lý do chốt hướng
**chỉ finetune trên 200h FPT**. Hệ quả: `finetune.py --use-mux` (trộn lại *data gốc*) là
**không làm được** — 2.100h public là hàng thay thế không cùng phân phối.

**Shuffle manifest không đổi kết luận** (+0,05 vs +0,22 của run chưa shuffle). Và
joiner-only đóng băng encoder nên vô hại nhưng cũng vô ích trên 2.340h.

### Decode-side (2026-08-10, khung cũ)

Đo lại 6 checkpoint trên cùng một thước:

| checkpoint | micro WER | ghi chú |
|---|---|---|
| `epoch-35.pt` | 32,88% | gốc |
| `callbacks_joiner_lr44e6/epoch-37.pt` | **32,57%** | −0,31, CI ôm 0 |
| `callbacks/epoch-36.pt` (full finetune 2.340h) | 34,05% | **+1,17** ✅ hại |
| `callbacks_clamp54/epoch-37.pt` | 34,35% | **+1,47** ✅ hại |

**Beam search:** ep35 greedy 32,88 → beam8 32,30 → beam16 32,26. `beam8−greedy = −0,58`
[−1,10, −0,11] ✅. Nhưng thứ tự **ĐẢO** so với greedy: `lr44e6 ep37` hơn 0,31 pp ở greedy mà
**thua** 0,27 pp ở beam16. ⇒ **chốt bằng greedy là kết luận ngược dấu.**

**Cơ chế:** WER là **đường cong chữ U theo độ dài output** (ref 6.687 từ), đáy ~6.540:

| | #từ hyp | so ref | WER |
|---|---:|---:|---:|
| ep35 greedy | 6.404 | −4,2% | 32,88% |
| ep35 beam16 | 6.545 | −2,1% | **32,26%** ← đáy |
| ep35 beam16+bp3 | 7.278 | +8,8% | 35,25% |

`blank_penalty` và finetune-joiner **vặn cùng một núm** (model nói nhiều hay ít): `ep37 greedy`
có del% 9,85 ≈ `ep35 beam4` del% 9,80. Dùng cả hai là vặn quá tay. Beam luôn đổi `del↓` lấy
`sub↑ + ins↑`.

### Sửa lại ghi chú cũ: joiner chỉ trôi về MỘT hướng

Từng ghi là lr thấp/cao làm joiner trôi ngược chiều nhau. **Sai.** Độ dài output greedy qua
Run C (lr 4,4e-6): ep35 6.404 → ep36 **6.614** → ep37 6.520 → ep38 6.472; del% 8,99 → 9,85 →
10,26. Sau một pha thích nghi ở ep36 thì nó **bò ngược về** mức ep35. Cùng hướng với sụp đổ ở
lr 1,3e-4, chỉ chậm hơn. `ep37` là đáy vì bắt đúng lúc gần đáy chữ U, không phải vì học được.

Giả thuyết: lattice RNN-T bị blank áp đảo (T khung ≫ U nhãn) nên nghiệm thiên blank là điểm
hút; TTS sạch/ngắn/đều càng ít phạt việc dừng sớm.

⚠️ Bằng chứng hai vế **không ngang nhau**: vế lr thấp đo corpus-level 478 cut; vế lr cao chỉ
có **một câu đếm tay** (42 từ GT → 37/33/16/15), và checkpoint Run A/B đã bị xoá.

---

## Đã chốt

| | |
|---|---|
| ✅ dùng | `B_ep40` + beam16 + hotwords, `bp` = 0 hoặc 1 |
| ❌ loại | `blank_penalty=2` — WER +2,26 pp, thuật ngữ không tăng thêm so bp1 |
| ❌ loại | epoch 45 — sụp trên audio dài (182/548 từ) |
| ❌ loại | finetune trên 2.340h — +1,16 pp, data kém hơn data gốc |
| ❌ loại | `BAND_CLAMP_BIN` / `BAND_LIMIT_SR` — +0,30 pp, CI ôm 0 |
| ❌ loại | mở `joiner` ở lr ≥ 3e-5 — sụp 70,69% |
| ❌ loại | lr > 3e-5 cho `decoder`-only — 1,5e-4 sụp từ ep37, 6e-4 sụp ngay ep36, và **không** đổi lại thuật ngữ nào |
| ❌ loại | train thêm epoch ở 3e-5 — sau ep40 cả WER lẫn thuật ngữ đều xấu đi đơn điệu |
| ❌ loại | thêm checkpoint giữa epoch — ep36–40 trải 0,16 pp, dưới ngưỡng nhiễu 0,45 pp |
| ❌ loại | **`encoder_embed` + stack 0,1 (`C_front`)** — xấu hơn ep35 **ngay epoch đầu**, tăng đơn điệu tới +2,13 pp WER và **−9,6 pp thuật ngữ** vs B_ep40, cả hai CI không ôm 0. (So với chính ep35 thì thuật ngữ chỉ −0,7 pp, CI ôm 0 — cái hại đo được chắc chắn là **WER**, cộng với việc không lấy được +8,9 pp mà B lấy được) |
| ❌ loại | **`encoder.encoders.3` (`C_deep`)** — 30,34M param (43,6% model) mà **hoà** với B_ep40 (1M param) ở cả hai trục ⇒ vô ích, không phải có hại |
| ❌ loại | `decoder.embedding` thay cho `decoder` (`emb`) — hoà tuyệt đối với B_ep40, không có lý do đổi (nhưng ổn định hơn: trôi +0,69 vs +1,16 pp) |
| ❌ loại | **`joiner.decoder_proj` thêm vào `decoder` (`dproj`)** — ep36–38 hoà với B_ep40 (CI ôm 0), từ ep39 recall "tăng" nhưng đó là do nuốt nội dung: ep45 equiv 62,8% mà `beam16` trên 160s chỉ còn **21/548 từ** |
| ❌ loại | **train tiếp từ B_ep40 trên data dày hơn (`dense`)** — ep41/42 ra **đúng từng con số** của B_ep40 (Δequiv `+0,0 [+0,0, +0,0]`, không một cut nào đổi) ⇒ trần ep40 không phải bài toán lr hay khối lượng data |
| ⚠️ bắt buộc | **checkpoint nào có Δrecall > 0 KÈM ΔWER > 0 thì phải đếm số từ trên `audio_test/call_16k.wav` trước khi tin** — val cut 1–20s đã bỏ lọt chế độ "nói ít đi rồi để hotword chèn vào chỗ trống" **hai lần** (`B_ep45`, `dproj_ep39+`) |
| ✅ đã phân giải | trần thuật ngữ ~45% exact / ~66% equiv là **âm học** (CTC ≈ uniform ở mọi vị trí sai) — và **không phá được bằng phạm vi param**: 12 cấu hình từ 4.096 tới 30.343.052 param đều dừng ở cùng chỗ |
| ✅ đã phân giải | **sàn 32,6% là sàn của DATA, không phải của phạm vi param** — `C_deep` mở 43,6% model, gấp 30 lần `B`, vẫn dừng đúng 32,57% |
| ✅ đã xác minh | **`B_40_onnx` (ONNX fp32 của `B_ep40`) tương đương bản PyTorch** — lệch ≤ 0,07 pp WER, recall `exact` trùng 15/16 ô. Đã đẩy lên HF, khớp sha256. |
| ✅ đã phân giải | **PT ≠ ONNX là do fbank của PyTorch bị nén lilcom** (`tick_power=-5` ⇒ sai số ±0,0156), không phải lỗi export. Không thiên vị, triệt tiêu về 0,06 pp. |
| ⚠️ CHẶN deploy | `gipformer/bpe.vocab` **thiếu** ⇒ prod tắt biasing (có log warning). Và `model_registry.py:234-236` hardcode `*-epoch-35-avg-6.onnx` nên bộ `*-epoch-40.onnx` không được nạp. |
| ⚠️ BẪY | **hai bundle có kiểu chữ NGƯỢC nhau** (prod HOA / `B_40_onnx` thường). Lệch kiểu chữ ⇒ hotwords vô hiệu **im lặng**, không log. Đã sửa `_build_hotwords_file()` tự dò theo vocab. |
| ⚠️ chưa đo được | **quên kiến thức nền** — val 100% FPT call-center, không có val public. Đây là blocker cho mọi hướng finetune tiếp. |

## Hướng chưa thử

1. ~~**Bổ sung biến thể vào `hotwords.txt`** để phân biệt "danh sách thiếu" với "không nghe ra
   âm"~~ — **probe âm học đã trả lời thay, và câu trả lời là ÂM HỌC**: 18 vị trí sai không có
   dòng nào trong `hotwords.txt` có CTC ≈ uniform y hệt nhóm đã có dòng. Vẫn nên thêm biến thể
   vì rẻ (và ContextGraph chỉ khớp TỪ ĐẦU CỤM nên `CON QUAI PHAI` là dòng còn thiếu thật),
   nhưng **đừng kỳ vọng nó gỡ được trần 45%**.
2. **Cắt val public** (vài trăm cut ViVoice + bud500) — điều kiện cần để đo forgetting.
3. ~~**LoRA** trên encoder~~ — **hạ ưu tiên xuống thấp nhất sau `C_front`/`C_deep`**: LoRA vẫn
   là hướng acoustic, mà `C_front` cho thấy sửa acoustic trên TTS **có hại** và `C_deep` cho
   thấy thêm dung lượng **vô ích**. Chi tiết nếu vẫn muốn thử: r=8 thêm 1.260.032 param (1,99%),
   merge lại được nên ONNX không phình, nhưng recipe gốc đóng băng **cả joiner và decoder** nên
   phải sửa trước.
4. **Cân lại tỉ lệ FPT** nếu quay lại 2.340h: 8,6% → 30–50%.
5. ~~**Giới hạn update trong `decoder.embedding` ở riêng token thuộc thuật ngữ**~~ — **BỎ, đo
   2026-08-13: không tách được.** Ý tưởng là mask gradient của `decoder.embedding.weight`
   (2000 dòng × 512) để chỉ mở các dòng thuộc thuật ngữ. Đo thật: 87 cụm BPE ra **120/2000
   dòng (6,0%, 61.440 param)**, nhóm Ftel riêng chỉ 24 dòng. Nhưng **các dòng đó không phải
   token riêng của thuật ngữ** — thuật ngữ tiếng Việt là phiên âm bằng âm tiết thường
   (`ép pi ti`, `quai phai`, `mô đem`) nên chúng dùng chung token với từ vựng phổ thông:
   `▁hai`, `▁ba`, `▁bốn`, `▁năm`, `▁em`, `▁qua`, `i`, `t`, `m`…

   Đếm trên 30.000 cut của 200h: các token này xuất hiện **72,4% số lần ở ngữ cảnh THƯỜNG**,
   chỉ 27,6% nằm trong cụm thuật ngữ. `▁hai` 11.544 lần thường / 263 lần trong cụm; `▁em`,
   `▁ba`, `▁bốn`, `▁năm` **0 lần** trong cụm. ⇒ "Khoá phần lớn bảng, mở dòng cần thiết" là
   bất khả: dòng cần thiết **chính là** các dòng tần suất cao nhất. Mask theo dòng không tách
   được gradient thuật ngữ khỏi gradient từ thường.

   Ngoài ra mục tiêu ban đầu (bớt trôi từ thông thường) cũng không có gì để sửa:
   `B_ep40 − ep35` WER = −0,15 [−0,62, +0,34] ✗ — chưa đo được cái trôi nào.

   ➡️ **Vẫn chạy để xác nhận (run `h5`, 2026-08-13) — và kết luận trên đúng.** Xem mục
   *Hướng 5* ở trên: 65.536 param, cơ chế lock bit-exact, `ctc` bất động ở 33,18 cả 10 epoch;
   `exact` **không thắng `B_ep40` ô nào** và ở `ep38` (đáy val) thua **−2,8 pp** với CI không
   ôm 0. ⇒ **ĐÓNG hoàn toàn.**
6. **Tăng weight của loss cho câu chứa thuật ngữ** — chưa thử. Cùng mục đích với dồn mật độ data
   nhưng không làm dài epoch. ⚠️ **Đọc kỹ mốc so sánh trước khi kỳ vọng**: cách **duplicate cut**
   (`fbank_fpt200_dense`, 257.349 cut / 284,0h / 47,2% cut có thuật ngữ) **đã chạy và là null
   result tuyệt đối** — ep41/42 ra đúng từng con số của `B_ep40`, Δ bootstrap `+0,0 [+0,0, +0,0]`,
   không một cut nào đổi (mục *`B_dense`* ở trên). Reweight loss có thể khác duplicate ở chỗ nó
   không làm lệch thống kê batch, nhưng **cùng nhắm vào một giả thuyết đã bị bác**: rằng model
   thiếu lần nhìn thấy thuật ngữ. Probe âm học nói nguyên nhân là /waɪ/ không tồn tại trong
   phân phối, tăng số lần nhìn không tạo ra âm.
7. ~~**BitFit trên encoder**~~ — ❌ **ĐÃ CHẠY 2026-08-13, CÓ HẠI. ĐÓNG.** Xem mục *BitFit encoder*
   ở trên: chạm được encoder thật (`ctc` 33,18 → 34,02 đơn điệu, run đầu tiên làm `ctc` dịch) nhưng
   `exact` thua `B_ep40` −5,6 pp và WER hại +0,87 pp so với `ep35`. Cảnh báo "kỳ vọng nên thấp"
   bên dưới đúng, và còn nhẹ hơn thực tế. Phần dưới giữ lại vì nó là tài liệu cơ chế.

   **Thiết kế ban đầu (giữ để tham khảo):**

   Phạm vi nhỏ nhất chạm được toàn bộ encoder mà chỉ **vặn cường độ**, không tạo biểu diễn mới
   — mọi ma trận trọng số giữ nguyên từng bit. Hai loại núm:
   - `bypass_scale` (53 tensor, 19.072 param): `BypassModule.forward` là
     `src_orig + (src - src_orig) * bypass_scale`, khởi tạo 0,5 ⇒ cổng theo **từng kênh** quyết
     định mỗi layer được đi qua bao nhiêu.
   - `BiasNorm`: `bias` (17 tensor, 5.952) dịch điểm quy tâm + `log_scale` (17 tensor, 17 param)
     là gain. `scales = mean((x−bias)²)^−0.5 · exp(log_scale)`.

   | phạm vi | tensor | param | % model |
   |---|---|---|---|
   | **BitFit-gate** `re:(bypass_scale\|norm\.bias\|norm\.log_scale)$` | 87 | **25.041** | **0,036%** |
   | BitFit-full (thêm mọi `.bias` của Linear/Conv) | 404 | 176.175 | 0,257% |

   Lập luận: sai lệch cần sửa nằm ở acoustic (val 11 kHz băng hẹp vs train TTS 8 kHz) tức trong
   encoder, mà train **trọng số** encoder trên TTS đã đo là có hại (`C_front` +1,98 pp). BitFit
   ở giữa — đủ để bù lệch băng tần/kênh truyền, khó overfit đặc tính 14 giọng TTS.

   ⚠️ Kỳ vọng nên thấp: `C_deep` mở 30,3M param (gấp **1.220 lần**) mà `FPT-val` không nhúc nhích
   (32,85 → 32,87). 25.041 núm vặn cũng chỉ rút từ cùng nguồn data đó.

   **Hai bug đã sửa để chạy được** (`train.py`):
   - `TRAINABLE_MODULES` chỉ khớp *prefix* mà tên thật nằm ở **đuôi**
     (`...layers.0.bypass.bypass_scale`). Thêm chế độ `re:` khớp bằng `re.search` trên tên đầy đủ.
   - `TRAINABLE_MODULES=bypass` khớp **0 param** nhưng danh sách không rỗng nên vẫn chạy tiếp và
     đóng băng **toàn bộ 69,7M** — 10 epoch không cập nhật gì mà không báo lỗi. Nay `raise` khi
     `trainable == 0`, kèm gợi ý tra `architecture.txt`.
   - Kèm theo: `_frozen_top` trước đây suy từ chuỗi prefix (`p.split(".")[0]`), với regex sẽ ra
     rác và loại `encoder` khỏi optimizer — tức loại đúng khối chứa param cần train. Nay suy từ
     `requires_grad` thật.
   - Và: `run_new.sh:42` dùng `source` (bash bóc nháy) còn dòng 248 dùng `--env-file` (Docker
     **không** bóc). Regex bắt buộc phải có nháy trong file env ⇒ container nhận `'re:...'`.
     `train.py` nay tự bóc nháy bao ngoài.

   Chạy: `ENV_FILE=.env_fpt200_bitfit ./run_new.sh` (đã tạo sẵn, lr 3e-5 như `B`, ep36→45, GPU 0).

### Ba module KHÔNG cần thử — kiến trúc loại sẵn

`simple_am_proj` + `simple_lm_proj` + `ctc_output` = **3.078.000 param (4,42%)** nhưng **không
tham gia đường decode transducer**: `simple_*` chỉ xuất hiện ở `model.py:304-305` (tính pruned
loss lúc train), `ctc_output` chỉ ở `model.py:174,227` (nhánh CTC). `eval_beam.py` chỉ gọi
`forward_encoder` → `decoder`/`joiner`. ⇒ Finetune chúng **không đổi một chữ số nào** trong mọi
bảng ở file này.

Cộng với `decoder`/`joiner` đã sweep cạn và `encoder` vừa đóng ở mục trên ⇒ **không còn phạm vi
param nào chưa được đo**. Hướng còn lại là **data** (thoại thật) và **hậu xử lý** (bảng
`hotword_entries`), không phải kiến trúc.

**Đã loại khỏi danh sách: LM shallow fusion / LODR** (quyết định 2026-08-12). Lý do vận
hành: nó **vĩnh viễn** cần một model phụ lúc inference (default `LmScorer` là 104.810.448
param — lớn hơn cả model ASR 69,65M; bản nhỏ 512/512/2-layer vẫn 5,2M), cộng latency thêm,
trong khi trần thuật ngữ ~45% đã đo là **không do tầng ngôn ngữ trôi chưa đủ**.

## 🔬 Quantization int8 — benchmark 4 hàng (2026-11-15, máy mới RTX PRO 6000)

Mục tiêu plan QAT 12/11: int8 giữ WER chênh ≤0,2pp so fp32. Kết quả: **đạt ở mọi hàng,
và PTQ-dynamic là đủ — QAT không cần cho model này.**

Điều kiện đo: val from-scratch 10.830 cut (fbank_scratch dev-clean), WER micro, decode bằng
**sherpa-onnx 2.0.1 CPU** (đúng runtime prod), 4 mode. Audio >20s chia khúc 20s đồng nhất mọi
hàng ONNX (513/10.830 cut) — vì encoder ONNX (trace của icefall) bake Slice PE theo chiều dài
trace; file dài crash Reshape (giới hạn upstream, cả export official lẫn custom).

| hàng | greedy | beam16 | b16+hw+bp1 | Δ quant |
|---|---:|---:|---:|---:|
| PT-fp32 (PyTorch, tham chiếu, không chia khúc) | 7,84 | — | 7,54 | — |
| avg-96-100 fp32 ONNX | 8,00 | 7,76 | 7,69 | — |
| **avg-96-100 int8 PTQ-dynamic** | 7,99 | 7,79 | **7,75** | **+0,06** |
| twin (epoch-2, quant-observer 1 epoch) fp32 | 9,11 | 8,90 | 9,07 | — |
| twin int8 PTQ | 9,15 | 8,94 | 9,09 | +0,02 |
| QAT (epoch-6, fake-quant 4 epoch) fp32 | 8,31 | 8,05 | 7,97 | — |
| QAT int8 | 8,40 | 8,10 | 8,02 | +0,05 |

**Kết luận:**
1. Δ lượng tử 0,02–0,06pp ở cả 3 cặp fp32↔int8 — PTQ-dynamic gần như miễn phí trên zipformer2 69M.
2. **Deploy bằng `onnx_v4_avg/*avg-96-100.int8.onnx` (WER 7,75)**. QAT phản tác dụng ở đây:
   finetune tiếp (lr 1e-3, ScaledAdam) làm rời điểm averaged tối ưu — avg 7,69 → QAT-fp32 7,97
   (twin 1-epoch tụt 9,07, QAT phải 4 epoch để gỡ lên 7,97). Chi phí QAT = thời gian train,
   lợi ích = 0 vì PTQ đã đạt ngưỡng.
3. VALID-WER macro trong lúc train QAT (md=450): baseline 8,97 → ep2 9,80 → ep6 8,51 — xu hướng
   "hồi phục" nhìn bằng macro là thật nhưng bị che bởi việc điểm xuất phát là checkpoint averaged.
4. Chênh 7,69 vs 7,54 (ref PyTorch) = khác runtime (sherpa vs PyTorch decode + fbank riêng) +
   chia khúc 20s — không phải mất mát model.

Hàng hoá: `onnx_v4_avg/` (fp32+int8, kèm tokens/bpe.vocab — **bắt buộc copy kèm**, thiếu là
sherpa nhận recognizer rỗng mà không báo lỗi → crash `_Map_base::at` khi decode). Checkpoint
QAT + twin ở `callbacks_qat/`. File đo đạc: `eval_v4_*.jsonl`.

## Thêm: PTQ-int4 (MatMulNBits, 2026-11-15)

Weight-only int4 (block 128 symmetric, `MatMulNBitsQuantizer`) trên encoder của avg-96-100 —
sherpa-onnx 2.0.1 chạy được (kernel MatMulNBits có sẵn), bundle 272→52 MB (5,2×).

| | greedy | b16+hw+bp1 | Δ quant | size | tốc độ 1-thread |
|---|---:|---:|---:|---:|---:|
| fp32 | 8,00 | 7,69 | — | 272 MB | 22,1× RT |
| int8 PTQ | 7,99 | 7,75 | +0,06 | 75 MB | 30,9× RT |
| **int4 PTQ** | 8,12 | **7,79** | **+0,10** | 52 MB | 25,3× RT |

⇒ **int4 cũng gần như miễn phí trên model 69M** — QAT (int8 lẫn int4) không còn đất cứu:
tối đa thu hồi ~0,1pp trong khi chính việc train tiếp đã chứng minh tốn ~0,28pp (QAT-int8).
QAT-int4 đã dựng xong code (`--qat-bits 4`, `.env_qat4`) nhưng KHÔNG chạy vì (1) không có
lợi ích kỳ vọng, (2) image docker bị prune lần 2 bởi người khác + ổ không đủ build lại.

Khuyến nghị deploy cuối: **int8 (7,75%)** cần tốc độ / **int4 (7,79%)** cần size.
Lưu ý vận hành: cấu hình latency tốt trên máy tải là 1-stream × 1-2 thread (bench 15/11),
decode batch-32 chậm hơn từng câu vì padding.

## Bench A/B công bằng fp32/int8/int4 (2026-11-16, `scripts/bench_ab.py`)

Số tốc độ cũ (bench tuần tự, mỗi model một lượt) trên máy shared bị nhiễu tải của người
dùng khác — đo lại công bằng: 1 process nạp đồng thời 3 model, CÙNG config (greedy, batch 1,
1 thread), interleave block 25 cut xoay vòng model, 3 round lấy median, `taskset -c 4` ghim
1 core. Variance giữa các round < 2%.

| | fp32 | int8 PTQ | int4 PTQ (MatMulNBits b128) |
|---|---:|---:|---:|
| wall (median 3 round) | 48,7s | 32,3s | 52,4s |
| × realtime | 29,3× | **44,1×** | 27,2× |
| tương đối thời gian (int8=1) | 1,51 | 1,00 | 1,62 |

**Phát hiện lật số cũ: int4 chậm hơn CẢ fp32** (số cũ bảo int4 25,3× > fp32 22,1× — sai vì
đo không cách ly nhiễu). Giải thích: int8 dynamic chạy MLAS integer GEMM với AVX512-VNNI
(Xeon Icelake) — 1 lệnh 4 MAC int8/lane; MatMulNBits mô phỏng nhân 4-bit bằng LUT + rescale
partial sum từng block 128 → nhiều lệnh hơn hẳn. Lợi ích băng thông (weight 41 vs 69 MB)
không cứu nổi ở model 69M, batch-1. int4 chỉ thắng ở vùng memory-bound model tỉ tham số.

⇒ Khuyến nghị giữ nguyên: deploy int8; int4 chỉ khi size là ràng buộc cứng.

## Vấn đáp: ORT hỗ trợ int4 tốt chưa? (2026-11-16, encoder-only bench)

Phát hiện: bundle int4 cũ có 290 node MatMulNBits đều `accuracy_level=4` (= compute fp32,
chỉ lưu weight 4-bit) — quant lại với al=0 (kernel tự chọn) và al=1 (compute int8) rồi đo
encoder-only (T=1000, 1 thread, ghim core, interleave 3 round, ORT 1.31.1):

| cấu hình | ms/forward | × realtime | tương đối fp32 |
|---|---:|---:|---:|
| fp32 | 289 | 34,6× | 1,00 |
| **int8 dynamic** | **208** | **48,1×** | **0,72** |
| int4 al=4 (bundle cũ) | 324 | 30,9× | 1,12 |
| int4 al=0 | 301 | 33,2× | 1,04 |
| int4 al=1 | 295 | 33,9× | 1,02 |

Kết luận: đổi accuracy_level chỉ cứu ~9%, int4 tốt nhất vẫn chỉ NGANG fp32 và chậm hơn
int8 ~42%. Trên x86 CPU, int4 của ORT (MatMulNBits) là tối ưu DUNG LƯỢNG, không phải
tốc độ: kernel nhanh AVX512-VNNI của nó nhắm M=1 (decode LLM từng token), không phải
encoder ASR (M = hàng trăm frame). Issue theo dõi: microsoft/onnxruntime#23004 (int4
chậm hơn kỳ vọng ≥10×). Ghi chú API: từ ORT 1.31 module đổi tên `matmul_4bits_quantizer`
→ `matmul_nbits_quantizer`, `process()` trả về None (docs web chưa cập nhật).

⇒ Deploy CPU: int8. int4 chỉ khi size là ràng buộc cứng (đổi al=1 để bớt chậm 9% nếu dùng).
