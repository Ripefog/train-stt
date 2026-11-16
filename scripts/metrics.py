#!/usr/bin/env python3
"""Tính WER / CER / TER cho các JSONL do eval_beam.py sinh ra — KHÔNG decode lại.

eval_beam.py đã ghi ref + hyp của từng mode cho từng cut, nên mọi metric mức
văn bản đều tính lại được offline. Không cần GPU, không nạp checkpoint.

Ba metric:
  WER  micro = (S+D+I)/N từ, jiwer.wer_standardize — khớp số eval_beam đã in.
  CER  micro = như trên nhưng đơn vị KÝ TỰ, ĐÃ BỎ khoảng trắng. Bỏ space vì
       tiếng Việt viết rời từng âm tiết: giữ space thì ~1/6 ký tự là space và
       nó gần như luôn đúng, làm CER bị pha loãng xuống thấp giả tạo.
  TER  = tỷ lệ LỖI thuật ngữ = (số lần thuật ngữ xuất hiện trong ref mà hyp
       không có) / (tổng số lần xuất hiện) = 1 − recall thuật ngữ.

TER tính theo HAI cách, và cách thứ hai mới là cách đúng với production:
  exact : hyp phải ra ĐÚNG chuỗi trong word_mapping.json ("ép pi ti").
  equiv : hyp ra BẤT KỲ biến thể nào cùng canonical cũng tính đúng ("ép bê tê",
          "ép bi ti"...). Đúng vì sau ASR, transcribe.py:220 _load_hotword_map()
          đọc bảng MySQL hotword_entries rồi map mọi biến thể về canonical, nên
          chúng cho ra cùng một kết quả cuối. Nhóm biến thể đọc từ
          hotword_variants.tsv (dump của bảng đó).

Biến thể <= MIN_VARIANT_LEN ký tự bị bỏ (ví dụ "ép", "ếp" của FPT): quá ngắn nên
khớp bừa vào từ Việt thường, gây dương tính giả.
"""
import argparse
import collections
import json
import re
import unicodedata
from pathlib import Path

MIN_VARIANT_LEN = 4

# Thuật ngữ RIÊNG của FPT Telecom: tên thương hiệu + tên sản phẩm Ftel bán.
# Tách khỏi 87 thuật ngữ còn lại vì phần lớn số đó là từ tin học mượn tiếng Anh
# (router, download, reset...) mà model nền đã nghe được sẵn — chúng làm loãng
# con số. Đây mới là nhóm cả đợt finetune này nhắm vào.
# Khoá theo KEY của word_mapping.json (script tự tra sang dạng đọc).
FTEL_KEYS = [
    "fpt", "ép pê tê",            # -> ép pi ti
    "hifpt", "hai ép pê tê",      # -> hai ép pi ti
    "gic", "gi ai xi",            # -> gờ i xê
    "gigaeyes",                   # -> gi ga ai
    "skyeyes",                    # -> sờ kai ai
    "speedx",                     # -> sờ pít ích
    "hyperfast",                  # -> hai pơ phát
    "ultrafast",                  # -> un tra phát
]


def norm(s: str) -> str:
    """lowercase + NFC + gộp khoảng trắng. Không bỏ dấu: dấu là thông tin thật."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s).lower()).strip()


def load_terms(wm_path: Path):
    """Các value NHIỀU ÂM TIẾT của word_mapping.json = thuật ngữ cần đo.

    Bỏ value một âm tiết vì chúng trùng từ Việt thường ("không" khớp 380k cut).
    Trả về (danh sách thuật ngữ, dict thuật ngữ -> key canonical).
    """
    wm = json.loads(wm_path.read_text(encoding="utf-8"))["words"]
    term2keys = collections.defaultdict(set)
    for k, v in wm.items():
        if k.startswith("_") or " " not in v:
            continue
        # MỌI key trỏ về cùng value, không chỉ key đầu: "fpt" và "ép pê tê" cùng
        # ra "ép pi ti", mỗi key có thể khớp một canonical khác trong DB.
        term2keys[norm(v)].add(norm(k))
    ftel = {norm(wm[k]) for k in FTEL_KEYS if k in wm and " " in wm[k]}
    return sorted(term2keys, key=len, reverse=True), dict(term2keys), ftel


def load_groups(tsv_path: Path, wm_path: Path, term2key: dict, min_len: int):
    """thuật ngữ -> id nhóm tương đương, và id nhóm -> mọi chuỗi trong nhóm.

    id nhóm = chính dạng CHUẨN (value của word_mapping), không phải canonical của
    DB — nhờ vậy nhóm vẫn dựng được cho cụm mà hotword_entries chưa có dòng nào
    (HiFPT, GIC...).

    HAI nguồn biến thể:
      1. hotword_entries (hotword_variants.tsv), ghép theo canonical: key
         word_mapping "fpt" == cột canonical "FPT" sau lowercase.
      2. CHÍNH các key của word_mapping: file này là map biến-thể-đọc -> dạng
         chuẩn, nên "ép pê tê" -> "ép pi ti" biến key thành một biến thể hợp lệ.
         Nguồn này độc lập với DB và phủ được cụm DB còn thiếu.

    min_len: bỏ biến thể quá ngắn. "ép"/"ếp" (biến thể của FPT) trùng động từ
    tiếng Việt "ép" nên khớp bừa; đặt min_len=0 để đo xem nó ảnh hưởng bao nhiêu.
    """
    group2strs = {t: {t} for t in term2key}

    wm = json.loads(wm_path.read_text(encoding="utf-8"))["words"]
    for k, v in wm.items():
        if k.startswith("_") or " " not in v:
            continue
        kk, vv = norm(k), norm(v)
        if kk != vv and vv in group2strs and len(kk) >= min_len:
            group2strs[vv].add(kk)

    if tsv_path.is_file():
        canon2variants, n_drop = {}, 0
        for line in tsv_path.read_text(encoding="utf-8").splitlines():
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 3:
                continue
            canon, map_safe, raw = cols[0], cols[1] == "1", cols[2]
            try:
                variants = json.loads(raw)
            except json.JSONDecodeError:
                continue
            keep = []
            for it in variants:
                # Đúng logic decode_phonetic_pairs(): chuỗi trần kế thừa map_safe
                # của entry, dict {"p":..,"safe":..} tự ghi đè.
                if isinstance(it, dict):
                    ph, safe = it.get("p", ""), bool(it.get("safe", map_safe))
                else:
                    ph, safe = it, map_safe
                # CHỈ biến thể safe mới được _apply_hotword_mapping thay mù.
                # Biến thể không safe đẩy sang _llm_fix_asr_terms — LLM có sửa
                # đúng hay không thì không đo được ở đây, nên KHÔNG tính là hit.
                if safe and ph:
                    keep.append(norm(ph))
                else:
                    n_drop += 1
            canon2variants[norm(canon)] = keep
        if n_drop:
            print(f"bỏ {n_drop} biến thể map_safe=0 (prod không thay mù, đẩy cho LLM)")
        for term, keys in term2key.items():
            for key in keys:
                for v in canon2variants.get(key, []):
                    if len(v) >= min_len:
                        group2strs[term].add(v)

    group2strs = {g: sorted(s, key=len, reverse=True)
                  for g, s in group2strs.items() if len(s) > 1}
    # Một chuỗi thuộc 2 nhóm thì phép đo sai lặng lẽ (đếm sang nhóm khác) —
    # phải báo, đừng để dict comprehension ghi đè âm thầm.
    owner = {}
    for g, ss in group2strs.items():
        for s in ss:
            owner.setdefault(s, []).append(g)
    clash = {s: gs for s, gs in owner.items() if len(gs) > 1}
    if clash:
        print(f"⚠️  {len(clash)} chuỗi thuộc nhiều nhóm, đã gán vào nhóm đầu:")
        for s, gs in list(clash.items())[:10]:
            print(f"      {s!r} -> {gs}")
    term2group = {s: gs[0] for s, gs in owner.items()}
    return term2group, group2strs


def counts(rx, text):
    return collections.Counter(rx.findall(text)) if rx else collections.Counter()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+", help="file eval_beam.py đã ghi")
    ap.add_argument("--modes", default="greedy,beam16,beam16+hw,beam16+hw+bp1",
                    help="danh sách mode, phẩy phân tách; 'all' = mọi mode trong file")
    ap.add_argument("--word-mapping", default="/app/word_mapping.json")
    ap.add_argument("--variants", default="/app/hotword_variants.tsv")
    ap.add_argument("--label", default="", help="nhãn cho từng file, phẩy phân tách")
    ap.add_argument("--min-variant-len", type=int, default=MIN_VARIANT_LEN,
                    help="bỏ biến thể ngắn hơn N ký tự (0 = nhận HẾT, kể cả 'ép' "
                         "trùng động từ tiếng Việt nên dễ dương tính giả)")
    args = ap.parse_args()

    import jiwer

    terms, term2key, ftel_terms = load_terms(Path(args.word_mapping))
    term2group, group2strs = load_groups(
        Path(args.variants), Path(args.word_mapping), term2key,
        args.min_variant_len)

    rx_exact = re.compile(r"(?<!\w)(" + "|".join(re.escape(t) for t in terms) + r")(?!\w)")
    all_strs = sorted(set(terms) | set(term2group), key=len, reverse=True)
    rx_any = re.compile(r"(?<!\w)(" + "|".join(re.escape(t) for t in all_strs) + r")(?!\w)")

    labels = [x.strip() for x in args.label.split(",")] if args.label else []
    rows = []
    for fi, fp in enumerate(args.jsonl):
        lines = Path(fp).read_text(encoding="utf-8").splitlines()
        head = json.loads(lines[0])
        cuts = [json.loads(x) for x in lines[1:]]
        avail = list(head.get("_report", {}))
        modes = avail if args.modes == "all" else [
            m.strip() for m in args.modes.split(",") if m.strip()]
        label = labels[fi] if fi < len(labels) else Path(fp).stem

        refs = [norm(c["ref"]) for c in cuts]
        # ref chỉ đếm bằng rx_exact: thuật ngữ trong ref là dạng chuẩn của
        # word_mapping (transcript train đã normalize), biến thể chỉ có ở hyp.
        ref_terms = [counts(rx_exact, r) for r in refs]
        n_term = sum(sum(c.values()) for c in ref_terms)
        n_ftel = sum(n for c in ref_terms for t, n in c.items() if t in ftel_terms)

        for mo in modes:
            key = "hyp_" + mo
            if key not in cuts[0]:
                print(f"  ⚠️  {label}: không có mode {mo!r} (có: {avail})")
                continue
            hyps = [norm(c[key]) for c in cuts]

            w = jiwer.process_words(refs, hyps,
                                    reference_transform=jiwer.wer_standardize,
                                    hypothesis_transform=jiwer.wer_standardize)
            nw = w.substitutions + w.deletions + w.hits

            rc = [r.replace(" ", "") for r in refs]
            hc = [h.replace(" ", "") for h in hyps]
            ch = jiwer.process_characters(rc, hc)
            nc = ch.substitutions + ch.deletions + ch.hits

            hit_x = hit_e = ftel_hit = 0
            per_term = collections.Counter()
            for rt, hy in zip(ref_terms, hyps):
                if not rt:
                    continue
                cx = counts(rx_exact, hy)
                ca = counts(rx_any, hy)
                # equiv: gộp mọi chuỗi cùng nhóm về một khoá trước khi so
                cg = collections.Counter()
                for s, n in ca.items():
                    cg[term2group.get(s, s)] += n
                for t, n in rt.items():
                    hit_x += min(n, cx.get(t, 0))
                    he = min(n, cg.get(term2group.get(t, t), 0))
                    hit_e += he
                    if t in ftel_terms:
                        ftel_hit += he
                        per_term[t] += he

            rows.append(dict(
                label=label, mode=mo, nw=nw, nc=nc,
                wer=w.wer * 100, cer=ch.cer * 100,
                ter_x=(n_term - hit_x) / max(n_term, 1) * 100,
                ter_e=(n_term - hit_e) / max(n_term, 1) * 100,
                ftel=ftel_hit / max(n_ftel, 1) * 100,
                ftel_hit=ftel_hit, n_ftel=n_ftel,
                hit_x=hit_x, hit_e=hit_e, n_term=n_term, per_term=per_term))

    print(f"\n{len(terms)} thuật ngữ (word_mapping nhiều âm tiết), "
          f"{len(group2strs)} nhóm có biến thể trong hotword_entries "
          f"(bỏ biến thể < {args.min_variant_len} ký tự)\n")
    # HAI cột recall: exact và equiv. Trước đây bảng chỉ in exact trong khi chú
    # thích ngay dưới lại ghi "cột chính là equiv" — hit_e vẫn được tính nhưng
    # không bao giờ hiện ra, nên mọi số equiv phải tính tay ở ngoài.
    h = (f"{'checkpoint':<12s} {'mode':<15s} {'thuật ngữ exact':>18s} "
         f"{'thuật ngữ equiv':>18s} {'WER':>7s} {'CER':>7s}")
    print(h)
    print("-" * (len(h) - 2))
    for r in rows:
        print(f"{r['label']:<12s} {r['mode']:<15s} "
              f"{r['hit_x']:4d}/{r['n_term']} = {100 - r['ter_x']:5.1f}% "
              f"{r['hit_e']:4d}/{r['n_term']} = {100 - r['ter_e']:5.1f}% "
              f"{r['wer']:6.2f}% {r['cer']:6.2f}%")
    if rows:
        r = rows[0]
        print(f"\nmẫu: {r['nw']} từ, {r['nc']} ký tự (đã bỏ space), "
              f"{r['n_term']} lần xuất hiện thuật ngữ, trong đó {r['n_ftel']} "
              f"lần là thuật ngữ RIÊNG Ftel")
        print("Hai cột recall TÍNH TRÊN CẢ 87 cụm; TER = 100 − recall. exact = khớp "
              "đúng chuỗi word_mapping; equiv = mọi biến thể cùng canonical tính "
              "đúng (vì transcribe.py map hết về canonical sau ASR).")
        print("**Cột quyết định là EXACT** (chốt 2026-08-13): equiv phụ thuộc bảng "
              "hotword_entries và --min-variant-len, hai thứ nằm ngoài model, nên "
              "không so được giữa các lần đo cách xa nhau — đã có lần đổi tham số làm "
              "lật dấu 3 kết luận. equiv giữ lại làm tham chiếu: nó xấp xỉ thứ khách "
              "thực nhận sau bước map của prod, cao hơn exact ~9-12 pp.")
        print("recall Ftel = recall theo equiv, CHỈ trên nhóm thương hiệu/sản phẩm "
              f"Ftel: {sorted(ftel_terms & set().union(*(set(c) for c in ref_terms)))}")
        print("\nchi tiết nhóm Ftel (recall equiv từng cụm):")
        ref_ftel = collections.Counter()
        for c in ref_terms:
            for t, n in c.items():
                if t in ftel_terms:
                    ref_ftel[t] += n
        for t, tot in ref_ftel.most_common():
            cells = " ".join(f"{r['label']}/{r['mode']}={r['per_term'][t]}"
                             for r in rows)
            print(f"  {t:<16} ref {tot:3d}   {cells}")


if __name__ == "__main__":
    main()

# Chạy:
#   R=/home/anlnm/lmq/train_zipformer_5080
#   docker run --rm -v "$R/scripts:/app/scripts:ro" -v "$R/dump_val:/dv:ro" \
#     -v "$R/word_mapping.json:/app/word_mapping.json:ro" \
#     -v "$R/hotword_variants.tsv:/app/hotword_variants.tsv:ro" \
#     --entrypoint python3 zipformer-training:latest /app/scripts/metrics.py \
#     /dv/valbp1_ep35.jsonl /dv/valbp1_B_ep40.jsonl --label gipformer,B_ep40
