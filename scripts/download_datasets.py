#!/usr/bin/env python3
"""
Script to manage and download Vietnamese speech datasets for Tele + Sales + Medical STT.
Usage:
    python3 download_datasets.py --list
    python3 download_datasets.py --dataset OneVoice
    python3 download_datasets.py --group active
"""

import os
import sys
import argparse
from huggingface_hub import snapshot_download

DATASET_SPECS = {
    # --- Y tế (Medical) ---
    "VietMed-Label": {
        "repo_id": "doof-ferb/VietMed_labeled",
        "size_gb": 0.17,
        "hours": 16,
        "domain": "Medical (Bệnh án, lâm sàng, tên thuốc)",
        "group": "active",
    },
    "MultiMED": {
        "repo_id": "leduckhai/MultiMed-ST",
        "size_gb": 0.18,
        "hours": 16,
        "allow_patterns": ["vietnamese/*", "README.md"],
        "domain": "Medical (Khám chữa bệnh lâm sàng)",
        "group": "active",
    },
    "VietMDD": {
        "repo_id": "doof-ferb/VietMDD",
        "size_gb": 0.58,
        "hours": 5,
        "domain": "Medical / Pronunciation (Ngữ âm học, chẩn đoán)",
        "group": "active",
    },
    # --- Telecom & Sales / Call Center ---
    "OneVoice": {
        "repo_id": "nhatminh/OneVoice",
        "size_gb": 3.18,
        "hours": 40,
        "domain": "Telecom & Sales (Tổng đài CSKH, hội thoại Agent-Customer)",
        "group": "active",
    },
    "gigaspeech2_vie": {
        "repo_id": "doof-ferb/gigaspeech2_vie",
        "size_gb": 3.26,
        "hours": 21,
        "domain": "Telecom & Sales (Nói tự nhiên, tạp âm môi trường)",
        "group": "active",
    },
    "ViMD_Dataset": {
        "repo_id": "nguyendv02/ViMD_Dataset",
        "size_gb": 55.72,
        "hours": 100,
        "domain": "Telecom & Sales (Đa phương ngữ Bắc - Trung - Nam)",
        "group": "large",
    },
    # --- Nền tảng cốt lõi (Core Acoustic Base) ---
    "FOSD": {
        "repo_id": "doof-ferb/fpt_fosd",
        "size_gb": 0.76,
        "hours": 100,
        "domain": "General Base (Câu lệnh & giao tiếp FPT)",
        "group": "active",
    },
    "vivos": {
        "repo_id": "AILAB-VNUHCM/vivos",
        "size_gb": 1.37,
        "hours": 15,
        "domain": "General Base (Phòng thu 16kHz sạch, câu ngắn)",
        "group": "active",
    },
    "LSVSC": {
        "repo_id": "doof-ferb/LSVSC",
        "size_gb": 10.78,
        "hours": 100,
        "domain": "General Base (Quy mô lớn, cân bằng 3 miền)",
        "group": "active",
    },
    "vlsp2020": {
        "repo_id": "doof-ferb/vlsp2020_vinai_100h",
        "size_gb": 10.85,
        "hours": 100,
        "domain": "General Base (Đọc báo, tin tức chuẩn VLSP)",
        "group": "active",
    },
}

def download_dataset(name, dest_dir):
    spec = DATASET_SPECS[name]
    repo_id = spec["repo_id"]
    local_dir = os.path.join(dest_dir, name)
    allow_patterns = spec.get("allow_patterns")
    
    print(f"\n==========================================")
    print(f"Downloading [{name}] ({repo_id}) ~ {spec['size_gb']} GB")
    print(f"Domain: {spec['domain']}")
    print(f"Target: {local_dir}")
    print(f"==========================================")
    
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=allow_patterns,
    )
    print(f"[SUCCESS] Downloaded {name} to {local_dir}")

def main():
    default_dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    parser = argparse.ArgumentParser(description="Download Vietnamese ASR datasets for Tele+Sales+Medical")
    parser.add_argument("--dest", default=default_dest, help="Target folder")
    parser.add_argument("--list", action="store_true", help="List all datasets and sizes")
    parser.add_argument("--group", choices=["active", "large", "all"], help="Download a preset group")
    parser.add_argument("--dataset", nargs="+", help="Download specific dataset names")
    
    args = parser.parse_args()
    
    if args.list:
        print(f"{'Name':<18} {'Size (GB)':<10} {'Hours':<7} {'Group':<8} {'Domain'}")
        print("-" * 80)
        for k, v in DATASET_SPECS.items():
            print(f"{k:<18} {v['size_gb']:<10.2f} {v['hours']:<7} {v['group']:<8} {v['domain']}")
        return

    targets = []
    if args.dataset:
        for d in args.dataset:
            if d in DATASET_SPECS:
                targets.append(d)
            else:
                print(f"Warning: {d} not recognized. Skipping.")
    elif args.group == "active":
        targets = [k for k, v in DATASET_SPECS.items() if v["group"] == "active"]
    elif args.group == "large":
        targets = [k for k, v in DATASET_SPECS.items() if v["group"] == "large"]
    elif args.group == "all":
        targets = list(DATASET_SPECS.keys())
    else:
        parser.print_help()
        return

    os.makedirs(args.dest, exist_ok=True)
    total_gb = sum(DATASET_SPECS[t]["size_gb"] for t in targets)
    print(f"Selected {len(targets)} datasets. Total expected size: {total_gb:.2f} GB")
    
    for t in targets:
        try:
            download_dataset(t, args.dest)
        except Exception as e:
            print(f"[ERROR] Failed to download {t}: {e}")

if __name__ == "__main__":
    main()
