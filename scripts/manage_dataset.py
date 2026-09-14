"""Dataset management and standardization script for Forensic Signature AI.

Standardizes dataset files, resolves folder collisions, validates integrity,
and provides utilities for synchronizing with the training pipelines.
"""

import os
import re
import sys
import argparse
from collections import defaultdict
from pathlib import Path


def clean_junk_files(extract_dir: Path) -> int:
    """Remove OS temporary files such as desktop.ini and Thumbs.db."""
    deleted_count = 0
    for root, _, files in os.walk(extract_dir):
        for f in files:
            if f.lower() in ["desktop.ini", "thumbs.db", ".ds_store"]:
                file_path = Path(root) / f
                try:
                    file_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}", file=sys.stderr)
    return deleted_count


def natural_sort_key(filename: str):
    """Sort filenames naturally by embedded integer sequences."""
    nums = [int(n) for n in re.findall(r"\d+", filename)]
    return (nums, filename)


def restructure_dataset(extract_dir: Path) -> dict:
    """Standardize folder structure and rename all images in extract directory.

    Naming convention:
        Genuine:  extract/{signer:03d}/original_{signer}_{sample}.jpg
        Forgery:  extract/{signer:03d}_forg/forgeries_{signer}_{sample}.jpg

    Resolves collision in folder 180 by allocating Batch 2 to signer 687.
    """
    clean_junk_files(extract_dir)

    # 1. Resolve collision in folder 180 if still mixed
    dir_180_gen = extract_dir / "180"
    dir_180_forg = extract_dir / "180_forg"
    dir_687_gen = extract_dir / "687"
    dir_687_forg = extract_dir / "687_forg"

    if dir_180_gen.exists():
        files_180_gen = [f for f in os.listdir(dir_180_gen) if not f.endswith(".ini")]
        b2_gen = [f for f in files_180_gen if "-" not in f and not f.startswith("original_")]
        if b2_gen:
            dir_687_gen.mkdir(parents=True, exist_ok=True)
            for f in b2_gen:
                src = dir_180_gen / f
                dst = dir_687_gen / f
                os.replace(src, dst)

    if dir_180_forg.exists():
        files_180_forg = [f for f in os.listdir(dir_180_forg) if not f.endswith(".ini")]
        b2_forg = [f for f in files_180_forg if "-" not in f and not f.startswith("forgeries_")]
        if b2_forg:
            dir_687_forg.mkdir(parents=True, exist_ok=True)
            for f in b2_forg:
                src = dir_180_forg / f
                dst = dir_687_forg / f
                os.replace(src, dst)

    # 2. Iterate through all signer folders and standardize filenames
    total_renamed = 0
    subdirs = sorted(os.listdir(extract_dir))

    for d in subdirs:
        dp = extract_dir / d
        if not dp.is_dir():
            continue

        is_forg = d.endswith("_forg")
        signer_str = d.replace("_forg", "")
        if not signer_str.isdigit():
            continue
        signer_id = int(signer_str)

        files = [f for f in os.listdir(dp) if not f.endswith(".ini")]
        files.sort(key=natural_sort_key)

        prefix = "forgeries" if is_forg else "original"

        # Check if files already match target format
        already_standard = all(
            re.match(rf"^{prefix}_{signer_id}_\d+\.jpg$", f) for f in files
        )
        if already_standard:
            continue

        # Two-pass renaming with temporary prefix to prevent any collision
        temp_renames = []
        for idx, old_name in enumerate(files, start=1):
            ext = os.path.splitext(old_name)[1].lower() or ".jpg"
            temp_name = f"__tmp_{prefix}_{signer_id}_{idx}{ext}"
            src = dp / old_name
            dst = dp / temp_name
            os.replace(src, dst)
            temp_renames.append((temp_name, f"{prefix}_{signer_id}_{idx}.jpg"))

        for temp_name, final_name in temp_renames:
            src = dp / temp_name
            dst = dp / final_name
            os.replace(src, dst)
            total_renamed += 1

    return {"total_renamed": total_renamed}


def validate_dataset(extract_dir: Path) -> dict:
    """Validate naming and distribution across extract directory."""
    subdirs = sorted(os.listdir(extract_dir))
    genuine_dirs = [d for d in subdirs if not d.endswith("_forg") and d.isdigit()]
    forg_dirs = [d for d in subdirs if d.endswith("_forg") and d.replace("_forg", "").isdigit()]

    total_genuine_images = 0
    total_forgery_images = 0
    invalid_files = []

    for d in genuine_dirs:
        signer_id = int(d)
        dp = extract_dir / d
        for f in os.listdir(dp):
            if re.match(rf"^original_{signer_id}_\d+\.jpg$", f):
                total_genuine_images += 1
            else:
                invalid_files.append((d, f))

    for d in forg_dirs:
        signer_id = int(d.replace("_forg", ""))
        dp = extract_dir / d
        for f in os.listdir(dp):
            if re.match(rf"^forgeries_{signer_id}_\d+\.jpg$", f):
                total_forgery_images += 1
            else:
                invalid_files.append((d, f))

    return {
        "genuine_signers": len(genuine_dirs),
        "forgery_signers": len(forg_dirs),
        "total_genuine_images": total_genuine_images,
        "total_forgery_images": total_forgery_images,
        "total_images": total_genuine_images + total_forgery_images,
        "invalid_files_count": len(invalid_files),
        "invalid_files_sample": invalid_files[:5],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage Forensic Signature Dataset")
    parser.add_argument(
        "--extract-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "extract"),
        help="Path to extract directory",
    )
    parser.add_argument(
        "--action",
        choices=["restructure", "validate"],
        default="restructure",
        help="Action to perform",
    )

    args = parser.parse_args()
    target_dir = Path(args.extract_dir)

    if not target_dir.exists():
        print(f"Error: Directory {target_dir} not found.", file=sys.stderr)
        sys.exit(1)

    if args.action == "restructure":
        print(f"Standardizing dataset in {target_dir}...")
        res = restructure_dataset(target_dir)
        print(f"Restructure completed. Files renamed: {res['total_renamed']}")

    stats = validate_dataset(target_dir)
    print("\n=== Dataset Validation Statistics ===")
    print(f"Genuine Signers:       {stats['genuine_signers']}")
    print(f"Forgery Signers:       {stats['forgery_signers']}")
    print(f"Total Genuine Images:  {stats['total_genuine_images']}")
    print(f"Total Forgery Images:  {stats['total_forgery_images']}")
    print(f"Total Images:          {stats['total_images']}")
    print(f"Invalid Filenames:     {stats['invalid_files_count']}")
    if stats["invalid_files_count"] > 0:
        print(f"Samples of invalid files: {stats['invalid_files_sample']}")
        sys.exit(1)
    else:
        print("Verification: SUCCESS - All files match standard forensic naming!")
