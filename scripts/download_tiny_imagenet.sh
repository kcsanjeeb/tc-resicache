#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Download and prepare Tiny-ImageNet-200
# =============================================================================
# Downloads Tiny-ImageNet-200 into DATA_ROOT and prepares a validation folder in
# ImageFolder format:
#
#   tiny-imagenet-200/
#     train/<wnid>/images/*.JPEG
#     val/<wnid>/*.JPEG              <-- created by this script
#     val_original/images/*.JPEG     <-- original backup
#
# Default:
#   DATA_ROOT=/mnt/data2/data
#
# Usage:
#   cd /mnt/data2
#   nohup env DATA_ROOT=/mnt/data2/data bash /mnt/data2/download_tiny_imagenet.sh \
#     > /mnt/data2/download_tiny_imagenet.log 2>&1 &
# =============================================================================

DATA_ROOT=${DATA_ROOT:-/mnt/data2/data}
TINY_DIR="$DATA_ROOT/tiny-imagenet-200"
ZIP_PATH="$DATA_ROOT/tiny-imagenet-200.zip"

# Known public mirrors. The Stanford CS231n URL is the common official classroom mirror.
URLS=(
  "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
  "https://cs231n.stanford.edu/tiny-imagenet-200.zip"
)

mkdir -p "$DATA_ROOT"

echo "[tiny] DATA_ROOT=$DATA_ROOT"
echo "[tiny] TINY_DIR=$TINY_DIR"

if [[ -d "$TINY_DIR/train" && -f "$TINY_DIR/wnids.txt" ]]; then
  echo "[tiny] Existing Tiny-ImageNet folder found: $TINY_DIR"
else
  if [[ -f "$ZIP_PATH" ]]; then
    echo "[tiny] Existing zip found: $ZIP_PATH"
  else
    echo "[tiny] Downloading Tiny-ImageNet zip..."
    ok=0
    for url in "${URLS[@]}"; do
      echo "[tiny] Trying: $url"
      if command -v wget >/dev/null 2>&1; then
        if wget -O "$ZIP_PATH.tmp" "$url"; then
          mv "$ZIP_PATH.tmp" "$ZIP_PATH"
          ok=1
          break
        fi
      elif command -v curl >/dev/null 2>&1; then
        if curl -L -o "$ZIP_PATH.tmp" "$url"; then
          mv "$ZIP_PATH.tmp" "$ZIP_PATH"
          ok=1
          break
        fi
      else
        echo "[error] Neither wget nor curl is available." >&2
        exit 1
      fi
      rm -f "$ZIP_PATH.tmp"
    done

    if [[ "$ok" -ne 1 ]]; then
      echo "[error] Failed to download Tiny-ImageNet from known URLs." >&2
      echo "[error] Manually download tiny-imagenet-200.zip to: $ZIP_PATH" >&2
      exit 1
    fi
  fi

  echo "[tiny] Checking zip integrity..."
  unzip -t "$ZIP_PATH" >/tmp/tiny_imagenet_ziptest.log 2>&1 || {
    echo "[error] Zip test failed. See /tmp/tiny_imagenet_ziptest.log"
    rm -f "$ZIP_PATH"
    exit 1
  }

  echo "[tiny] Extracting..."
  unzip -q "$ZIP_PATH" -d "$DATA_ROOT"
fi

echo "[tiny] Preparing val ImageFolder layout..."
python3 - "$TINY_DIR" <<'PY'
from pathlib import Path
import shutil
import json
import os
import hashlib

root = Path(__import__("sys").argv[1])
val_dir = root / "val"
ann = val_dir / "val_annotations.txt"
images = val_dir / "images"
manifest = root / "tiny_imagenet_manifest.json"

if not root.exists():
    raise SystemExit(f"Missing root: {root}")
if not ann.exists():
    raise SystemExit(f"Missing annotations: {ann}")

# If already prepared, skip.
prepared_marker = val_dir / ".imagefolder_prepared"
if prepared_marker.exists():
    print("[tiny] val folder already prepared")
else:
    # Back up original val images/annotations once.
    orig = root / "val_original"
    if not orig.exists():
        orig.mkdir(parents=True, exist_ok=True)
        if images.exists():
            shutil.copytree(images, orig / "images")
        shutil.copy2(ann, orig / "val_annotations.txt")

    mapping = {}
    with open(ann, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                mapping[parts[0]] = parts[1]

    if not images.exists():
        # Maybe images already moved into class directories but marker absent.
        class_dirs = [p for p in val_dir.iterdir() if p.is_dir() and p.name != "images"]
        if class_dirs:
            prepared_marker.write_text("prepared\n")
            print("[tiny] val class folders detected; wrote marker")
        else:
            raise SystemExit(f"Missing val images folder: {images}")
    else:
        for img_name, cls in mapping.items():
            src = images / img_name
            if not src.exists():
                continue
            dst_dir = val_dir / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst_dir / img_name))
        try:
            images.rmdir()
        except OSError:
            pass
        prepared_marker.write_text("prepared\n")
        print("[tiny] val folder prepared")

wnids = []
wnids_file = root / "wnids.txt"
if wnids_file.exists():
    wnids = [x.strip() for x in wnids_file.read_text().splitlines() if x.strip()]

num_train = sum(1 for _ in (root / "train").glob("*")) if (root / "train").exists() else 0
num_val_classes = sum(1 for p in val_dir.iterdir() if p.is_dir() and p.name not in ("images",))
num_val_images = sum(1 for p in val_dir.glob("*/*") if p.is_file() and p.suffix.lower() in [".jpeg", ".jpg", ".png"])

data = {
    "root": str(root),
    "wnids_count": len(wnids),
    "train_class_dirs": num_train,
    "val_class_dirs": num_val_classes,
    "val_images": num_val_images,
    "imagefolder_prepared": prepared_marker.exists(),
}
manifest.write_text(json.dumps(data, indent=2), encoding="utf-8")
print(json.dumps(data, indent=2))
PY

echo "[tiny] Done."
echo "[tiny] Manifest:"
cat "$TINY_DIR/tiny_imagenet_manifest.json"
