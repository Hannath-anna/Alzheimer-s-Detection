"""
balance_dataset.py — Alzheimer MRI Dataset Balancer
=====================================================
Creates a balanced copy of dataset/train/ → dataset_balanced/train/

Strategy:
  - OVERSAMPLE minority classes by generating augmented images
  - UNDERSAMPLE the dominant class (Non_Demented) to a cap
  - TARGET: ~8 000 images per class (configurable via TARGET_PER_CLASS)
  - val/ and test/ are copied as-is (no balancing — keep eval honest)

Run BEFORE training on the balanced set:
    python balance_dataset.py
Then update DATASET_DIR in train_cnn.py to "dataset_balanced"
"""

import os
import shutil
import random
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator, img_to_array, array_to_img, load_img
)
from tqdm import tqdm

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
SRC_DIR       = "dataset"
DST_DIR       = "dataset_balanced"
TARGET_PER_CLASS = 8_000       # Target images per class in train split
IMG_SIZE      = (224, 224)
SEED          = 42
random.seed(SEED)
np.random.seed(SEED)

# Augmentation pipeline used for oversampling
aug = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.12,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
)


def get_image_files(folder):
    """Return sorted list of image file paths in folder."""
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in exts
    ]
    return sorted(files)


def augment_and_save(src_files, dst_folder, n_needed, prefix="aug"):
    """
    Generate n_needed augmented images from src_files and save to dst_folder.
    Cycles through src_files repeatedly if needed.
    """
    os.makedirs(dst_folder, exist_ok=True)
    saved = 0
    src_cycle = src_files[:]
    random.shuffle(src_cycle)
    idx = 0

    with tqdm(total=n_needed, desc=f"  Augmenting → {os.path.basename(dst_folder)}", leave=False) as pbar:
        while saved < n_needed:
            img_path = src_cycle[idx % len(src_cycle)]
            idx += 1

            try:
                img = load_img(img_path, target_size=IMG_SIZE)
                arr = img_to_array(img)
                arr = arr.reshape((1,) + arr.shape)

                for batch in aug.flow(arr, batch_size=1, seed=SEED + saved):
                    aug_img = array_to_img(batch[0])
                    out_name = f"{prefix}_{saved:06d}.jpg"
                    aug_img.save(os.path.join(dst_folder, out_name), quality=95)
                    saved += 1
                    pbar.update(1)
                    break   # Only 1 augmented image per source per iteration
            except Exception as e:
                print(f"    ⚠️  Skipped {img_path}: {e}")

            if saved >= n_needed:
                break


def copy_files(src_files, dst_folder, max_count=None):
    """Copy up to max_count files from src_files into dst_folder."""
    os.makedirs(dst_folder, exist_ok=True)
    files_to_copy = src_files[:max_count] if max_count else src_files
    for f in files_to_copy:
        shutil.copy2(f, dst_folder)


def balance_split(src_split, dst_split, target_per_class, oversample=True, undersample=True):
    """
    Process one split (train/val/test).
    For train: oversample minority + undersample majority.
    For val/test: copy files as-is (no resampling — keep evaluation honest).
    """
    os.makedirs(dst_split, exist_ok=True)
    class_folders = sorted([
        d for d in os.listdir(src_split)
        if os.path.isdir(os.path.join(src_split, d))
    ])

    print(f"\n  Processing split: {os.path.basename(src_split)}")
    counts = {}
    for cls in class_folders:
        files = get_image_files(os.path.join(src_split, cls))
        counts[cls] = len(files)
        print(f"    {cls:<25} {len(files):>6} images")

    for cls in class_folders:
        src_cls   = os.path.join(src_split, cls)
        dst_cls   = os.path.join(dst_split, cls)
        src_files = get_image_files(src_cls)
        n_existing = len(src_files)

        if not oversample and not undersample:
            # val/test: copy everything as-is
            copy_files(src_files, dst_cls)
            continue

        if n_existing >= target_per_class:
            # Undersample: randomly pick target_per_class files
            selected = random.sample(src_files, target_per_class)
            copy_files(selected, dst_cls)
            print(f"    ✂️  {cls:<25} undersampled {n_existing} → {target_per_class}")
        else:
            # Copy all originals
            copy_files(src_files, dst_cls)
            # Oversample the deficit
            n_needed = target_per_class - n_existing
            print(f"    🔼 {cls:<25} oversampling +{n_needed} (has {n_existing}, target {target_per_class})")
            augment_and_save(src_files, dst_cls, n_needed, prefix=f"aug_{cls[:4]}")

    # Verify
    print(f"\n  ✅ Verification — {os.path.basename(dst_split)}:")
    total = 0
    for cls in class_folders:
        n = len(get_image_files(os.path.join(dst_split, cls)))
        print(f"    {cls:<25} {n:>6} images")
        total += n
    print(f"    {'TOTAL':<25} {total:>6} images")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  Alzheimer Dataset Balancer")
    print(f"  Source      : {SRC_DIR}/")
    print(f"  Destination : {DST_DIR}/")
    print(f"  Target/class: {TARGET_PER_CLASS:,} (train only)")
    print("=" * 60)

    # Remove old balanced dataset if it exists
    if os.path.exists(DST_DIR):
        print(f"\n🗑️  Removing existing {DST_DIR}/ ...")
        shutil.rmtree(DST_DIR)

    # ── Train split: balance
    balance_split(
        src_split=os.path.join(SRC_DIR, 'train'),
        dst_split=os.path.join(DST_DIR, 'train'),
        target_per_class=TARGET_PER_CLASS,
        oversample=True,
        undersample=True,
    )

    # ── Val split: copy as-is (honest evaluation)
    print("\n📋 Copying val/ as-is...")
    val_src = os.path.join(SRC_DIR, 'val')
    val_dst = os.path.join(DST_DIR, 'val')
    if os.path.exists(val_src):
        shutil.copytree(val_src, val_dst)
        print(f"   Copied {val_src} → {val_dst}")

    # ── Test split: copy as-is
    print("\n📋 Copying test/ as-is...")
    test_src = os.path.join(SRC_DIR, 'test')
    test_dst = os.path.join(DST_DIR, 'test')
    if os.path.exists(test_src):
        shutil.copytree(test_src, test_dst)
        print(f"   Copied {test_src} → {test_dst}")

    print("\n" + "=" * 60)
    print(f"✅ Done! Balanced dataset saved to: {DST_DIR}/")
    print()
    print("Next steps:")
    print("  1. Open train_cnn.py")
    print(f'  2. Change:  DATASET_DIR = "dataset"')
    print(f'     To:      DATASET_DIR = "dataset_balanced"')
    print("  3. Run:     python train_cnn.py")
    print("=" * 60)
