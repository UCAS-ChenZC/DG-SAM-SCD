import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image


def read_mask(mask_path):
    """
    读取标签图。
    如果是单通道标签图，返回 H x W。
    如果是 RGB 标签图，返回 H x W x C。
    """
    return np.array(Image.open(mask_path))


def calc_change_ratio(mask1, mask2):
    """
    计算 label1 和 label2 的像素变化比例。
    支持单通道 mask 和 RGB 彩色 mask。
    """
    if mask1.shape != mask2.shape:
        raise ValueError(f"Shape mismatch: {mask1.shape} vs {mask2.shape}")

    # 单通道标签图：H x W
    if mask1.ndim == 2:
        diff = mask1 != mask2
        total_pixels = mask1.shape[0] * mask1.shape[1]

    # RGB/RGBA 标签图：H x W x C
    elif mask1.ndim == 3:
        diff = np.any(mask1 != mask2, axis=-1)
        total_pixels = mask1.shape[0] * mask1.shape[1]

    else:
        raise ValueError(f"Unsupported mask dimension: {mask1.ndim}")

    change_pixels = np.count_nonzero(diff)
    change_ratio = change_pixels / total_pixels

    return change_ratio, change_pixels, total_pixels


def delete_file(path, dry_run=True):
    if path.exists():
        if dry_run:
            print(f"[DRY-RUN] Delete: {path}")
        else:
            path.unlink()
            print(f"[DELETE] {path}")
    else:
        print(f"[WARNING] File not found: {path}")


def main(args):
    root = Path(args.root)

    im1_dir = root / "im1"
    im2_dir = root / "im2"
    label1_dir = root / "label1"
    label2_dir = root / "label2"

    required_dirs = [im1_dir, im2_dir, label1_dir, label2_dir]
    for d in required_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Folder not found: {d}")

    label1_files = sorted(label1_dir.glob("*.png"))

    delete_count = 0
    keep_count = 0
    error_count = 0

    print(f"Dataset root: {root}")
    print(f"Threshold: {args.threshold * 100:.2f}%")
    print(f"Dry run: {args.dry_run}")
    print("-" * 80)

    for label1_path in label1_files:
        filename = label1_path.name
        label2_path = label2_dir / filename

        if not label2_path.exists():
            print(f"[WARNING] Missing label2: {label2_path}")
            error_count += 1
            continue

        try:
            mask1 = read_mask(label1_path)
            mask2 = read_mask(label2_path)

            change_ratio, change_pixels, total_pixels = calc_change_ratio(mask1, mask2)

        except Exception as e:
            print(f"[ERROR] {filename}: {e}")
            error_count += 1
            continue

        if change_ratio <= args.threshold:
            print(
                f"[LOW-CHANGE] {filename} | "
                f"change_ratio={change_ratio * 100:.4f}% | "
                f"{change_pixels}/{total_pixels}"
            )

            delete_file(im1_dir / filename, dry_run=args.dry_run)
            delete_file(im2_dir / filename, dry_run=args.dry_run)
            delete_file(label1_dir / filename, dry_run=args.dry_run)
            delete_file(label2_dir / filename, dry_run=args.dry_run)

            delete_count += 1
        else:
            keep_count += 1

    print("-" * 80)
    print(f"Finished.")
    print(f"Keep samples: {keep_count}")
    print(f"Delete samples: {delete_count}")
    print(f"Error samples: {error_count}")

    if args.dry_run:
        print("\n当前是 dry-run 模式，并没有真正删除文件。")
        print("确认无误后，加上 --delete 参数执行真正删除。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="数据集根目录，里面应包含 im1, im2, label1, label2 四个文件夹"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0,
        help="变化像素比例阈值，默认 0.05 表示 5%"
    )

    parser.add_argument(
        "--delete",
        action="store_true",
        help="真正删除文件。不加该参数时只预览，不删除"
    )

    args = parser.parse_args()

    # 默认 dry-run；只有加 --delete 才真正删除
    args.dry_run = not args.delete

    main(args)