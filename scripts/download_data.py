"""Download COCO val2017 images and annotations into DATA/coco/.

Usage:  uv run python scripts/download_data.py [--force]
"""

import argparse

from objdetect.data.coco import download_coco_val


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="re-download even if files exist"
    )
    args = parser.parse_args()
    download_coco_val(force=args.force)
    print("COCO val2017 ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
