from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path


COMPETITION = "umud-challenge-muscle-architecture-in-ultrasound-data"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--competition", default=COMPETITION)
    parser.add_argument("--keep-zip", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    zip_path = args.out / f"{args.competition}.zip"
    extract_dir = args.out / args.competition

    if not zip_path.exists() and not extract_dir.exists():
        cmd = [
            sys.executable,
            "-m",
            "kaggle",
            "competitions",
            "download",
            "-c",
            args.competition,
            "-p",
            str(args.out),
        ]
        subprocess.run(cmd, check=True)

    if zip_path.exists() and not extract_dir.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)

    if zip_path.exists() and not args.keep_zip:
        zip_path.unlink()

    print(f"Data directory: {extract_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

