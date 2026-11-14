from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KERNEL_DIR = ROOT / "kaggle" / "fascicle_unetpp_384"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "kaggle" / "fascicle_unetpp_384"


def metadata(kernel_dir: Path) -> dict:
    path = kernel_dir / "kernel-metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))


def kaggle_command(*arguments: str, check: bool = True) -> subprocess.CompletedProcess:
    command = [kaggle_executable(), *arguments]
    return subprocess.run(command, check=check, text=True)


def kaggle_executable() -> str:
    executable = ROOT / ".venv" / "bin" / "kaggle"
    return str(executable) if executable.exists() else "kaggle"


def kernel_ref(kernel_dir: Path) -> str:
    return str(metadata(kernel_dir)["id"])


def get_status(kernel_dir: Path, check: bool = True) -> subprocess.CompletedProcess:
    return kaggle_command("kernels", "status", kernel_ref(kernel_dir), check=check)


def wait_for_kernel(kernel_dir: Path, interval: int) -> int:
    while True:
        status_result = subprocess.run(
            [
                kaggle_executable(),
                "kernels",
                "status",
                kernel_ref(kernel_dir),
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        status_text = (status_result.stdout + status_result.stderr).strip()
        print(f"[{time.strftime('%H:%M:%S')}] {status_text}", flush=True)
        lowered = status_text.lower()
        if "complete" in lowered:
            return 0
        if "error" in lowered or "cancel" in lowered or "failed" in lowered:
            return 1
        time.sleep(interval)


def main() -> int:
    parser = argparse.ArgumentParser(description="Push and monitor the UMUD Kaggle GPU kernel.")
    parser.add_argument(
        "action", choices=["push", "status", "logs", "follow", "wait", "download"]
    )
    parser.add_argument("--kernel-dir", type=Path, default=DEFAULT_KERNEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    kernel_dir = args.kernel_dir.resolve()
    reference = kernel_ref(kernel_dir)

    if args.action == "push":
        return kaggle_command(
            "kernels",
            "push",
            "--path",
            str(kernel_dir),
            "--accelerator",
            "NvidiaTeslaT4",
        ).returncode
    if args.action == "status":
        return get_status(kernel_dir).returncode
    if args.action in {"logs", "follow"}:
        command = ["kernels", "logs"]
        if args.action == "follow":
            command.append("--follow")
        command.append(reference)
        return kaggle_command(*command).returncode
    if args.action == "wait":
        return wait_for_kernel(kernel_dir, args.interval)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    return kaggle_command(
        "kernels", "output", reference, "--path", str(args.output_dir), "--force"
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
