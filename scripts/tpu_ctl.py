"""Create / SSH / sync / delete the LedgerLite v5e-8 TPU from Windows or Linux."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_GCLOUD_PY = Path(r"C:\Program Files (x86)\Google\Cloud SDK\google-cloud-sdk\lib\gcloud.py")
if os.name == "nt" and _GCLOUD_PY.exists():
    GCLOUD_CMD: list[str] = [sys.executable, str(_GCLOUD_PY)]
else:
    bin_ = (
        os.environ.get("GCLOUD")
        or shutil.which("gcloud.cmd")
        or shutil.which("gcloud")
        or r"C:\Program Files (x86)\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
    )
    GCLOUD_CMD = [bin_]

PROJECT = os.environ.get("PROJECT_ID", "tpu-builder1")
ZONE = os.environ.get("ZONE", "us-west4-a")
TPU_NAME = os.environ.get("TPU_NAME", "ledgerlite-sft-v5e8")
ACCEL = os.environ.get("ACCEL", "v5litepod-8")
RUNTIME = os.environ.get("RUNTIME", "v2-alpha-tpuv5-lite")
REMOTE_DIR = os.environ.get("REMOTE_DIR", "/home/Tinevimbo/qx-foundational-model")
VENV_PY = "/home/Tinevimbo/qx-venv/bin/python"


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    resolved = [*GCLOUD_CMD, *cmd[1:]] if cmd and cmd[0] == "gcloud" else cmd
    print("+", " ".join(resolved), flush=True)
    return subprocess.run(resolved, check=check)


def tpu_ssh(command: str, check: bool = True) -> subprocess.CompletedProcess:
    return run(
        [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "ssh",
            TPU_NAME,
            f"--zone={ZONE}",
            f"--project={PROJECT}",
            f"--command={command}",
        ],
        check=check,
    )


def create() -> None:
    run(
        [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "create",
            TPU_NAME,
            f"--zone={ZONE}",
            f"--project={PROJECT}",
            f"--accelerator-type={ACCEL}",
            f"--version={RUNTIME}",
        ]
    )


def status() -> None:
    run(
        [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "describe",
            TPU_NAME,
            f"--zone={ZONE}",
            f"--project={PROJECT}",
        ],
        check=False,
    )


def setup() -> None:
    tpu_ssh("curl -LsSf https://astral.sh/uv/install.sh | sh")
    tpu_ssh(
        "/home/Tinevimbo/.local/bin/uv python install 3.12"
    )
    tpu_ssh("mkdir -p /home/Tinevimbo/qx-foundational-model/scripts")
    _scp(
        ROOT / "scripts" / "tpu_bootstrap.sh",
        "/home/Tinevimbo/qx-foundational-model/scripts/tpu_bootstrap.sh",
    )
    tpu_ssh("bash /home/Tinevimbo/qx-foundational-model/scripts/tpu_bootstrap.sh")


def _scp(local: Path, remote: str) -> None:
    run(
        [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "scp",
            "--recurse",
            str(local),
            f"{TPU_NAME}:{remote}",
            f"--zone={ZONE}",
            f"--project={PROJECT}",
        ]
    )


def sync() -> None:
    data = ROOT / "data" / "sft"
    if not data.exists():
        raise SystemExit("data/sft missing — run python scripts/prepare_sft.py first")
    tpu_ssh(f"mkdir -p {REMOTE_DIR}/data {REMOTE_DIR}/src {REMOTE_DIR}/scripts {REMOTE_DIR}/configs")
    _scp(ROOT / "src", f"{REMOTE_DIR}/")
    _scp(ROOT / "scripts", f"{REMOTE_DIR}/")
    _scp(ROOT / "configs", f"{REMOTE_DIR}/")
    if (ROOT / "pyproject.toml").exists():
        _scp(ROOT / "pyproject.toml", f"{REMOTE_DIR}/")
    _scp(data, f"{REMOTE_DIR}/data/")
    tpu_ssh(f"find {REMOTE_DIR}/scripts -name '*.sh' -exec sed -i 's/\\r$//' {{}} +", check=False)


def train(config: str) -> None:
    import yaml
    remote_cfg = config.replace("\\", "/")
    cfg_path = ROOT / config
    exp_dir = f"{REMOTE_DIR}/outputs/exp"
    if cfg_path.exists():
        try:
            with cfg_path.open(encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                out = cfg.get("output_dir", "outputs/exp").replace("\\", "/")
                exp_dir = f"{REMOTE_DIR}/{out}"
        except Exception:
            pass
    log = f"{exp_dir}/train.log"
    tpu_ssh(f"mkdir -p {exp_dir}")
    tpu_ssh(
        f"cd {REMOTE_DIR} && PYTHONPATH=src nohup {VENV_PY} -u scripts/train_sft.py "
        f"--config {remote_cfg} > {log} 2>&1 < /dev/null & echo STARTED"
    )


def pull(exp: str) -> None:
    dest = ROOT / "outputs" / exp
    dest.mkdir(parents=True, exist_ok=True)
    run(
        [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "scp",
            "--recurse",
            f"{TPU_NAME}:{REMOTE_DIR}/outputs/{exp}",
            str(ROOT / "outputs") + "/",
            f"--zone={ZONE}",
            f"--project={PROJECT}",
        ]
    )


def delete() -> None:
    run(
        [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "delete",
            TPU_NAME,
            f"--zone={ZONE}",
            f"--project={PROJECT}",
            "--quiet",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=["create", "status", "setup", "sync", "train", "pull", "delete", "ssh"],
    )
    parser.add_argument("arg", nargs="?")
    args = parser.parse_args()
    if args.action == "create":
        create()
    elif args.action == "status":
        status()
    elif args.action == "setup":
        setup()
    elif args.action == "sync":
        sync()
    elif args.action == "train":
        if not args.arg:
            raise SystemExit("train requires a config path")
        train(args.arg)
    elif args.action == "pull":
        if not args.arg:
            raise SystemExit("pull requires an exp name")
        pull(args.arg)
    elif args.action == "delete":
        delete()
    elif args.action == "ssh":
        cmd = args.arg or "bash -l"
        tpu_ssh(cmd)


if __name__ == "__main__":
    main()
