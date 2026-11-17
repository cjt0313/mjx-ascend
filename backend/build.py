#!/usr/bin/env python3
"""Prepare pinned sources and build the ACLNN PJRT plugin without system changes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run(*args: str, cwd: Path | None = None, env: dict | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, env=env, text=True).strip()


def verify(source: Path) -> None:
    hashes = json.loads((ROOT / "source-sha256.json").read_text())
    invalid = [
        name
        for name, expected in hashes.items()
        if not (source / name).is_file()
        or hashlib.sha256((source / name).read_bytes()).hexdigest() != expected
    ]
    if invalid:
        raise RuntimeError(f"Source verification failed: {invalid}")
    print(f"Verified {len(hashes)} pinned source files")


def prepare(work: Path, pins: dict) -> Path:
    for name in ("xla_npu", "xla"):
        dest = work / name
        pin = pins[name]
        if not dest.exists():
            subprocess.run(["git", "clone", "--no-checkout", pin["url"], str(dest)], check=True)
            subprocess.run(
                ["git", "-C", str(dest), "checkout", "--detach", pin["commit"]], check=True
            )
        if run("git", "rev-parse", "HEAD", cwd=dest) != pin["commit"]:
            raise RuntimeError(f"Unexpected revision in {dest}; use a new work directory")
    source = work / "xla_npu"
    patch = ROOT / "xla-npu.patch"
    applied = subprocess.run(
        ["git", "apply", "--reverse", "--check", str(patch)],
        cwd=source,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if applied.returncode:
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=source, check=True)
        subprocess.run(["git", "apply", str(patch)], cwd=source, check=True)
    verify(source)
    xla = work / "xla"
    if run("git", "status", "--porcelain", "--untracked-files=no", cwd=xla):
        raise RuntimeError("XLA tracked sources contain local changes")
    link = xla / "xla" / "xla_npu"
    if not link.exists() and not link.is_symlink():
        link.symlink_to(source / "xla_npu", target_is_directory=True)
    if link.resolve() != (source / "xla_npu").resolve():
        raise RuntimeError(f"Unexpected source link: {link}")
    return xla


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="Persistent directory for sources, build/cache and output",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--bazel", default="bazel", help="Bazel 7.7.0 executable")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if sys.version_info[:2] != (3, 11):
        parser.error("Python 3.11 is required")
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    pins = json.loads((ROOT / "versions.json").read_text())
    xla = prepare(work, pins)
    if args.prepare_only:
        return
    if run(args.bazel, "--version") != "bazel " + pins["bazel"]:
        parser.error("Bazel 7.7.0 is required")
    java = subprocess.run(["java", "-version"], capture_output=True, text=True, check=True)
    if '"21' not in java.stderr + java.stdout:
        parser.error("JDK 21 is required; configure JAVA_HOME and PATH")
    env = dict(os.environ)
    for key, dirname in (
        ("TMPDIR", "tmp"),
        ("XDG_CACHE_HOME", "cache"),
        ("ASCEND_CACHE_PATH", "ascend-cache"),
        ("ASCEND_WORK_PATH", "ascend-logs"),
    ):
        path = work / dirname
        path.mkdir(exist_ok=True)
        env[key] = str(path)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run([sys.executable, "configure.py", "--backend=CPU"], cwd=xla, env=env, check=True)
    import sysconfig

    libdir = sysconfig.get_config_var("LIBDIR")
    command = [
        args.bazel,
        f"--output_user_root={work / 'bazel-user'}",
        f"--output_base={work / 'bazel-output'}",
        f"--host_jvm_args=-Djava.io.tmpdir={env['TMPDIR']}",
        "build",
        f"--repository_cache={work / 'repository-cache'}",
        f"--symlink_prefix={work / 'links'}/",
        "--incompatible_disallow_empty_glob=false",
        "--remote_download_minimal",
        "--spawn_strategy=local",
        "--verbose_failures",
        "--copt=-w",
        f"--jobs={args.jobs}",
        "--define=PYTHON_VERSION=3.11",
        f"--define=PYTHON_LIBDIR={libdir}",
        "--define=NPU_AUTO_FUSE_BACKEND=0",
        f"--action_env=PYTHON_BIN_PATH={sys.executable}",
        f"--python_path={sys.executable}",
        f"--repo_env=TMPDIR={env['TMPDIR']}",
        f"--action_env=TMPDIR={env['TMPDIR']}",
        "//xla/xla_npu/pjrt/c:pjrt_c_api_npu_plugin.so",
    ]
    subprocess.run(command, cwd=xla, env=env, check=True)
    output = work / "output"
    output.mkdir(exist_ok=True)
    plugin = output / "pjrt_c_api_npu_plugin.so"
    shutil.copy2(
        work
        / "bazel-output/execroot/xla/bazel-out/k8-opt/bin/xla/xla_npu/pjrt/c/pjrt_c_api_npu_plugin.so",
        plugin,
    )
    verify(work / "xla_npu")
    print(f"Plugin: {plugin}\nSHA256: {hashlib.sha256(plugin.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
