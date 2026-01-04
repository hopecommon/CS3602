from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class EnvInfo:
    git_commit: Optional[str]
    python: str
    platform: str
    torch: str
    cuda: Optional[str]
    transformers: Optional[str]
    hf_datasets: Optional[str]
    gpu_name: Optional[str]
    driver: Optional[str]
    pg19_sample_file: Optional[str] = None
    pg19_sample_length: Optional[str] = None
    wikitext_sample_file: Optional[str] = None
    wikitext_sample_length: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_version(modname: str) -> Optional[str]:
    try:
        mod = __import__(modname)
        return getattr(mod, "__version__", None)
    except Exception:
        return None


def _git_commit(repo_root: str | None = None) -> Optional[str]:
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        out = subprocess.check_output(cmd, cwd=repo_root, stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def _nvidia_smi_driver() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").splitlines()[0].strip()
    except Exception:
        return None


def collect_env_info(repo_root: str | None = None) -> EnvInfo:
    import torch  # local import to avoid importing torch for non-torch tooling

    gpu_name = None
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = None

    return EnvInfo(
        git_commit=_git_commit(repo_root=repo_root),
        python=platform.python_version(),
        platform=platform.platform(),
        torch=getattr(torch, "__version__", "unknown"),
        cuda=getattr(torch.version, "cuda", None),
        transformers=_safe_version("transformers"),
        hf_datasets=_safe_version("datasets"),
        gpu_name=gpu_name,
        driver=_nvidia_smi_driver(),
        pg19_sample_file=os.environ.get("PG19_SAMPLE_FILE"),
        pg19_sample_length=os.environ.get("PG19_SAMPLE_LENGTH"),
        wikitext_sample_file=os.environ.get("WIKITEXT_SAMPLE_FILE"),
        wikitext_sample_length=os.environ.get("WIKITEXT_SAMPLE_LENGTH"),
    )


def env_compatible(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    """
    Conservative compatibility check for reusing baselines across machines/runs.

    We require exact match on torch+cuda+gpu_name; other fields are informative only.
    """
    if not a or not b:
        return False
    for k in ("torch", "cuda", "gpu_name"):
        if a.get(k) != b.get(k):
            return False
    return True
