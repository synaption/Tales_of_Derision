"""What is this machine, and how big a model can it hold?

Fruit Brains is meant to run on a Raspberry Pi *and* on a desktop with a big
GPU, so the model is chosen from what it finds rather than hardcoded.  Detection
is deliberately cheap and failure-tolerant: anything it can't work out falls
back to the CPU ladder, which is the safe direction to be wrong in.

``FRUITBRAINS_MODEL`` always wins over any of this.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


# (minimum VRAM in MB, ollama tag). GPU first, biggest first.
GPU_LADDER = (
    (22000, "qwen2.5:14b"),
    (10000, "qwen2.5:7b"),
    (6000, "qwen2.5:3b"),
    (3500, "qwen2.5:1.5b"),
)
# (minimum system RAM in MB, ollama tag) for CPU-only machines -- a Pi lands here.
CPU_LADDER = (
    (16000, "qwen2.5:3b"),
    (8000, "qwen2.5:1.5b"),
    (3000, "qwen2.5:0.5b"),
    (0, "smollm2:360m"),
)


@dataclass(frozen=True)
class Hardware:
    gpu: str = ""            # human-readable GPU name, "" if none usable
    vram_mb: int = 0
    ram_mb: int = 0
    pi: bool = False

    @property
    def label(self) -> str:
        if self.gpu:
            return f"{self.gpu} ({self.vram_mb / 1024:.0f} GB VRAM)"
        machine = "Raspberry Pi" if self.pi else f"CPU {platform.machine()}"
        return f"{machine} ({self.ram_mb / 1024:.0f} GB RAM)"


def _run(command: list[str], timeout: float = 3.0) -> str:
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        return done.stdout if done.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _nvidia() -> tuple[str, int]:
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits"])
    for line in out.splitlines():
        name, _, mem = line.partition(",")
        digits = re.sub(r"[^0-9]", "", mem)
        if name.strip() and digits:
            return name.strip(), int(digits)
    return "", 0


def _amd() -> tuple[str, int]:
    best = 0
    for node in Path("/sys/class/drm").glob("card*/device/mem_info_vram_total"):
        try:
            best = max(best, int(node.read_text().strip()) // (1024 * 1024))
        except (OSError, ValueError):
            continue
    return ("AMD GPU", best) if best else ("", 0)


def _apple() -> tuple[str, int]:
    """Apple Silicon shares memory with the GPU; call half of it usable."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return "", 0
    return "Apple Silicon (Metal)", _ram_mb() // 2


def _ram_mb() -> int:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(re.sub(r"[^0-9]", "", line)) // 1024
    except OSError:
        pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // (1024 * 1024)
    except (ValueError, OSError, AttributeError):
        return 0


def _is_pi() -> bool:
    for path in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        try:
            if "raspberry pi" in Path(path).read_text(errors="replace").lower():
                return True
        except OSError:
            continue
    try:
        return "raspberry pi" in Path("/proc/cpuinfo").read_text(errors="replace").lower()
    except OSError:
        return False


def probe() -> Hardware:
    """Best-effort look at the machine. Never raises."""
    for detect in (_nvidia, _amd, _apple):
        name, vram = detect()
        if name and vram:
            return Hardware(name, vram, _ram_mb(), _is_pi())
    return Hardware("", 0, _ram_mb(), _is_pi())


def recommend_model(hw: Hardware | None = None) -> str:
    """Biggest model this machine can hold comfortably."""
    hw = probe() if hw is None else hw
    if hw.gpu:
        for floor, tag in GPU_LADDER:
            if hw.vram_mb >= floor:
                return tag
    for floor, tag in CPU_LADDER:
        if hw.ram_mb >= floor:
            return tag
    return CPU_LADDER[-1][1]
