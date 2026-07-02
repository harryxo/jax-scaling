"""Spec sheet for every accelerator we can reach from a browser.

TODO(verify-on-hardware): all numbers are from public datasheets and must be
re-checked in Act 1 / Act 5 before anything is recorded. bf16 FLOPs unless
noted. hbm_bw is bytes/s.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Chip:
    name: str
    peak_flops: float   # bf16 FLOPs/s
    hbm_bytes: float
    hbm_bw: float       # bytes/s

    @property
    def crossover_batch(self) -> float:
        """Arithmetic intensity the chip needs to be compute-bound (Act 1)."""
        return self.peak_flops / self.hbm_bw


# Colab / Kaggle menu, July 2026
T4 = Chip("NVIDIA T4", 65e12, 16e9, 3.0e11)          # fp16
L4 = Chip("NVIDIA L4", 121e12, 24e9, 3.0e11)
A100 = Chip("NVIDIA A100 40GB", 312e12, 40e9, 1.6e12)
H100 = Chip("NVIDIA H100", 990e12, 80e9, 3.35e12)
V5E = Chip("TPU v5e", 197e12, 16e9, 8.2e11)
V6E = Chip("TPU v6e (Trillium)", 918e12, 32e9, 1.6e12)
