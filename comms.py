"""Cost models for collectives over a ring of N devices.

`bytes_` is always the size of the FULL (logical, unsharded) array.
`link.bandwidth` is one-directional bandwidth along the mesh axis the
collective runs over, in bytes/second.

Ring cost intuition:
  - all-gather: every device must end up with the full array. Each shard
    (bytes_/n) takes n-1 hops around the ring, and the hops pipeline, so
    total time ~ bytes_ * (n-1)/n / bandwidth.
  - reduce-scatter: same traffic pattern in reverse (partial sums travel
    the ring), same cost.
  - all-reduce = reduce-scatter + all-gather = 2x the cost. Note the cost
    is (nearly) independent of n: more devices means more hops but
    proportionally smaller shards.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Link:
    name: str
    bandwidth: float  # bytes/s, one direction

    def __str__(self):
        return f"{self.name} ({self.bandwidth / 1e9:.0f} GB/s)"


# Datasheet value: one direction of one ICI axis.
V5E_ICI = Link("TPU v5e ICI, per axis", 4.5e10)
FAKE_CPU = Link("fake CPU devices (no real interconnect)", float("nan"))

# MEASURED on Kaggle v5e-8 (2x4 torus), 2026-07-01, 256MB float32 arrays:
#   all-reduce:  effective 75.0 GB/s  (1.67x single-axis spec)
#   all-gather:  effective 54.5 GB/s  (1.21x single-axis spec)
# Links are bidirectional and the slice has two torus axes, so XLA routes
# beyond the naive one-direction ring. All-reduce (= reduce-scatter +
# all-gather) gives the compiler more routing freedom, hence the higher
# effective bandwidth. Until modeled properly, predictions should use a
# band of [2x, 1x] the single-axis spec (both measurements land inside).
V5E8_ALLREDUCE_MEASURED = Link("v5e-8 all-reduce, measured 2026-07-01", 7.5e10)
V5E8_ALLGATHER_MEASURED = Link("v5e-8 all-gather, measured 2026-07-01", 5.45e10)


def all_gather_s(bytes_: float, n: int, link: Link) -> float:
    return bytes_ * (n - 1) / n / link.bandwidth


def reduce_scatter_s(bytes_: float, n: int, link: Link) -> float:
    return bytes_ * (n - 1) / n / link.bandwidth


def all_reduce_s(bytes_: float, n: int, link: Link) -> float:
    return 2 * bytes_ * (n - 1) / n / link.bandwidth
