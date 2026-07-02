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


# TODO(verify-on-hardware): datasheet values; Act 5b measures these for real.
V5E_ICI = Link("TPU v5e ICI, per axis", 4.5e10)
FAKE_CPU = Link("fake CPU devices (no real interconnect)", float("nan"))


def all_gather_s(bytes_: float, n: int, link: Link) -> float:
    return bytes_ * (n - 1) / n / link.bandwidth


def reduce_scatter_s(bytes_: float, n: int, link: Link) -> float:
    return bytes_ * (n - 1) / n / link.bandwidth


def all_reduce_s(bytes_: float, n: int, link: Link) -> float:
    return 2 * bytes_ * (n - 1) / n / link.bandwidth
