"""Three parallelism strategies as sharding annotations — nothing else.

The entire difference between data parallelism, FSDP, and tensor parallelism
in this codebase is which PartitionSpec each parameter leaf gets. The model
code and training step never change. XLA inserts the collectives.

  dp:    replicate params, shard the batch.
         Comm: all-reduce of gradients each step.
  fsdp:  shard params AND batch on the same axis.
         Comm: all-gather params before use, reduce-scatter gradients.
  tp:    split the matmuls themselves (Megatron-style): column-shard the
         in-projections, row-shard the out-projections.
         Comm: all-reduce activations inside every layer.
"""

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

AXIS = "x"

# Megatron pairing: column-parallel producers feed row-parallel consumers,
# so the pair needs only one all-reduce on the way out.
_TP_SPECS = {
    "wq": P(None, AXIS), "wk": P(None, AXIS), "wv": P(None, AXIS), "w1": P(None, AXIS),
    "wo": P(AXIS, None), "w2": P(AXIS, None),
    "embed": P(None, AXIS), "pos": P(None, AXIS),
}


def _leaf_name(path):
    return path[-1].key if hasattr(path[-1], "key") else str(path[-1])


def param_specs(strategy: str, params, n_devices: int):
    """PartitionSpec pytree matching `params` for the given strategy."""
    def spec(path, leaf):
        name = _leaf_name(path)
        if strategy == "dp":
            return P()
        if strategy == "fsdp":
            # Shard the leading axis wherever it divides the mesh.
            return P(AXIS) if leaf.shape[0] % n_devices == 0 else P()
        if strategy == "tp":
            return _TP_SPECS.get(name, P())
        raise ValueError(strategy)

    return jax.tree_util.tree_map_with_path(spec, params)


def batch_spec(strategy: str):
    # tp keeps the batch whole: it parallelizes within each matmul instead.
    return P() if strategy == "tp" else P(AXIS)


def shard(params, strategy: str, mesh: Mesh):
    """device_put the params according to the strategy. This one call is the
    whole 'implementation' of the strategy."""
    specs = param_specs(strategy, params, mesh.devices.size)
    return jax.tree.map(
        lambda x, s: jax.device_put(x, NamedSharding(mesh, s)), params, specs)
