import math

import equinox as eqx
import jax
import jax.numpy as jnp


@eqx.filter_jit
def get_batches(data, random_key, batch_size, drop_last):
    if not drop_last:
        raise NotImplementedError

    n_samples = jax.tree.leaves(data)[0].shape[0]
    n_batches = math.floor(n_samples / batch_size)

    idx = jax.random.permutation(random_key, jnp.arange(n_samples))[
        : n_batches * batch_size
    ]
    return jax.tree.map(
        lambda x: x[idx].reshape(n_batches, batch_size, *x.shape[1:]), data
    )
