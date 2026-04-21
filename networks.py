import functools as ft
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from equinox.nn._misc import default_init


class EquivariantMLP(eqx.Module):
    layers: list[eqx.Module]
    layers_norm: list[eqx.Module]
    activation: callable

    def __init__(
        self,
        key,
        in_size,
        depth,
        width,
        out_size,
        activation,
        dtype=jnp.float32,
    ):
        layers_norm = []
        layers = []
        if depth == 0:
            key, subkey = jax.random.split(key)
            layers.append(
                EquivariantLinear(
                    in_size,
                    out_size,
                    dtype=dtype,
                    key=subkey,
                )
            )
        else:
            key, subkey = jax.random.split(key)
            layers.append(
                EquivariantLinear(
                    in_size,
                    width,
                    dtype=dtype,
                    key=subkey,
                )
            )
            layers_norm.append(eqx.nn.LayerNorm(width, dtype=dtype))
            for i in range(depth - 1):
                key, subkey = jax.random.split(key)
                layers.append(
                    EquivariantLinear(
                        width,
                        width,
                        dtype=dtype,
                        key=subkey,
                    )
                )
                layers_norm.append(eqx.nn.LayerNorm(width, dtype=dtype))
            key, subkey = jax.random.split(key)
            layers.append(
                EquivariantLinear(
                    width,
                    out_size,
                    dtype=dtype,
                    key=subkey,
                )
            )

        self.layers = layers
        self.layers_norm = layers_norm
        self.activation = activation

    def __call__(self, x, p):
        block_interms = []

        for layer, layer_norm in zip(self.layers[:-1], self.layers_norm):
            x = layer(x, p=p)
            # x = layer_norm(x)
            x = self.activation(x)
            p = layer.q
            block_interms.append(x)

        x = self.layers[-1](x, p=p)
        return x, block_interms


class EquivariantLinear(eqx.Module):
    """Hidden layer: Equivariant to permutation."""

    linear: eqx.nn.Linear
    u: jax.Array
    q: jax.Array

    def __init__(
        self,
        in_features,
        out_features,
        key,
        dtype=jnp.float32,
    ):
        key1, key2 = jax.random.split(key)
        self.linear = eqx.nn.Linear(in_features, out_features, key=key1, dtype=dtype)

        lim = 1 / math.sqrt(out_features)

        self.u = None
        # self.u = default_init(key2, shape=(out_features), dtype=dtype, lim=lim)

        self.q = default_init(
            key2, shape=(out_features, out_features), dtype=dtype, lim=lim
        )

    # @property
    # def q(self):
    #     # q = jnp.eye(self.linear.weight.shape[0]) - jnp.outer(self.u, self.u) / jnp.sum(
    #     #     self.u**2
    #     # )

    #     return 1/2 * (self.)

    # @property
    # def weight(self):
    #     W = self.linear.weight

    #     W_permuted = W[self.q_idx, :][:, self.p_idx]
    #     W_sym = 0.5 * (W + W_permuted)

    #     return W_sym

    # @property
    # def bias(self):
    #     b = self.linear.bias
    #     if b is not None:
    #         b_sym = 0.5 * (b + b[self.q_idx])
    #         return b_sym
    #     else:
    #         return None

    def __call__(self, x, p):
        W = self.linear.weight
        b = self.linear.bias

        # Q = jnp.eye(W.shape[0]) - 2 * jnp.outer(self.u, self.u) / jnp.sum(self.u**2)

        # Q = 0.5 * (self.q + self.q.T)

        Q = self.q

        if p.ndim == 1:
            W_permuted = Q @ W[:, p]
        else:
            W_permuted = Q @ W @ p

        W_sym = 0.5 * (W + W_permuted)

        if b is not None:
            b_sym = 0.5 * (b + Q @ b)
        else:
            b_sym = None

        return W_sym @ x + (b_sym if b_sym is not None else 0)


class InvariantLinear(eqx.Module):
    """Output layer: Invariant to permutation (collapses to a single scalar)."""

    linear: eqx.nn.Linear

    def __init__(self, in_features, out_features, key, dtype=jnp.float32):
        self.linear = eqx.nn.Linear(in_features, out_features, key=key, dtype=dtype)

    def __call__(self, x, p):
        W = self.linear.weight
        b = self.linear.bias

        if p.ndim == 1:
            W_permuted = W[:, self.p_idx]
        else:
            W_permuted = W @ p

        W_sym = 0.5 * (W + W_permuted)

        return W_sym @ x + (b if b is not None else 0)


class ValueNet(eqx.Module):
    body: eqx.nn.MLP
    value_head: eqx.nn.MLP
    avg_symmetries: bool
    name: str

    def __init__(
        self,
        key,
        in_size,
        body_depth,
        body_width,
        embed_dim,
        activation,
        avg_symmetries,
        name=None,
    ):
        random_key, subkey = jax.random.split(key)
        self.value_head = eqx.nn.Linear(
            key=subkey,
            in_features=embed_dim,
            out_features=1,
            dtype=jnp.float32,
        )

        random_key, subkey = jax.random.split(random_key)

        self.body = eqx.nn.MLP(
            key=subkey,
            in_size=in_size,
            out_size=embed_dim,
            depth=body_depth,
            width_size=body_width,
            activation=activation,
        )

        self.avg_symmetries = avg_symmetries

        self.name = self.__class__.__name__
        if self.name is not None:
            self.name = name

    def forward(self, x):
        x1 = x.astype(jnp.float32)
        x2 = jnp.flip(x1, axis=1)

        x1 = jnp.ravel(x1)
        x2 = jnp.ravel(x2)

        o1 = self.body(x1)
        o2 = self.body(x2)
        v1 = jnp.squeeze(self.value_head(o1))
        v2 = jnp.squeeze(self.value_head(o2))
        v = (v1 + v2) / 2

        if self.avg_symmetries:
            return v
        else:
            return v1


class InvariantValueNet(eqx.Module):
    body: eqx.nn.MLP
    value_head: eqx.nn.MLP
    p: jax.Array
    name: str

    def __init__(
        self,
        key,
        in_size,
        body_depth,
        body_width,
        embed_dim,
        activation,
        name=None,
    ):
        random_key = key
        self.p = (
            jnp.flip(jnp.arange(84).reshape(6, 7, 2), axis=1)
            .reshape(-1)
            .astype(jnp.int32)
        )
        random_key, subkey = jax.random.split(random_key)

        self.body = EquivariantMLP(
            key=subkey,
            in_size=in_size,
            out_size=embed_dim,
            depth=body_depth,
            width=body_width,
            activation=activation,
        )

        random_key, subkey = jax.random.split(key)
        self.value_head = InvariantLinear(
            key=subkey, in_features=embed_dim, out_features=1
        )

        self.name = self.__class__.__name__
        if name is not None:
            self.name = name

    def forward(self, x):
        x1 = x.astype(jnp.float32)
        x1 = jnp.ravel(x1)
        o1, _ = self.body(x1, self.p)
        p = self.body.layers[-1].q
        v1 = jnp.squeeze(self.value_head(o1, p=p))

        return v1
