import functools as ft

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


class DenseResNetBlock(eqx.Module):
    layers: list[eqx.Module]
    layers_norm: list[eqx.Module]
    activation: callable

    def __init__(self, key, depth, width_size, activation, linear_module=None):
        if linear_module is None:
            linear_module = eqx.nn.Linear

        keys = jax.random.split(key, depth)
        self.layers = []

        self.layers_norm = []

        for i in range(depth):
            layer = linear_module(
                width_size,
                width_size,
                key=keys[i],
                dtype=jnp.float32,
            )
            layer_norm = eqx.nn.LayerNorm(width_size, dtype=jnp.float32)
            self.layers.append(layer)
            self.layers_norm.append(layer_norm)

        self.activation = activation

    def __call__(self, x):
        identity = x
        block_interms = []
        for layer, layer_norm in zip(self.layers, self.layers_norm):
            x = layer(x)
            x = layer_norm(x)
            x = self.activation(x)
            block_interms.append(x)
            
        return x + identity, block_interms


class DenseResNet(eqx.Module):
    input_layer: eqx.Module
    blocks: list[DenseResNetBlock]
    output_layer: eqx.Module
    activation: callable

    def __init__(
        self,
        key,
        in_size,
        out_size,
        depth,
        width_size,
        n_blocks,
        activation,
        linear_module=None,
    ):
        if linear_module is None:
            linear_module = eqx.nn.Linear
        keys = jax.random.split(key, n_blocks + 2)
        self.input_layer = linear_module(
            in_size,
            width_size,
            key=keys[0],
            dtype=jnp.float32,
        )

        self.blocks = []
        for i in range(n_blocks):
            block = DenseResNetBlock(
                keys[i + 1], depth, width_size, activation, linear_module=linear_module
            )
            self.blocks.append(block)

        self.output_layer = linear_module(
            width_size,
            out_size,
            key=keys[-1],
            dtype=jnp.float32,
        )

        self.activation = activation

    def __call__(self, x):
        x = self.input_layer(x)
        x = self.activation(x)
        first = x
        interms = []
        blocks_interms = []
        for block in self.blocks:
            x, block_interms = block(x)
            interms.append(x)
            blocks_interms.append(block_interms)
    
        x = self.output_layer(x)
        last = x
        return x, interms, blocks_interms, first, last


class EquivariantLinear(eqx.Module):
    """Hidden layer: Equivariant to permutation."""

    linear: eqx.nn.Linear
    p_idx: jax.Array
    q_idx: jax.Array

    def __init__(self, in_features, out_features, p_idx, q_idx, key, dtype=jnp.float32):
        self.linear = eqx.nn.Linear(in_features, out_features, key=key, dtype=dtype)
        self.p_idx = jnp.asarray(p_idx, dtype=jnp.int32)
        self.q_idx = jnp.asarray(q_idx, dtype=jnp.int32)

    @property
    def weight(self):
        W = self.linear.weight
        W_permuted = W[self.q_idx, :][:, self.p_idx]
        W_sym = 0.5 * (W + W_permuted)
        return W_sym

    @property
    def bias(self):
        b = self.linear.bias
        if b is not None:
            b_sym = 0.5 * (b + b[self.q_idx])
            return b_sym
        else:
            return None

    def __call__(self, x):
        W = self.linear.weight
        b = self.linear.bias

        # Apply the reflection constraint: W_sym = 0.5 * (W + Q W P)
        # In array indexing: Q is applied to rows (outputs), P to cols (inputs)
        W_permuted = W[self.q_idx, :][:, self.p_idx]
        W_sym = 0.5 * (W + W_permuted)

        # Apply constraint to bias: b_sym = 0.5 * (b + Q b)
        if b is not None:
            b_sym = 0.5 * (b + b[self.q_idx])
        else:
            b_sym = None

        return W_sym @ x + (b_sym if b_sym is not None else 0)


class InvariantLinear(eqx.Module):
    """Output layer: Invariant to permutation (collapses to a single scalar)."""

    linear: eqx.nn.Linear
    p_idx: jax.Array

    def __init__(self, in_features, out_features, p_idx, key, dtype=jnp.float32):
        # Output feature is 1 (the value evaluation)
        self.linear = eqx.nn.Linear(in_features, out_features, key=key, dtype=dtype)
        self.p_idx = jnp.asarray(p_idx, dtype=jnp.int32)

    def __call__(self, x):
        W = self.linear.weight
        b = self.linear.bias

        # For a scalar output, Q is the identity matrix.
        # W_sym = 0.5 * (W + W P)
        W_permuted = W[:, self.p_idx]
        W_sym = 0.5 * (W + W_permuted)

        # Bias is a scalar, so it doesn't need to be permuted
        return W_sym @ x + (b if b is not None else 0)



class ValueHead(eqx.Module):
    layers: list[eqx.Module]
    activation: callable

    def __init__(
        self,
        key,
        in_size,
        head_depth,
        head_width,
        activation,
        dtype=jnp.float32,
    ):

        self.layers = []
        for k in range(head_depth):
            random_key, subkey = jax.random.split(key)

            l = eqx.nn.Linear(
                in_size if k == 0 else head_width,
                head_width if k < head_depth - 1 else 1,
                key=subkey,
                dtype=dtype,
            )
            self.layers.append(l)

        self.activation = activation

    def __call__(self, x):
        head_interms = []
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
            head_interms.append(x)
        x = self.layers[-1](x)
        return x, head_interms
    
class ValueNet(eqx.Module):
    body: eqx.nn.MLP
    value_head: eqx.nn.MLP
    avg_symmetries: bool

    def __init__(
        self,
        key,
        in_size,
        head_depth,
        head_width,
        body_depth,
        body_width,
        body_n_blocks,
        embed_dim,
        activation,
        n_actions,
        avg_symmetries,
        simple,
    ):
        random_key, subkey = jax.random.split(key)

        self.value_head = ValueHead(
            subkey,
            in_size=embed_dim,
            head_depth=head_depth,
            head_width=head_width,
            activation=activation,
        )
        
        random_key, subkey = jax.random.split(random_key)
        if simple:
            self.body = eqx.nn.MLP(
                key=subkey,
                in_size=in_size,
                out_size=embed_dim,
                depth=body_depth,
                width_size=body_width,
                activation=activation,
            )
        else:
            self.body = DenseResNet(
                subkey,
                in_size=in_size,
                out_size=embed_dim,
                depth=body_depth,
                width_size=body_width,
                n_blocks=body_n_blocks,
                activation=activation,
            )

        self.avg_symmetries = avg_symmetries

    def forward(self, x):
        x1 = x.astype(jnp.float32)
        x2 = jnp.flip(x1, axis=1)

        x1 = jnp.ravel(x1)
        x2 = jnp.ravel(x2)

        o1, interms1, blocks_interms1, first1, last1 = self.body(x1)
        o2, interms2, blocks_interms2, first2, last2 = self.body(x2)
        
        v1, head_interms1 = self.value_head(o1)
        v1 = jnp.squeeze(v1)
        v2, head_interms2 = self.value_head(o2)
        v2 = jnp.squeeze(v2)
        
        v = (v1 + v2) / 2
        
        if self.avg_symmetries:
            # return v, (interms1, interms2), (blocks_interms1, blocks_interms2), (first1, first2), (last1, last2)
            return v, interms1, blocks_interms1, first1, last1, head_interms1
        else:
            return v1, interms1, blocks_interms1, first1, last1, head_interms1


class InvariantValueHead(eqx.Module):
    layers: list[eqx.Module]
    activation: callable

    def __init__(
        self,
        key,
        in_size,
        head_depth,
        head_width,
        activation,
        p_idx,
        dtype=jnp.float32,
    ):
        p_idx = jnp.asarray(p_idx, dtype=jnp.int32)

        self.layers = []
        for k in range(head_depth):
            random_key, subkey = jax.random.split(key)

            l = InvariantLinear(
                in_size if k == 0 else head_width,
                head_width if k < head_depth - 1 else 1,
                p_idx=p_idx,
                key=subkey,
                dtype=dtype,
            )
            self.layers.append(l)

        self.activation = activation

    def __call__(self, x):
        head_interms = []
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
            head_interms.append(x)
        x = self.layers[-1](x)
        return x, head_interms


class InvariantValueNet(eqx.Module):
    body: eqx.nn.MLP
    value_head: eqx.nn.MLP
    avg_symmetries: bool

    def __init__(
        self,
        key,
        in_size,
        head_depth,
        head_width,
        body_depth,
        body_width,
        body_n_blocks,
        embed_dim,
        activation,
        n_actions,
        avg_symmetries,
        simple,
    ):
        p_idx = np.flip(np.arange(84).reshape(6, 7, 2), axis=1).reshape(-1)
        q_idx = jnp.arange(embed_dim)[::-1]
        # q_idx = jnp.arange(embed_dim).reshape(-1, 2)[:, ::-1].reshape(-1)
        random_key, subkey = jax.random.split(key)
        self.value_head = InvariantValueHead(
            subkey,
            in_size=embed_dim,
            head_depth=head_depth,
            head_width=head_width,
            activation=activation,
            p_idx=q_idx,
        )
        random_key, subkey = jax.random.split(random_key)

        linear_module = ft.partial(EquivariantLinear, p_idx=q_idx, q_idx=q_idx)
        if simple:
            self.body = eqx.nn.MLP(
                key=subkey,
                in_size=in_size,
                out_size=embed_dim,
                depth=body_depth,
                width_size=body_width,
                activation=activation,
            )
        else:
            self.body = DenseResNet(
                subkey,
                in_size=in_size,
                out_size=embed_dim,
                depth=body_depth,
                width_size=body_width,
                n_blocks=body_n_blocks,
                activation=activation,
                linear_module=linear_module,
            )
            random_key, subkey = jax.random.split(random_key)
            input_layer = EquivariantLinear(
                in_size,
                embed_dim,
                p_idx=p_idx,
                q_idx=q_idx,
                key=subkey,
                dtype=jnp.float32,
            )
            self.body = eqx.tree_at(lambda m: m.input_layer, self.body, input_layer)
        self.avg_symmetries = avg_symmetries

    def forward(self, x):
        x1 = x.astype(jnp.float32)

        x2 = jnp.flip(x1, axis=1)

        x1 = jnp.ravel(x1)
        x2 = jnp.ravel(x2)

        o1, interms1, blocks_interms1, first1, last1 = self.body(x1)
        o2, interms2, blocks_interms2, first2, last2 = self.body(x2)
        
        v1, head_interms1 = self.value_head(o1)
        v1 = jnp.squeeze(v1)
        
        v2, head_interms2 = self.value_head(o2)
        v2 = jnp.squeeze(v2)
        
        v = (v1 + v2) / 2
        
        if self.avg_symmetries:
            # return v, (interms1, interms2), (blocks_interms1, blocks_interms2), (first1, first2), (last1, last2)
            return v, interms1, blocks_interms1, first1, last1, head_interms1
        else:
            return v1, interms1, blocks_interms1, first1, last1, head_interms1

