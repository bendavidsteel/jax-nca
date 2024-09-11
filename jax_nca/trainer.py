import functools
import os
from collections.abc import Iterable
from datetime import datetime
import operator

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import tqdm
from flax.training import train_state  # Useful dataclass to keep train state
import flaxmodels as fm
from tensorboardX import SummaryWriter


from jax_nca.utils import make_circle_masks


def get_tensorboard_logger(
    experiment_name: str, base_log_path: str = "tensorboard_logs"
):
    log_path = "{}/{}_{}".format(base_log_path, experiment_name, datetime.now())
    train_writer = SummaryWriter(log_path, flush_secs=10)
    full_log_path = os.path.join(os.getcwd(), log_path)
    print(
        "Follow tensorboard logs with: python -m tensorboard.main --logdir '{}'".format(
            full_log_path
        )
    )
    return train_writer


def create_train_state(rng, nca, learning_rate, shape):
    nca_seed = nca.create_seed(
        nca.num_hidden_channels, nca.num_target_channels, shape=shape[:-1], batch_size=1
    )
    """Creates initial `TrainState`."""
    params = nca.init(rng, nca_seed, rng)["params"]
    tx = optax.chain(
        # optax.clip_by_global_norm(10.0),
        optax.adam(learning_rate),
    )
    return train_state.TrainState.create(apply_fn=nca.apply, params=params, tx=tx)


def clip_grad_norm(grad):
    factor = 1.0 / (
        jnp.linalg.norm(jax.tree_util.tree_leaves(jax.tree.map(jnp.linalg.norm, grad))[0])
        + 1e-8
    )
    return jax.tree.map((lambda x: x * factor), grad)


@functools.partial(jax.jit, static_argnames=("apply_fn", "num_steps"))
def train_step_static_image(
    apply_fn, state, seeds: jnp.array, targets: jnp.array, num_steps: int, rng
):
    def mse_loss(pred, y):
        squared_diff = jnp.square(pred - y)
        return jnp.mean(squared_diff, axis=[-3, -2, -1])

    def loss_fn(params):
        def forward(carry, inp):
            carry = apply_fn({"params": params}, carry, rng)
            return carry, carry

        x, outs = jax.lax.scan(forward, seeds, None, length=num_steps)
        rgb, a = x[..., :3], jnp.clip(x[..., 3:4], 0.0, 1.0)
        rgb = jnp.clip(1.0 - a + rgb, 0.0, 1.0)

        outs = jnp.transpose(outs, [1, 0, 2, 3, 4])
        subset = outs[:, -8:]  # B 12 H W C
        return jnp.mean(
            jax.vmap(mse_loss)(subset[..., :4], jnp.expand_dims(targets, 1))
        ), (x, rgb)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, aux), grads = grad_fn(state.params)
    grads = clip_grad_norm(grads)
    updated, rgb = aux
    return state.apply_gradients(grads=grads), loss, grads, updated, rgb


@functools.partial(jax.jit, static_argnames=("apply_fn", "num_steps"))
def train_step_clip(
    apply_fn, clip_loss_fn, state, seeds: jnp.array, num_steps: int, rng
):
    def loss_fn(params):
        def forward(carry, inp):
            carry = apply_fn({"params": params}, carry, rng)
            return carry, carry

        x, outs = jax.lax.scan(forward, seeds, None, length=num_steps)
        rgb = x[..., :3]

        # compute overflow loss  
        overflow_loss = (x-x.clamp(-1.0, 1.0)).abs().sum() 

        # 2 compute loss
        loss = clip_loss_fn(rgb) + overflow_loss

        return loss

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, aux), grads = grad_fn(state.params)
    grads = clip_grad_norm(grads)
    updated, rgb = aux
    return state.apply_gradients(grads=grads), loss, grads, updated, rgb

@functools.partial(jax.jit, static_argnames=("apply_fn", "vgg_loss_fn", "num_steps"))
def train_step_vgg(
    apply_fn, vgg_loss_fn, state, seeds: jnp.array, num_steps: int, rng
):
    def loss_fn(params):
        def forward(carry, inp):
            carry = apply_fn({"params": params}, carry, rng)
            return carry, carry

        x, outs = jax.lax.scan(forward, seeds, None, length=num_steps)
        rgb = x[..., :3]

        outs = jnp.transpose(outs, [1, 0, 2, 3, 4])
        subset = outs[:, -1:]  # B 12 H W C

        return vgg_loss_fn(subset[..., :3]), (x, rgb)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, aux), grads = grad_fn(state.params)
    grads = clip_grad_norm(grads)
    updated, rgb = aux
    return state.apply_gradients(grads=grads), loss, grads, updated, rgb


class SamplePool:
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.pool = [None] * max_size

    def __getitem__(self, idx):
        if isinstance(idx, Iterable):
            return [self.pool[i] for i in idx]
        return idx

    def __setitem__(self, idx, v):
        if isinstance(idx, Iterable):
            for i in range(len(idx)):
                index = idx[i]
                self.pool[index] = v[i]
        else:
            self.pool[idx] = v

    def sample(self, num_samples: int):
        indices = np.random.randint(0, self.max_size, num_samples)
        return self.__getitem__(indices), indices


def flatten(d):
    df = pd.json_normalize(d, sep="_")
    return df.to_dict(orient="records")[0]


class Trainer:
    def __init__(self, nca, pool_size: int = 1024, n_damage: int = 0):
        self.img_shape = self.dataset.img_shape
        self.nca = nca
        self.pool_size = pool_size
        self.n_damage = n_damage
        self.state = None

    @functools.partial(jax.jit, static_argnames=("self", "pool", "batch_size"))
    def _create_pool(self, pool, batch_size):
        samples, indices = pool.sample(batch_size)
        for j in range(len(samples)):
            if samples[j] is None:
                samples[j] = self.nca.create_seed(
                    self.nca.num_hidden_channels,
                    self.nca.num_target_channels,
                    shape=self.img_shape[:-1],
                    batch_size=1,
                )[0]
        samples[0] = self.nca.create_seed(
            self.nca.num_hidden_channels,
            self.nca.num_target_channels,
            shape=self.img_shape[:-1],
            batch_size=1,
        )[0]
        batch = np.stack(samples)
        return batch, indices
    
    @functools.partial(jax.jit, static_argnames=("self"))
    def _damage(self, batch):
        if self.n_damage > 0:
            damage = (
                1.0
                - make_circle_masks(
                    int(self.n_damage), self.img_shape[0], self.img_shape[1]
                )[..., None]
            )
            batch[-self.n_damage :] *= damage
        return batch
    
    def emit_metrics(
        self, train_writer, i: int, batch, outputs, targets, loss, metrics={}
    ):
        train_writer.add_scalar("loss", loss, i)
        # train_writer.add_scalar("log10(loss)", math.log10(loss), i)
        train_writer.add_images("batch", self.nca.to_rgb(batch), i, dataformats="NHWC")
        train_writer.add_images("outputs", outputs, i, dataformats="NHWC")
        train_writer.add_images("targets", targets, i, dataformats="NHWC")
        for k in metrics:
            train_writer.add_scalar(k, metrics[k], i)

class EmojiTrainer(Trainer):
    def __init__(self, dataset, nca, pool_size: int = 1024, n_damage: int = 0):
        self.dataset = dataset
        super().__init__(nca, pool_size=pool_size, n_damage=n_damage)

    def train(
        self,
        num_epochs,
        batch_size: int = 8,
        seed: int = 10,
        lr: float = 0.001,
        min_steps: int = 64,
        max_steps: int = 96,
    ):
        pool = SamplePool(self.pool_size)

        writer = get_tensorboard_logger("EMOJITrainer")
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        self.state = create_train_state(init_rng, self.nca, lr, self.dataset.img_shape)

        bar = tqdm.tqdm(np.arange(num_epochs))

        for i in bar:
            num_steps = int(np.random.randint(min_steps, max_steps))
            batch, indices = self._create_pool(pool, batch_size)
            batch = self._damage(batch)

            batch = jnp.array(batch)
            targets, rgb_targets = self.dataset.get_batch(batch_size)
            targets = jnp.array(targets)

            self.state, loss, grads, outputs, rgb_outputs = train_step_clip(
                self.nca.apply,
                self.state,
                batch,
                targets,
                num_steps=num_steps,
                rng=rng,
            )

            pool[indices] = np.array(outputs)

            bar.set_description("Loss: {}".format(loss.item()))

            self.emit_metrics(
                writer,
                i,
                batch,
                rgb_outputs,
                rgb_targets,
                loss.item()
            )

        return self.state
    
class CLIPTrainer(Trainer):
    def __init__(self, text, nca, model_name="wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M", pool_size: int = 1024, n_damage: int = 0):
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = FlaxCLIPModel.from_pretrained(model_name)
        self.text = text
        super().__init__(nca, pool_size=pool_size, n_damage=n_damage)


    def _clip_loss_fn(self, rgb):
        inputs = self.processor(
            text=self.text,
            images=rgb,
            return_tensors="np",
            padding="max_length",
            max_length=128,
            truncation=True,
        )
        outputs = self.model(**inputs)
        return outputs.loss

    def train(
        self,
        num_epochs,
        batch_size: int = 8,
        seed: int = 10,
        lr: float = 0.001,
        min_steps: int = 64,
        max_steps: int = 96,
    ):
        pool = SamplePool(self.pool_size)

        writer = get_tensorboard_logger("CLIPTrainer")
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        self.state = create_train_state(init_rng, self.nca, lr, self.dataset.img_shape)

        bar = tqdm.tqdm(np.arange(num_epochs))

        for i in bar:
            num_steps = int(np.random.randint(min_steps, max_steps))
            batch, indices = self._create_pool(pool, batch_size)
            batch = self._damage(batch)

            batch = jnp.array(batch)
            targets, rgb_targets = self.dataset.get_batch(batch_size)
            targets = jnp.array(targets)

            self.state, loss, grads, outputs, rgb_outputs = train_step_clip(
                self.nca.apply,
                self._clip_loss_fn,
                self.state,
                batch,
                num_steps=num_steps,
                rng=rng,
            )

            pool[indices] = np.array(outputs)

            bar.set_description("Loss: {}".format(loss.item()))

            self.emit_metrics(
                writer,
                i,
                batch,
                rgb_outputs,
                rgb_targets,
                loss.item()
            )

        return self.state

class VGGTrainer(Trainer):
    def __init__(self, nca, dataset, seed: int = 10, pool_size: int = 1024, n_damage: int = 0):
        self.dataset = dataset
        self.vgg = fm.VGG16(output='activations', pretrained='imagenet', include_head=False)
        self.key = jax.random.PRNGKey(seed)
        x, _ = self.dataset.get_batch(1)
        self.vgg_params = self.vgg.init(self.key, x[0])
        self.target_activations = self.vgg.apply(self.vgg_params, x[0], train=False)
        self.activation_layers = [f"conv{i}_1" for i in range(1, 6)]
        super().__init__(nca, pool_size=pool_size, n_damage=n_damage)

    def _vgg_loss_fn(self, rgb):
        # compare kernel activations of nca and target image
        rgb_activations = self.vgg.apply(self.vgg_params, rgb, train=False)

        def activation_diff(layer_activations, target_activations):
            return jnp.mean(jnp.square(layer_activations - target_activations))

        # Use JAX tree functions to compute differences
        differences = jax.tree.map(
            activation_diff,
            {layer: rgb_activations[layer] for layer in self.activation_layers},
            {layer: self.target_activations[layer] for layer in self.activation_layers}
        )

        return jax.tree.reduce(operator.add, differences)

        # Sum up the differences
        return jnp.sum(jnp.array(list(differences.values())))

    def train(
        self,
        num_epochs,
        batch_size: int = 8,
        lr: float = 0.001,
        min_steps: int = 64,
        max_steps: int = 96,
    ):
        pool = SamplePool(self.pool_size)

        writer = get_tensorboard_logger("CLIPTrainer")
        
        rng, init_rng = jax.random.split(self.key)
        self.state = create_train_state(init_rng, self.nca, lr, self.dataset.img_shape)

        bar = tqdm.tqdm(np.arange(num_epochs))

        for i in bar:
            num_steps = int(np.random.randint(min_steps, max_steps))
            batch, indices = self._create_pool(pool, batch_size)
            batch = self._damage(batch)

            batch = jnp.array(batch)
            targets, rgb_targets = self.dataset.get_batch(batch_size)
            targets = jnp.array(targets)

            self.state, loss, grads, outputs, rgb_outputs = train_step_vgg(
                self.nca.apply,
                self._vgg_loss_fn,
                self.state,
                batch,
                num_steps=num_steps,
                rng=rng,
            )

            pool[indices] = np.array(outputs)

            bar.set_description(f"Loss: {loss.item():.4f}")

            self.emit_metrics(
                writer,
                i,
                batch,
                rgb_outputs,
                rgb_targets,
                loss.item()
            )

        return self.state


def main():
    from jax_nca.nca import NCA
    from jax_nca.dataset import ImageDataset
    
    dataset = ImageDataset(emoji='🦎', img_size=64)
    nca = NCA(16, 3, trainable_perception=False, cell_fire_rate=1.0, alpha_living_threshold=0.1)
    trainer = EmojiTrainer(dataset, nca, n_damage=0)
    trainer.train(1000, batch_size=8, lr=2e-4, min_steps=64, max_steps=96)
    state = trainer.state

if __name__=="__main__":
    main()