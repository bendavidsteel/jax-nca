import jax
import jax.numpy as jnp
import numpy as np

from jax_nca.dataset import WebLinkDataset
from jax_nca.nca import TextureNCA
from jax_nca.trainer import VGGTrainer
import PIL.Image

def main():
    img_size = 128
    dataset = WebLinkDataset(link='https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Drying_mud_with_120_degree_cracks%2C_Sicily.jpg/1920px-Drying_mud_with_120_degree_cracks%2C_Sicily.jpg', img_size=img_size)

    nca = TextureNCA(8, alive_layer=False)

    trainer = VGGTrainer(nca, dataset, seed=10)

    state = trainer.train(2000, batch_size=2, lr=2e-4, min_steps=32, max_steps=64)

    num_steps = 128
    nca_seed = nca.create_seed(nca.num_hidden_channels, nca.num_target_channels, shape=(img_size, img_size), batch_size=1)
    rng = jax.random.PRNGKey(0)
    _, outputs = nca.multi_step(state.params, nca_seed, rng, num_steps=num_steps)
    stacked = jnp.squeeze(jnp.stack(outputs))
    rgbs = np.array(nca.to_rgb(stacked))

    imgs = [PIL.Image.fromarray((rgbs[i] * 255).astype(np.uint8)) for i in range(num_steps)]
    imgs[0].save('drying_mud.gif', save_all=True, append_images=imgs[1:])


if __name__ == "__main__":
    main()