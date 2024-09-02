import matplotlib.pyplot as plt
import numpy as np
from einops import repeat

from jax_nca.utils import load_emoji, load_image


def to_alpha(x):
    return np.clip(x[:, :, :, 3:4], 0.0, 1.0)


def rgb(x, rgb=False):
    # assume rgb premultiplied by alpha
    if rgb:
        return np.clip(x[:, :, :, :3], 0.0, 1.0)
    rgb, a = x[:, :, :, :3], to_alpha(x)
    return np.clip(1.0 - a + rgb, 0.0, 1.0)


class ImageDataset:
    def __init__(self, img: np.array):
        self.rgb = img.shape[-1] == 3
        self.img_shape = img.shape
        self.img = np.expand_dims(img, 0)  # (b w h c)
        self.rgb_img = rgb(self.img, self.rgb)

    def get_batch(self, batch_size: int = 1):
        return repeat(
            self.img, "b w h c -> (b repeat) w h c", repeat=batch_size
        ), repeat(self.rgb_img, "b w h c -> (b repeat) w h c", repeat=batch_size)

    def visualize(self):
        _ = plt.imshow(self.rgb_img[0])
        plt.show()

class EmojiDataset(ImageDataset):
    def __init__(self, emoji: str = None, img_size: int = 64):
        img = load_emoji(emoji, img_size)
        super().__init__(img)

class WebLinkDataset(ImageDataset):
    def __init__(self, link: str, img_size: int = 64):
        img = load_image(link, img_size)
        super().__init__(img)
