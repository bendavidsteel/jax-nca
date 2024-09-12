import io

import numpy as np
import PIL.Image
import requests


def load_image(url, size, pad=True):
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    img = PIL.Image.open(io.BytesIO(r.content))
    if pad:
        img.thumbnail((40, 40), PIL.Image.LANCZOS)
    else:
        img = img.resize((size, size))
    img = np.float32(img) / 255.0
    if img.shape[-1] == 4:
        # premultiply RGB by Alpha
        img[..., :3] *= img[..., 3:]
    if pad:
        # pad to self.h, self.h
        diff = size - 40
        img = np.pad(img, ((diff // 2, diff // 2), (diff // 2, diff // 2), (0, 0)))
    
    return img


def load_emoji(emoji, size, code=None):
    if code is None:
        code = hex(ord(emoji))[2:].lower()
    url = (
        "https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u%s.png?raw=true"
        % code
    )
    return load_image(url, size)


def make_circle_masks(n, h, w, r=None):
    x = np.linspace(-1.0, 1.0, w)[None, None, :]
    y = np.linspace(-1.0, 1.0, h)[None, :, None]
    center = np.random.uniform(-0.5, 0.5, size=[2, n, 1, 1])
    if r is None:
        r = np.random.uniform(0.1, 0.4, size=[n, 1, 1])
    x, y = (x - center[0]) / r, (y - center[1]) / r
    mask = x * x + y * y < 1.0
    return mask.astype(float)
