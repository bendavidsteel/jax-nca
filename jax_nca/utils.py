import io

import numpy as np
import PIL.Image
import requests


def load_image(url, size):
    r = requests.get(url)
    img = PIL.Image.open(io.BytesIO(r.content))
    img.thumbnail((40, 40), PIL.Image.ANTIALIAS)
    img = np.float32(img) / 255.0
    # premultiply RGB by Alpha
    img[..., :3] *= img[..., 3:]
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
