import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def autocontrast_channel(channel):
    # 计算直方图
    hist, bins = np.histogram(channel.ravel(), 256, [0, 255])

    # 自动对比度调整：找到最小和最大非零直方图索引
    nonzero = hist > 0
    min_val = hist[nonzero].min()
    max_val = hist[nonzero].max()

    # 创建查找表（LUT）
    lut = np.linspace(min_val, max_val, 256)
    lut = np.clip(lut, 0, 255).astype(np.uint8)

    # 应用查找表
    contrasted_channel = lut[channel]

    return contrasted_channel

def AutoContrast(image, _):
    img = np.array(image)
    # 对每个通道分别进行自动对比度调整
    channels = img.transpose((2, 0, 1))  # 将通道放在第一个维度
    contrasted_channels = [autocontrast_channel(ch) for ch in channels]

    # 将处理后的通道合并回图像
    image = np.stack(contrasted_channels, axis=0).transpose((1, 2, 0))
    pil_image = Image.fromarray(image.astype('uint8'))
    return pil_image


def Brightness(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Equalize(img, _):
    # 分离图像的红色、绿色、蓝色和alpha通道
    r, g, b, y = img.split()
    r = PIL.ImageOps.equalize(r)
    g = PIL.ImageOps.equalize(g)
    b = PIL.ImageOps.equalize(b)
    y = PIL.ImageOps.equalize(y)
    # 将处理后的颜色通道和原始的alpha通道合并
    image = Image.merge('RGBA', (r, g, b, y))
    return image


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Identity(img, v):
    return img


def Posterize(img, v):  # [4, 8]
    # 分离图像的红色、绿色、蓝色和alpha通道
    r, g, b, a = img.split()
    v = int(v)
    v = max(1, v)
    # 对每个颜色通道进行 posterize 处理
    r = PIL.ImageOps.posterize(r, v)
    g = PIL.ImageOps.posterize(g, v)
    b = PIL.ImageOps.posterize(b, v)
    a = PIL.ImageOps.posterize(a, v)
    # 将处理后的颜色通道和原始的alpha通道合并
    image = Image.merge('RGBA', (r, g, b, a))
    return image


def Rotate(img, v):  # [-30, 30]
    # assert -30 <= v <= 30
    # if random.random() > 0.5:
    #    v = -v
    return img.rotate(v)


def Sharpness(img, v):  # [0.1,1.9]
    assert v >= 0.0
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v):  # [-0.3, 0.3]
    # assert -0.3 <= v <= 0.3
    # if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    # assert -0.3 <= v <= 0.3
    # if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    # assert -0.3 <= v <= 0.3
    # if random.random() > 0.5:
    #    v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    # assert v >= 0.0
    # if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    # assert -0.3 <= v <= 0.3
    # if random.random() > 0.5:
    #    v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    # assert 0 <= v
    # if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    # 分离图像的红色、绿色、蓝色和alpha通道
    r, g, b, a = img.split()
    # 对每个颜色通道进行 posterize 处理
    r = PIL.ImageOps.solarize(r, v)
    g = PIL.ImageOps.solarize(g, v)
    b = PIL.ImageOps.solarize(b, v)
    a = PIL.ImageOps.solarize(a, v)
    # 将处理后的颜色通道和原始的alpha通道合并
    image = Image.merge('RGBA', (r, g, b, a))
    return image


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2] => change to [0, 0.5]
    assert 0.0 <= v <= 0.5
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def augment_list():
    l = [
        (AutoContrast, 0, 1),
        (Brightness, 0.05, 0.95),
        (Color, 0.05, 0.95),
        (Contrast, 0.05, 0.95),
        (Equalize, 0, 1),
        (Identity, 0, 1),
        (Posterize, 4, 8),
        (Rotate, -30, 30),
        (Sharpness, 0.05, 0.95),
        (ShearX, -0.3, 0.3),
        (ShearY, -0.3, 0.3),
        (Solarize, 0, 256),
        (TranslateX, -0.3, 0.3),
        (TranslateY, -0.3, 0.3)
    ]
    return l


def augment_list_no_color():
    l = [
        # (AutoContrast, 0, 1),
        (Brightness, 0.05, 0.95),
        # (Color, 0.05, 0.95),
        # (Contrast, 0.05, 0.95),
        (Equalize, 0, 1),
        (Identity, 0, 1),
        # (Posterize, 4, 8),
        (Rotate, -30, 30),
        (Sharpness, 0.05, 0.95),
        (ShearX, -0.3, 0.3),
        (ShearY, -0.3, 0.3),
        # (Solarize, 0, 256),
        (TranslateX, -0.3, 0.3),
        (TranslateY, -0.3, 0.3)
    ]
    return l


class RandAugment:
    def __init__(self, n, m, exclude_color_aug=False):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        if not exclude_color_aug:
            self.augment_list = augment_list()
        else:
            self.augment_list = augment_list_no_color()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, min_val, max_val in ops:
            val = min_val + float(max_val - min_val) * random.random()
            img = op(img, val)
        cutout_val = random.random() * 0.5
        img = Cutout(img, cutout_val)  # for fixmatch
        return img

if __name__ == '__main__':
    # randaug = RandAugment(3,5)
    # print(randaug)
    # for item in randaug.augment_list:
    #     print(item)
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    img = PIL.Image.open('./u.jpg')
    randaug = RandAugment(3, 6)
    img = randaug(img)
    import matplotlib
    from matplotlib import pyplot as plt

    plt.imshow(img)
    plt.show()
