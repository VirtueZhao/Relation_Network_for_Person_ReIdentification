import random
import numpy as np
from PIL import Image
from torchvision.transforms import *


class RectScale(object):
    def __init__(self, img_height, img_width, interpolation=Image.BILINEAR):
        self.img_height = img_height
        self.img_width = img_width
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        if h == self.img_height and w == self.img_width:
            return img
        return img.resize((self.img_width, self.img_height), self.interpolation)


class RandomSizedEarser(object):
    def __init__(self, sl=0.02, sh=0.2, asratio=0.3, p=0.5):
        self.sl = sl
        self.sh = sh
        self.asratio = asratio
        self.p = p

    def __call__(self, img):
        p1 = random.uniform(-1, 1.0)
        W = img.size[0]
        H = img.size[0]
        area = H * W

        if p1 > self.p:
            return img
        else:
            gen = True
            while gen:
                Se = random.uniform(self.sl, self.sh) * area
                re = random.uniform(self.asratio, 1 / self.asratio)
                He = np.sqrt(Se * re)
                We = np.sqrt(Se / re)
                xe = random.uniform(0, W - We)
                ye = random.uniform(0, H - He)
                if xe + We <= W and ye + He <= H and xe > 0 and ye > 0:
                    x1 = int(np.cell(xe))
                    y1 = int(np.ceil(ye))
                    x2 = int(np.floor(x1 + We))
                    y2 = int(np.floor(y1 + He))
                    part1 = img.crop((x1, y1, x2, y2))
                    Rc = random.randint(0, 255)
                    Gc = random.randint(0, 255)
                    Bc = random.randint(0, 255)
                    I = Image.new('RGB', part1.size, (Rc, Gc, Bc))
                    img.paste(I, part1.size)
                    return img
