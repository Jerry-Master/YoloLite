import time
from pathlib import Path

import cv2
import torch
from numpy import random
import numpy as np
import matplotlib.pyplot as plt

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression
from utils.plots import plot_one_box


def parse_img(img, stride):
    img_size = 320
    ratio = img.shape[1] / img_size
    img = letterbox(img, img_size, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img)
    img_t = torch.from_numpy(img).float()
    img_t /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img_t.ndimension() == 3:
        img_t = img_t.unsqueeze(0)
    return img_t, ratio


def parse_output(out, path, names, colors, ratio):
    im0 = cv2.imread(path)[:,:,::-1] / 1.
    for *xyxy, conf, cls in out:
        label = f'{names[int(cls)]} {conf:.2f}'
        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3, ratio=ratio)
    return im0.astype('uint8')


def detect(path):
    t0 = time.time()
    model = attempt_load('weights/yolov7.pt', map_location='cpu')

    strides = int(model.stride.max())
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    
    img = cv2.imread(path)
    img_t, ratio = parse_img(img, strides)
    t1 = time.time()
    print('Loading time:', t1 - t0)

    t0 = time.time()
    out = model(img_t, augment=False)[0]
    out = non_max_suppression(out, 0.25, 0.45, classes=None, agnostic=None)
    out = out[0].detach().numpy()
    out = parse_output(out, path, names, colors, ratio)
    t1 = time.time()
    print('Inference time:', t1 - t0)
    return out
    

if __name__ == '__main__':
    img = detect('horses.jpg')
    plt.imshow(img)
    plt.show()
