import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import cv2
from numpy import random
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pickle
import time

IMG_SIZE = 640


def plot_one_box(x, img, color=None, label=None, line_thickness=3, ratio=None):
    if ratio is not None:
        x = x[0]*ratio[0], x[1]*ratio[1], x[2]*ratio[0], x[3]*ratio[1]
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def parse_inp(img):
    t0 = time.time()
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    print(img.flatten()[:30])
    t1 = time.time()
    print('Resizing time:', t1 - t0)
    t0 = time.time()
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    t1 = time.time()
    print('Transposing time:', t1 - t0)
    t0 = time.time()
    img = np.ascontiguousarray(img)
    t1 = time.time()
    print('As contiguous time:', t1 - t0)
    t0 = time.time()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    print(img.flatten()[:30])
    t1 = time.time()
    print('To float time:', t1 - t0)
    t0 = time.time()
    if len(img.shape) == 3:
        img = np.expand_dims(img, axis=0)
    t1 = time.time()
    print('Expand time:', t1 - t0)
    t0 = time.time()
    img_t = tf.cast(tf.convert_to_tensor(img), dtype=tf.float32)
    t1 = time.time()
    print('Casting time:', t1 - t0)
    return img_t

def detect(path):
    t0 = time.time()
    model = tf.saved_model.load('weights/tf')
    t1 = time.time()
    print('Loading time:', t1 - t0)
    t0 = time.time()
    img = cv2.imread(path).astype('float')
    t1 = time.time()
    print('Image loading time:', t1 - t0)
    t0 = time.time()
    img_t = parse_inp(img)
    t1 = time.time()
    print('Image preprocessing time:', t1 - t0)

    t0 = time.time()
    out = model(images=img_t)
    out = out['output'].numpy()
    t1 = time.time()
    print('Inference time:', t1 - t0)
    return out

def plot_result(path, out):
    im0 = cv2.imread(path)[:,:,::-1] / 1.
    with open('names.pickle', 'rb') as f:
        names = pickle.load(f)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    ratio = (im0.shape[1]/IMG_SIZE, im0.shape[0]/IMG_SIZE)
    for _, *xyxy, cls, conf in out:
        label = f'{names[int(cls)]} {conf:.2f}'
        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3, ratio=ratio)
    plt.imshow(im0.astype('uint8'))
    plt.show()

if __name__ == '__main__':
    path = 'IMG_2975.png'
    out = detect(path)
    plot_result(path, out)
