# -*- coding: utf-8 -*-
# @Time    : 18-11-1 下午3:45
# @Author  : wanghai
# @Email   : 
# @File    : angle.py
# @Software: PyCharm Community Edition
from config import cfg
import numpy as np
from PIL import Image
from scipy.ndimage import filters, interpolation
from opencv_dnn_detect import angle_detect  # 文字方向检测
import cv2


def global_tune_angle(im):
    """
    大方向估计并调整图片偏移角度
    :param im An :py:class:`~PIL.Image.Image` object.
    :return: angle: int
              im  : Image
            检测到的需要逆时针旋转的度数，以及旋转过后的图
    """
    img = np.array(im)
    angle = angle_detect(np.copy(img))  # 图片的整体大方向调整，逆时针旋转 镜像
    print("------ angle: %d" % angle)
    if angle != 0:
        # 逆时针旋转
        if angle == 90:
            im = im.transpose(Image.ROTATE_90)
        elif angle == 180:
            im = im.transpose(Image.ROTATE_180)
        elif angle == 270:
            im = im.transpose(Image.ROTATE_270)
    return angle, im


def fine_tune_angle(im):
    """
        微调倾斜的图片
        :param im An :py:class:`~PIL.Image.Image` object.
        :return: degree: float
                  im  : Image
                检测到的需要逆时针旋转的度数，以及旋转过后的图
        """
    degree = estimate_skew_angle(np.array(im.convert('L')))  # 一通道的图
    print("------ degree : %f" % degree)
    return degree, im


def estimate_skew_angle(raw):
    """
    检测倾斜角度
    """
    raw = resize_im(raw)
    image = raw - np.amin(raw)  # 归一化
    image = image / np.amax(image)
    m = interpolation.zoom(image, 0.5)
    m = filters.percentile_filter(m, 80, size=(20, 2))
    m = filters.percentile_filter(m, 80, size=(2, 20))
    m = interpolation.zoom(m, 1.0 / 0.5)
    w, h = min(image.shape[1], m.shape[1]), min(image.shape[0], m.shape[0])
    a = np.array([x + 1 for x in image[:h, :w] - m[:h, :w]])
    flat = np.clip(a, 0, 1)
    d0, d1 = flat.shape
    o0, o1 = int(0.1 * d0), int(0.1 * d1)
    flat = np.amax(flat) - flat
    flat -= np.amin(flat)
    est = flat[o0:d0 - o0, o1:d1 - o1]
    angles = range(-15, 15)  # TODO:+-15,应该是足够了
    estimates = []
    for a in angles:
        # https://docs.scipy.org/doc/scipy-0.7.x/reference/generated/scipy.ndimage.interpolation.rotate.html
        roest = interpolation.rotate(est, a, order=0, mode='constant')
        v = np.mean(roest, axis=1)
        v = np.var(v)
        estimates.append((v, a))
    _, a = max(estimates)
    return a


def resize_im(im, scale=cfg.scale, max_scale=cfg.max_scale):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale is not None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, (0, 0), fx=f, fy=f)
