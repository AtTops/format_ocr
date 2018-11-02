# -*- coding: utf-8 -*-
# @Time    : 18-11-1 下午3:45
# @Author  : wanghai
# @Email   : 
# @File    : angle.py
# @Software: PyCharm Community Edition
import config as cfg
import numpy as np
from PIL import Image
from scipy.ndimage import filters, interpolation
from config import yoloCfg, yoloWeights
from config import AngleModelPb, AngleModelPbtxt
from config import IMGSIZE
from opencv_dnn_detect import angle_detect  # 文字方向检测
import cv2

textNet = cv2.dnn.readNetFromDarknet(yoloCfg, yoloWeights)
# 文字方向检测模型
angleNet = cv2.dnn.readNetFromTensorflow(AngleModelPb, AngleModelPbtxt)


def eval_angle(im, detect_angle=False, if_adjust_degree=True):
    """
    估计图片偏移角度
    :param im:
    :param detect_angle: 是否检测文字朝向?图片？
    :param if_adjust_degree: 获得调整文字识别结果
    :return:
    """
    angle = 0
    degree = 0.0
    img = np.array(im)
    if detect_angle:
        angle = angle_detect(img=np.copy(img))  # 图片？文字倾斜角度检测，有待改善（少数改变全局的问题 TODO）
        if angle != 0:
            if angle == 90:
                im = im.transpose(Image.ROTATE_90)
            elif angle == 180:
                im = im.transpose(Image.ROTATE_180)
            elif angle == 270:
                im = im.transpose(Image.ROTATE_270)

    if if_adjust_degree:
        degree = estimate_skew_angle(np.array(im.convert('L')))  # 一通道的图
    return angle, degree, im.rotate(degree)


def estimate_skew_angle(raw):
    """
    估计图像文字角度
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
    angles = range(-15, 15) # TODO:+-15,应该是足够了
    estimates = []
    for a in angles:
        # https://docs.scipy.org/doc/scipy-0.7.x/reference/generated/scipy.ndimage.interpolation.rotate.html
        roest = interpolation.rotate(est, a, order=0, mode='constant')
        v = np.mean(roest, axis=1)
        v = np.var(v)
        estimates.append((v, a))
    _, a = max(estimates)
    return a


def resize_im(im, scale=cfg.SCALE, max_scale=cfg.MAX_SCALE):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale is not None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, (0, 0), fx=f, fy=f)
