"""Blob helper functions."""
import numpy as np
import cv2
from ..fast_rcnn.config import cfg

def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    将包含若干图片像素信息的list转换成blob数据块。这里的处理仅仅只是将所有的图片进行左上角的对齐。
    :param ims: 一个list，里面包含若干个图片的像素信息。
    :return： 处理之后的blob数据块。
    """
    # 返回各个维度的最大长度，这里真真有用的是最大的高度和宽度。
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    # 获取图片的总数目
    num_images = len(ims)
    # 根据图片总数目，最大高度宽度等信息，生成一个全0numpy数组，用以将图片的左上角对齐。
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3), dtype=np.float32)
    # 对每个图片
    for i in range(num_images):
        im = ims[i]
        # 进行赋值操作，这样的复制过程正好从blob数组的左上角开始。
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

    # 返回
    return blob

def prep_im_for_blob(im, pixel_means, target_size, max_size):
    """Mean subtract and scale an image for use in a blob."""
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    if cfg.TRAIN.RANDOM_DOWNSAMPLE:
        r = 0.6 + np.random.rand() * 0.4
        im_scale *= r
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)

    return im, im_scale
