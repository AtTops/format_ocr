import numpy as np
import math
import time
from config import cfg
from PIL import Image
from numpy import cos, sin
from crnn.crnn_ import crnnOcr as crnnOcr
from core_helper.angle import global_tune_angle, fine_tune_angle
from core_helper.text import text_detect


def model(img, global_tune=cfg.global_tune, fine_tune=cfg.fine_tune, config={}, if_im=cfg.if_im,
          left_adjust=cfg.left_adjust, right_adjust=cfg.right_adjust, alpha=cfg.alpha):
    """
    调整文字识别结果
    :param img An :py:class:`~PIL.Image.Image` object.
    :param global_tune 是否检测并纠正总体朝向  0.5s左右
    :param config 配置参数
    :param if_im:
    :param left_adjust:
    :param right_adjust:
    :param alpha:
    :param fine_tune:
    :return:
    """
    # 1. 角度检测、修正等
    angle = 0
    degree = 0.0
    t0 = time.time()
    if global_tune:
        angle, img = global_tune_angle(img)
    if fine_tune:
        degree, img = fine_tune_angle(img)
    img = img.rotate(degree)
    img = letterbox_image(img, cfg.img_size)
    print("角度检测调整耗时:{}s".format(time.time() - t0))

    config['img'] = letterbox_image(img)  # 缩放，参考yolo3

    # 2. 画文本框
    text_recs, tmp = text_detect(**config)
    # sorted_box = sort_box(text_recs)

    # 3. 识别文本
    result, text_pix_per_avg = crnnRec(np.array(img), text_recs, if_im, left_adjust, right_adjust, alpha)

    # 4. 对3中结果排序
    return img, result, angle


def letterbox_image(image, size=cfg.img_size):
    """
        缩放，参考yolo3
        resize image with unchanged aspect ratio using padding
        Reference: https://github.com/qqwweee/keras-yolo3/blob/master/yolo3/utils.py
    :param image: Image
    :param size: config (1024,1024)
    :return:
    """

    image_w, image_h = image.size
    w, h = size

    if max(image_w, image_h) < min(size):
        resized_image = image
        new_w = w
        new_h = h
    else:
        new_w = int(image_w * min(w * 1.0 / image_w, h * 1.0 / image_h))
        new_h = int(image_h * min(w * 1.0 / image_w, h * 1.0 / image_h))
        resized_image = image.resize((new_w, new_h), Image.BICUBIC)

    padded_image = Image.new('RGB', size, (128, 128, 128))
    padded_image.paste(resized_image, ((w - new_w) // 2, (h - new_h) // 2))
    return padded_image


def sort_box(box, drift=5):
    """
    对box排序,页面排版
        box[index, 0] = x1
        box[index, 1] = y1
        box[index, 2] = x2
        box[index, 3] = y2
        box[index, 4] = x3
        box[index, 5] = y3
        box[index, 6] = x4
        box[index, 7] = y4
    """
    x_drift = 0
    y_drift = 0
    box = sorted(box, key=lambda x: sum([x[1], x[3], x[5], x[7]]))
    return list(box)


def crnnRec(im, text_recs, if_im, left_adjust, right_adjust, alpha):
    """
    crnn模型，ocr识别
    :param im
    :param text_recs text box
    :param if_im 是否输出box对应的img
    :param left_adjust:
    :param right_adjust:
    :param alpha:
    :return
    """
    results = []
    img = Image.fromarray(im)  # TODO: why img all 128
    text_recs_len = len(text_recs)
    text_pix_count = 0
    t0 = time.time()
    for index, rec in enumerate(text_recs):
        # box是否倾斜（并不是主要步骤）
        degree, cx, cy, w, h = center_and_degree(rec)
        # 转图   左右微调并degree度数旋转（并不是主要步骤） TODO: add if
        partImg, w, h = rotate_cut_img(img, degree, rec, w, h, left_adjust, right_adjust, alpha)
        if if_im:
            partImg.show()
        partImg_ = partImg.convert('L')
        simPred = crnnOcr(partImg_)  # 识别的文本
        # 计算该框平均一个文字占用多少像素
        text_pix_per = math.ceil(w / len(simPred))
        text_pix_count += text_pix_per
        if simPred.strip() != u'':
            results.append({'cx': cx, 'cy': cy, 'text': simPred, 'w': w, 'h': h, 'degree': degree * 180.0 / np.pi})
    print("这张图共%d个框，识别总耗时：%f" % (text_recs_len, time.time() - t0))
    text_pix_per_avg = math.ceil(text_pix_count / text_recs_len)
    return results, text_pix_per_avg


def center_and_degree(box):
    """
    :return box中心坐标;w、h;角度
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
    cx = (x1 + x2 + x3 + x4) / 4.0
    cy = (y1 + y2 + y3 + y4) / 4.0
    w = (np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) + np.sqrt((x3 - x4) ** 2 + (y3 - y4) ** 2)) / 2  # 加上y轴严谨些
    h = (np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2) + np.sqrt((x1 - x4) ** 2 + (y1 - y4) ** 2)) / 2  # 加上x轴严谨些
    sinA = (h * (x1 - cx) - w * (y1 - cy)) * 1.0 / (h * h + w * w) * 2
    degree = np.arcsin(sinA)
    return degree, cx, cy, w, h


def rotate_cut_img(im, degree, box, w, h, left_adjust=cfg.left_adjust, right_adjust=cfg.right_adjust, alpha=cfg.alpha):
    """
       如果框稍微有些斜，纠正(cut一个稍大的矩形); left_adjust + right_adjust
    :param im:
    :param degree:
    :param box:
    :param w:
    :param h:
    :param left_adjust:
    :param right_adjust:
    :param alpha:
    :return: 纠正cut之后的box部分的图，新的w，h
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
    x_center, y_center = np.mean([x1, x2, x3, x4]), np.mean([y1, y2, y3, y4])
    degree_ = degree * 180.0 / np.pi
    right = 0
    left = 0
    if right_adjust:
        right = 1
    if left_adjust:
        left = 1

    box = (max(1, x_center - w / 2 - left * alpha * (w / 2)),  # xmin
           y_center - h / 2,  # ymin
           min(x_center + w / 2 + right * alpha * (w / 2), im.size[0] - 1),  # xmax
           y_center + h / 2)  # ymax

    newW = box[2] - box[0]
    newH = box[3] - box[1]
    tmpImg = im.rotate(degree_, center=(x_center, y_center)).crop(box)
    return tmpImg, newW, newH


def xy_rotate_box(cx, cy, w, h, angle):
    """
    绕 cx,cy点 w,h 旋转 angle度
    x_new = (x-cx)*cos(angle) - (y-cy)*sin(angle)+cx
    y_new = (x-cx)*sin(angle) + (y-cy)*sin(angle)+cy
    :return 旋转纠正之后的坐标
    """
    cx = float(cx)
    cy = float(cy)
    w = float(w)
    h = float(h)
    angle = float(angle)
    x1, y1 = rotate(cx - w / 2, cy - h / 2, angle, cx, cy)
    x2, y2 = rotate(cx + w / 2, cy - h / 2, angle, cx, cy)
    x3, y3 = rotate(cx + w / 2, cy + h / 2, angle, cx, cy)
    x4, y4 = rotate(cx - w / 2, cy + h / 2, angle, cx, cy)
    return [x1, y1, x2, y2, x3, y3, x4, y4]


def rotate(x, y, angle, cx, cy):
    angle = angle  # *pi/180
    x_new = (x - cx) * cos(angle) - (y - cy) * sin(angle) + cx
    y_new = (x - cx) * sin(angle) + (y - cy) * cos(angle) + cy
    return x_new, y_new
