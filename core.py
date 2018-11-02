import config as cfg
import numpy as np
from PIL import Image
from numpy import cos, sin
from crnn.crnn import crnnOcr as crnnOcr
from core_helper.angle import eval_angle
from core_helper.text import text_detect


def xy_rotate_box(cx, cy, w, h, angle):
    """
    绕 cx,cy点 w,h 旋转 angle 的坐标
    x_new = (x-cx)*cos(angle) - (y-cy)*sin(angle)+cx
    y_new = (x-cx)*sin(angle) + (y-cy)*sin(angle)+cy
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
    return x1, y1, x2, y2, x3, y3, x4, y4


def rotate(x, y, angle, cx, cy):
    angle = angle  # *pi/180
    x_new = (x - cx) * cos(angle) - (y - cy) * sin(angle) + cx
    y_new = (x - cx) * sin(angle) + (y - cy) * cos(angle) + cy
    return x_new, y_new


def model(img, detect_angle=False, config={}, if_im=True, left_adjust=False, right_adjust=False, alph=0.2,
          if_adjust_degree=False):
    """

    :param img 调整文字识别结果
    :param detect_angle 是否检测文字朝向
    :param config 配置参数
    :param if_im:
    :param left_adjust:
    :param right_adjust:
    :param alph:
    :param if_adjust_degree:
    :return:
    """
    # 1. 角度检测、修正等
    angle, degree, img = eval_angle(img, detect_angle=detect_angle, if_adjust_degree=if_adjust_degree)
    img = letterbox_image(img, cfg.IMGSIZE)

    # 2. 画文本框
    config['img'] = img
    import time
    t = time.time()
    text_recs, tmp = text_detect(**config)
    print("bbox 总耗时:{}s".format(time.time() - t))
    sorted_box = sort_box(text_recs)

    # 3. 识别文本
    result = crnnRec(np.array(img), sorted_box, if_im, left_adjust, right_adjust, alph)
    return img, result, angle


def letterbox_image(image, size):
    """ 缩放，参考yolo3
        resize image with unchanged aspect ratio using padding
        Reference: https://github.com/qqwweee/keras-yolo3/blob/master/yolo3/utils.py
    :param image:
    :param size:
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

    boxed_image = Image.new('RGB', size, (128, 128, 128))
    boxed_image.paste(resized_image, ((w - new_w) // 2, (h - new_h) // 2))
    return boxed_image


def sort_box(box):
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

    box = sorted(box, key=lambda x: sum([x[1], x[3], x[5], x[7]]))
    return list(box)

import time

def crnnRec(im, text_recs, if_im=False, left_adjust=False, right_adjust=False, alph=0.2):
    """
    crnn模型，ocr识别
    :param im
    :param text_recs text box
    :param if_im 是否输出box对应的img
    :param left_adjust:
    :param right_adjust:
    :param alph:
    :return
    """
    results = []
    img = Image.fromarray(im)
    for index, rec in enumerate(text_recs):
        t = time.time()
        degree, w, h, cx, cy = solve(rec)
        if left_adjust or right_adjust:
            partImg, w, h = rotate_cut_img(img, degree, rec, w, h, left_adjust, right_adjust, alph)
            # 暂时保留，可能之后有用
            newBox = xy_rotate_box(cx, cy, w, h, degree)
            partImg_ = partImg.convert('L')
            # partImg.show()
            simPred = crnnOcr(partImg_)  # 识别的文本
        else:
            simPred = crnnOcr(img.convert('L'))
        if simPred.strip() != u'':
            results.append(
                {'cx': cx, 'cy': cy, 'text': simPred, 'w': w, 'h': h, 'degree': degree * 180.0 / np.pi})
        print("==============ocr 本次耗时:{}s".format(time.time() - t))
    return results


def solve(box):
    """
    绕 cx,cy点 w,h 旋转 angle 的坐标
    x = cx-w/2
    y = cy-h/2
    x1-cx = -w/2*cos(angle) +h/2*sin(angle)
    y1 -cy= -w/2*sin(angle) -h/2*cos(angle)

    h(x1-cx) = -wh/2*cos(angle) +hh/2*sin(angle)
    w(y1 -cy)= -ww/2*sin(angle) -hw/2*cos(angle)
    (hh+ww)/2sin(angle) = h(x1-cx)-w(y1 -cy)
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
    cx = (x1 + x3 + x2 + x4) / 4.0
    cy = (y1 + y3 + y4 + y2) / 4.0
    w = (np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) + np.sqrt((x3 - x4) ** 2 + (y3 - y4) ** 2)) / 2
    h = (np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2) + np.sqrt((x1 - x4) ** 2 + (y1 - y4) ** 2)) / 2
    sinA = (h * (x1 - cx) - w * (y1 - cy)) * 1.0 / (h * h + w * w) * 2
    angle = np.arcsin(sinA)
    return angle, w, h, cx, cy


def rotate_cut_img(im, degree, box, w, h, left_adjust=False, right_adjust=False, alph=0.2):
    # x_center, y_center = np.mean(box[:4]), np.mean(box[4:8]) # bbox的中心坐标
    x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
    x_center, y_center = np.mean([x1, x2, x3, x4]), np.mean([y1, y2, y3, y4])
    degree_ = degree * 180.0 / np.pi
    right = 0
    left = 0
    if right_adjust:
        right = 1
    if left_adjust:
        left = 1

    box = (max(1, x_center - w / 2 - left * alph * (w / 2))  # xmin
           , y_center - h / 2,  # ymin
           min(x_center + w / 2 + right * alph * (w / 2), im.size[0] - 1)  # xmax
           , y_center + h / 2)  # ymax

    newW = box[2] - box[0]
    newH = box[3] - box[1]
    tmpImg = im.rotate(degree_, center=(x_center, y_center)).crop(box)
    return tmpImg, newW, newH
