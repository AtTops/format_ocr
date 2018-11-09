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
          left_adjust=cfg.left_adjust, right_adjust=cfg.right_adjust, alpha=cfg.alpha,
          result_typeset_opotion=cfg.result_typeset_opotion):
    """
    调整文字识别结果
    :param img An :py:class:`~PIL.Image.Image` object.
    :param
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

    # 3. 识别文本
    results_original, text_pix_per_avg = crnnRec(np.array(img), text_recs, if_im, left_adjust, right_adjust, alpha)

    # 4. 对3中结果排序
    slide_x_pix = cfg.slide_x_threshold * text_pix_per_avg
    slide_x_pix = max(config['MAX_HORIZONTAL_GAP'], slide_x_pix)  # x方向，小于该值的，不应该分开
    slide_y_pix = cfg.slide_y_threshold * text_pix_per_avg
    results_sorted = typeset_result(results_original, slide_x_pix, slide_y_pix, result_typeset_opotion)

    return img, results_original, results_sorted, angle


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
        part_img, w, h = rotate_cut_img(img, degree, rec, w, h, left_adjust, right_adjust, alpha)
        if if_im:
            part_img.show()
        simPred = crnnOcr(part_img.convert('L'))  # 识别的文本
        # 计算该框平均一个文字占用多少像素
        if len(simPred) > 0:
            text_pix_per = math.ceil(w / len(simPred))
            text_pix_count += text_pix_per
        else:
            text_recs_len -= 1
        if simPred.strip() != u'':
            results.append({'cx': cx, 'cy': cy, 'text': simPred, 'w': w, 'h': h, 'degree': degree * 180.0 / np.pi})
    print("这张图共%d个框，识别总耗时：%f" % (text_recs_len, time.time() - t0))
    text_pix_per_avg = math.ceil(text_pix_count / text_recs_len)
    print('>> text_pix_per_avg: ', text_pix_per_avg)
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

    new_w = box[2] - box[0]
    new_h = box[3] - box[1]
    tmp_img = im.rotate(degree_, center=(x_center, y_center)).crop(box)
    return tmp_img, new_w, new_h


def typeset_result(result, slide_x_pix=None, slide_y_pix=None, result_typeset_opotion=cfg.result_typeset_opotion):
    """
    对结果排序,页面排版
    """
    if result_typeset_opotion == 0:  # “智能排版”
        count_row = int(1024 / slide_y_pix) + 1
        count_col = int(1024 / slide_x_pix) + 1
        results_matrix = np.zeros((count_row, count_col), dtype=object)
        for index, _ in enumerate(result):
            print('here: ', result[index]["text"])
            x1 = result[index]["cx"] - result[index]["w"] / 2
            y1 = result[index]["cy"] - result[index]["h"] / 2
            col_index = row_index = 0
            for k in range(1, count_row):
                if k * slide_y_pix > y1:
                    row_index = k - 1
                    print('row_index : ', row_index)
                    break
                else:
                    continue
            for i in range(1, count_col):
                if i * slide_x_pix > x1:
                    col_index = i - 1
                    print('col_index : ', col_index)
                    break
                else:
                    continue
            results_matrix[row_index, col_index] = result[index]["text"]

        # 空白替换   (slide_y_pix = cfg.slide_y_threshold * text_pix_per_avg) TODO: 解决可能出现的覆盖问题
        space = ' ' * (math.ceil(slide_x_pix * cfg.slide_y_threshold / slide_y_pix) + 1)
        # space = ' '
        for row in range(results_matrix.shape[0]):
            for col in range(results_matrix.shape[1]):
                if results_matrix[row, col] == 0:
                    results_matrix[row, col] = space
        return results_matrix
        # if result_typeset_opotion == 10:  # 普通横向优先排版
        #     box = sorted(box, key=lambda x: sum([x[1], x[3], x[5], x[7]]))
        #     return list(box)
        # else:  # 普通纵向优先排版
        #     box = sorted(box, key=lambda x: sum([x[1], x[3], x[5], x[7]]))
        #     return list(box)
