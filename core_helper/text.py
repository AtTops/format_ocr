# -*- coding: utf-8 -*-
# @Time    : 18-11-1 下午6:31
# @Author  : wanghai
# @Email   : 
# @File    : text.py
# @Software: PyCharm Community Edition
import config as cfg
import numpy as np
import cv2
from config import DISPLAY
from detector.detectors import TextDetector
from matplotlib import cm

if cfg.opencvFlag:
    import opencv_dnn_detect as detect
else:
    import darknet_detect as detect


def text_detect(img,
                MAX_HORIZONTAL_GAP=30,
                MIN_V_OVERLAPS=0.6,
                MIN_SIZE_SIM=0.6,
                TEXT_PROPOSALS_MIN_SCORE=0.7,
                TEXT_PROPOSALS_NMS_THRESH=0.3,
                TEXT_LINE_NMS_THRESH=0.3,
                MIN_RATIO=1.0,
                LINE_MIN_SCORE=0.8,
                TEXT_PROPOSALS_WIDTH=5,
                MIN_NUM_PROPOSALS=1, ):
    # 画文本框
    boxes, scores = detect.text_detect(np.array(img))

    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    textdetector = TextDetector(MAX_HORIZONTAL_GAP, MIN_V_OVERLAPS, MIN_SIZE_SIM)
    shape = img.size[::-1]
    boxes = textdetector.detect(boxes,
                                scores[:, np.newaxis],
                                shape,
                                TEXT_PROPOSALS_MIN_SCORE,
                                TEXT_PROPOSALS_NMS_THRESH,
                                TEXT_LINE_NMS_THRESH,
                                MIN_RATIO,
                                LINE_MIN_SCORE,
                                TEXT_PROPOSALS_WIDTH,
                                MIN_NUM_PROPOSALS)
    # 画框结束
    text_recs, tmp = draw_boxes(np.array(img), boxes, color=None, caption='Box_Image', wait=True, display=DISPLAY)
    return text_recs, tmp


def draw_boxes(im, bboxes, color=(255, 191, 0), display=True, caption="no_name", wait=True):
    """
        boxes: bounding boxes
        cv2 中是bgr
    """
    text_recs = np.zeros((len(bboxes), 8), np.int)
    # print("该图检测到 %d 个文本框。" % len(bboxes))
    img = im.copy()
    index = 0
    for box in bboxes:
        if color is None:
            if len(box) == 8 or len(box) == 9:
                c = tuple(cm.jet([box[-1]])[0, 2::-1] * 255)
            else:
                c = tuple(np.random.randint(0, 256, 3))
        else:
            c = color

        b1 = box[6] - box[7] / 2
        b2 = box[6] + box[7] / 2
        x1 = box[0]
        y1 = box[5] * box[0] + b1
        x2 = box[2]
        y2 = box[5] * box[2] + b1
        x3 = box[2]
        y3 = box[5] * box[2] + b2
        x4 = box[0]
        y4 = box[5] * box[0] + b2

        disX = x2 - x1
        disY = y2 - y1
        width = np.sqrt(disX * disX + disY * disY)
        fTmp0 = y3 - y1
        fTmp1 = fTmp0 * disY / width
        x = np.fabs(fTmp1 * disX / width)
        y = np.fabs(fTmp1 * disY / width)
        if box[5] < 0:
            x1 -= x
            y1 += y
            x4 += x
            y4 -= y
        else:
            x2 += x
            y2 += y
            x3 -= x
            y3 -= y
        # 画线,起点，终点，线宽
        if display:
            draw(img, x1, y1, x2, y2, x3, y3, x4, y4, c)
        text_recs[index, 0] = x1
        text_recs[index, 1] = y1
        text_recs[index, 2] = x2
        text_recs[index, 3] = y2
        text_recs[index, 4] = x3
        text_recs[index, 5] = y3
        text_recs[index, 6] = x4
        text_recs[index, 7] = y4
        index += 1
        # cv2.rectangle(im, tuple(box[:2]), tuple(box[2:4]), c,2)

    if display:
        cv2.imshow(caption, img)
        if wait:
            cv2.waitKey(0)
        else:
            cv2.waitKey(2000)
        cv2.destroyAllWindows()
    return text_recs, im


def draw(im, x1, y1, x2, y2, x4, y4, x3, y3, color=None):
    """
    画框，ctpn检测出来的框(x3与x4的坐标不必纠结)
    """
    cv2.line(im, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
    cv2.line(im, (int(x1), int(y1)), (int(x3), int(y3)), color, 1)
    cv2.line(im, (int(x3), int(y3)), (int(x4), int(y4)), color, 1)
    cv2.line(im, (int(x4), int(y4)), (int(x2), int(y2)), color, 1)
    return im
