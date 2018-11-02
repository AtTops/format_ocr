# -*- coding: utf-8 -*-
# @Time    : 18-10-24 上午11:11
# @Author  :
# @Email   : 
# @File    : demo.py
# @Software: PyCharm Community Edition

import os
import core
from glob import glob
from PIL import Image
import time
from config import DETECTANGLE

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

paths = glob('./test_img/cpsm*.jpg')
# paths = ['./test_img/mp.jpg', './test_img/mp1.jpg', './test_img/mp2.jpg', './test_img/mp3.jpg']

if __name__ == '__main__':
    print("OCR Starting!")
    img = Image.open(paths[3]).convert("RGB")
    t = time.time()
    _, result, angle = core.model(img,
                                  detect_angle=DETECTANGLE,  # 是否进行图片/文字方向检测
                                  config=dict(MAX_HORIZONTAL_GAP=80,  # 字符之间的最大间隔，用于文本行的合并 这些配置的应用在画文本框时候
                                              MIN_V_OVERLAPS=0.6,
                                              MIN_SIZE_SIM=0.6,
                                              TEXT_PROPOSALS_MIN_SCORE=0.2,
                                              TEXT_PROPOSALS_NMS_THRESH=0.3,
                                              TEXT_LINE_NMS_THRESH=0.99,  # 文本行之间测iou值
                                              MIN_RATIO=1.0,
                                              LINE_MIN_SCORE=0.2,
                                              TEXT_PROPOSALS_WIDTH=0,
                                              MIN_NUM_PROPOSALS=0,
                                              ),
                                  left_adjust=True,  # 对检测的文本行进行向左延伸
                                  right_adjust=True,  # 对检测的文本行进行向右延伸
                                  alph=0.2,  # 对检测的文本行进行向右、左延伸的倍数
                                  if_adjust_degree=True  # 前提是detect_angle为True(TODO : wrong, 为False时，框与图都不旋转)
                                  )
    print("\n>>> 检测识别1  总耗时:{}s".format(time.time() - t))
    for index, _ in enumerate(result):
        print(result[index]["text"])
    print("=======================================\n")

    img2 = Image.open(paths[4]).convert("RGB")
    t2 = time.time()
    _, result, angle = core.model(img2,
                                  detect_angle=DETECTANGLE,  # 是否进行文字方向检测
                                  config=dict(MAX_HORIZONTAL_GAP=80,  # 字符之间的最大间隔，用于文本行的合并
                                              MIN_V_OVERLAPS=0.6,
                                              MIN_SIZE_SIM=0.6,
                                              TEXT_PROPOSALS_MIN_SCORE=0.2,
                                              TEXT_PROPOSALS_NMS_THRESH=0.3,
                                              TEXT_LINE_NMS_THRESH=0.99,  # 文本行之间测iou值
                                              MIN_RATIO=1.0,
                                              LINE_MIN_SCORE=0.2,
                                              TEXT_PROPOSALS_WIDTH=0,
                                              MIN_NUM_PROPOSALS=0,
                                              ),
                                  left_adjust=False,  # 对检测的文本行进行向左延伸
                                  right_adjust=False,  # 对检测的文本行进行向右延伸
                                  alph=0.2,  # 对检测的文本行进行向右、左延伸的倍数
                                  if_adjust_degree=False
                                  )
    print("\n>>> 检测识别2  总耗时:{}s".format(time.time() - t2))
    for index, _ in enumerate(result):
        print(result[index]["text"])
    print("=======================================\n")

    img3 = Image.open(paths[5]).convert("RGB")
    t3 = time.time()
    _, result, angle = core.model(img3,
                                  detect_angle=DETECTANGLE,  # 是否进行文字方向检测
                                  config=dict(MAX_HORIZONTAL_GAP=80,  # 字符之间的最大间隔，用于文本行的合并
                                              MIN_V_OVERLAPS=0.6,
                                              MIN_SIZE_SIM=0.6,
                                              TEXT_PROPOSALS_MIN_SCORE=0.2,
                                              TEXT_PROPOSALS_NMS_THRESH=0.3,
                                              TEXT_LINE_NMS_THRESH=0.99,  # 文本行之间测iou值
                                              MIN_RATIO=1.0,
                                              LINE_MIN_SCORE=0.2,
                                              TEXT_PROPOSALS_WIDTH=0,
                                              MIN_NUM_PROPOSALS=0,
                                              ),
                                  left_adjust=False,  # 对检测的文本行进行向左延伸
                                  right_adjust=False,  # 对检测的文本行进行向右延伸
                                  alph=0.2,  # 对检测的文本行进行向右、左延伸的倍数
                                  if_adjust_degree=False
                                  )
    print("\n>>> 检测识别3  总耗时:{}s".format(time.time() - t3))
    for index, _ in enumerate(result):
        print(result[index]["text"])
    print("=======================================\n")


    img4 = Image.open(paths[6]).convert("RGB")
    t4 = time.time()
    _, result, angle = core.model(img4,
                                  detect_angle=DETECTANGLE,  # 是否进行文字方向检测(TODO:猜测是图方向)
                                  config=dict(MAX_HORIZONTAL_GAP=80,  # 字符之间的最大间隔，用于文本行的合并
                                              MIN_V_OVERLAPS=0.6,
                                              MIN_SIZE_SIM=0.6,
                                              TEXT_PROPOSALS_MIN_SCORE=0.2,
                                              TEXT_PROPOSALS_NMS_THRESH=0.3,
                                              TEXT_LINE_NMS_THRESH=0.99,  # 文本行之间测iou值
                                              MIN_RATIO=1.0,
                                              LINE_MIN_SCORE=0.2,
                                              TEXT_PROPOSALS_WIDTH=0,
                                              MIN_NUM_PROPOSALS=0,
                                              ),
                                  left_adjust=False,  # 对检测的文本行进行向左延伸
                                  right_adjust=False,  # 对检测的文本行进行向右延伸
                                  alph=0.2,  # 对检测的文本行进行向右、左延伸的倍数
                                  if_adjust_degree=True  # TODO:猜测是框调整
                                  )
    print("\n>>> 检测识别3  总耗时:{}s".format(time.time() - t4))
    for index, _ in enumerate(result):
        print(result[index]["text"])
    print("=======================================\n")
