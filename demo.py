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

# paths = glob('./test_img/mp*.jpg')
paths = ['./test_img/mp.jpg', './test_img/mp1.jpg', './test_img/mp2.jpg', './test_img/mp3.jpg']

if __name__ == '__main__':
    print("OCR Starting!")
    img = Image.open(paths[0]).convert("RGB")
    t = time.time()
    _, result, angle = core.model(img,
                                  global_tune=False,  # 图片的整体大方向调整，逆时针旋转 镜像. 大约0.5s
                                  fine_tune=True,  # 微调倾斜角（如果能保证图像水平，或者global_tune之后为水平，则不需要微调）. 大约2s
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
                                  left_adjust=False,  # 对检测的文本行进行向左延伸
                                  right_adjust=False,  # 对检测的文本行进行向右延伸
                                  alph=0.2  # 对检测的文本行进行向右、左延伸的倍数
                                  )
    print("检测识别1  总耗时:{}s\n".format(time.time() - t))
    for index, _ in enumerate(result):
        print(result[index]["text"])
    print("=======================================\n")

    img2 = Image.open(paths[1]).convert("RGB")
    t2 = time.time()
    _, result, angle = core.model(img2,
                                  global_tune=DETECTANGLE,  # 图片的整体大方向调整，逆时针旋转 镜像
                                  fine_tune=True,  # 微调倾斜角（如果能保证图像水平，或者global_tune之后为水平，则不需要微调）
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
                                  right_adjust=True,  # 对检测的文本行进行向右延伸
                                  alph=0.2  # 对检测的文本行进行向右、左延伸的倍数
                                  )
    print("检测识别2  总耗时:{}s\n".format(time.time() - t2))
    for index, _ in enumerate(result):
        print(result[index]["text"])
    print("=======================================\n")

    img3 = Image.open(paths[2]).convert("RGB")
    t3 = time.time()
    _, result, angle = core.model(img3,
                                  global_tune=False,  # 图片的整体大方向调整，逆时针旋转 镜像
                                  fine_tune=False,  # 微调倾斜角（如果能保证图像水平，或者global_tune之后为水平，则不需要微调）
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
                                  left_adjust=True,  # 对检测的文本行进行向左延伸
                                  right_adjust=False,  # 对检测的文本行进行向右延伸
                                  alph=0.2  # 对检测的文本行进行向右、左延伸的倍数
                                  )
    print("检测识别3  总耗时:{}s\n".format(time.time() - t3))
    for index, _ in enumerate(result):
        print(result[index]["text"])
    print("=======================================\n")

    img4 = Image.open(paths[3]).convert("RGB")
    t4 = time.time()
    _, result, angle = core.model(img4,
                                  global_tune=DETECTANGLE,  # 图片的整体大方向调整，逆时针旋转 镜像
                                  fine_tune=False,  # 微调倾斜角（如果能保证图像水平，或者global_tune之后为水平，则不需要微调）
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
                                  left_adjust=True,  # 对检测的文本行进行向左延伸
                                  right_adjust=True,  # 对检测的文本行进行向右延伸
                                  alph=0.2  # 对检测的文本行进行向右、左延伸的倍数
                                  )
    print("检测识别4  总耗时:{}s\n".format(time.time() - t4))
    for index, _ in enumerate(result):
        print(result[index]["text"])
    print("=======================================\n")
