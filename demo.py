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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

# paths = glob('./test_img/4.jpg')
paths = ['./test_img/mpp.jpg', './test_img/mp1.jpg', './test_img/mp2.jpg', './test_img/mp3.jpg']

if __name__ == '__main__':
    print("OCR Starting!")
    img = Image.open(paths[0]).convert("RGB")
    t = time.time()
    # TODO if res_sorted is null
    img, res_origin, res_sorted, angle = core.model(img,
                                                    global_tune=False,  # 图片的整体大方向调整，逆时针旋转 镜像. 大约0.5s
                                                    fine_tune=False,
                                                    # 微调倾斜角（如果能保证图像水平，或者global_tune之后为水平，则不需要微调）. 大约2s
                                                    config=dict(MAX_HORIZONTAL_GAP=50,
                                                                # (1.5~2.5较佳)字符之间的最大间隔，用于文本行的合并 TODO:最好是自动计算
                                                                MIN_V_OVERLAPS=0.6,  # 小 ==》斜 TODO
                                                                MIN_SIZE_SIM=0.6,
                                                                TEXT_PROPOSALS_MIN_SCORE=0.1,
                                                                # 值越小,候选框越多（一些模棱两可的文字）
                                                                TEXT_PROPOSALS_NMS_THRESH=0.3  # 候选框非极大值抑制
                                                                ),
                                                    if_im=False,
                                                    left_adjust=False,  # 对检测的文本行进行向左延伸
                                                    right_adjust=False,  # 对检测的文本行进行向右延伸
                                                    alpha=0.2,  # 对检测的文本行进行向右、左延伸的倍数
                                                    result_typeset_opotion=0,  # 结果排版方案 TODO: other two type
                                                    )
    print("检测识别1  总耗时:{}s\n".format(time.time() - t))
    # for index, _ in enumerate(re_origin):
    #     print(res_origin[index]["text"])
    if len(res_sorted) < 1:
        print("we are sorry, no text are detected!")
    else:
        out_path = './out_result/' + paths[0][11:-4] + '.txt'
        print(out_path)
        out = open(out_path, 'w')
        for row in range(res_sorted.shape[0]):
            for col in range(res_sorted.shape[1]):
                print(res_sorted[row][col], end=''),
                out.write(res_sorted[row][col])
            print()
            out.write('\n')
        out.close()
    print("=======================================\n")
    #
    # img2 = Image.open(paths[1]).convert("RGB")
    # t2 = time.time()
    # _img, results_original, results_sorted, angle = core.model(img2,
    #                               global_tune=True,  # 图片的整体大方向调整，逆时针旋转. 镜像
    #                               fine_tune=True,  # 微调倾斜角（如果能保证图像水平，或者global_tune之后为水平，则不需要微调）
    #                               config=dict(MAX_HORIZONTAL_GAP=90,  # 字符之间的最大横向间隙，用于文本行的合并
    #                                           MIN_V_OVERLAPS=0.6,
    #                                           MIN_SIZE_SIM=0.6,
    #                                           TEXT_PROPOSALS_MIN_SCORE=0.1,  # 值越小,候选框越多（一些模棱两可的文字）
    #                                           TEXT_PROPOSALS_NMS_THRESH=0.4  # 候选框非极大值抑制
    #                                           ),
    #                               if_im=False,
    #                               left_adjust=False,  # 对检测的文本行进行向左延伸
    #                               right_adjust=False,  # 对检测的文本行进行向右延伸
    #                               alpha=0.2  # 对检测的文本行进行向右、左延伸的倍数
    #                               )
    # print("检测识别2  总耗时:{}s\n".format(time.time() - t2))
    # for index, _ in enumerate(result):
    #     print(result[index]["text"])
    # print("=======================================\n")
