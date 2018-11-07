# -*- coding: utf-8 -*-
# @Time    : 18-11-6 下午12:45
# @Author  : wanghai
# @Email   : 
# @File    : demo_out_format.py
# @Software: PyCharm Community Edition

# -*- coding: utf-8 -*-
# @Time    : 18-10-24 上午11:11
# @Author  :
# @Email   :
# @File    : demo_out_format.py
# @Software: PyCharm Community Edition

import os
import core
from glob import glob
from PIL import Image
from config import cfg
import csv

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

paths = glob('./testA_part/all/*.jpg')
# paths = ['./test_img/mp.jpg', './test_img/mp1.jpg']

if __name__ == '__main__':
    print("OCR Starting!")
    with open('testA_part_result_test.csv', 'w') as csv_file:
        fieldheader = ['filename', 'content']
        writer = csv.DictWriter(csv_file, fieldheader)
        writer.writeheader()
        for path in paths:
            img = Image.open(path).convert("RGB")
            _, result, angle = core.model(img,
                                          global_tune=cfg.global_tune,  # 图片的整体大方向调整，逆时针旋转 镜像. 大约0.5s
                                          fine_tune=cfg.fine_tune,  # 微调倾斜角（如果能保证图像水平，或者global_tune之后为水平，则不需要微调）. 大约2s
                                          config=dict(MAX_HORIZONTAL_GAP=80,  # 字符之间的最大间隔，用于文本行的合并 TODO:最好是自动计算
                                                      MIN_V_OVERLAPS=0.6,  # 小 ==》斜 TODO
                                                      MIN_SIZE_SIM=0.6,
                                                      TEXT_PROPOSALS_MIN_SCORE=0.1,  # 值越小,候选框越多（一些模棱两可的文字）
                                                      TEXT_PROPOSALS_NMS_THRESH=0.4  # 候选框非极大值抑制
                                                      ),
                                          if_im=cfg.if_im,
                                          left_adjust=True,  # 对检测的文本行进行向左延伸
                                          right_adjust=True,  # 对检测的文本行进行向右延伸
                                          alpha=0.2  # 对检测的文本行进行向右、左延伸的倍数
                                          )
            content = ''
            row = {}  # TODO 修改为多行一次写入
            for index, _ in enumerate(result):
                content += result[index]["text"]
            row.update(filename=path[17:-4], content=content)
            writer.writerow(row)
        # writer.writerow({'mp': '7-DForc.郭美a刚g足扇僧:29370777707900273970ee0.o)110193721:2997311国警', 'mp1': '合号单币鄂昨eo电舒y营柱国)警翠女义职翠餐武署益罩发兴'})
