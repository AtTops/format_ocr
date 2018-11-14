# -*- coding: utf-8 -*-
# @Time    : 18-11-14 下午5:47
# @Author  : wanghai
# @Email   :
# @File    : app.py
# @Software: PyCharm Community Edition


def select_config(img_type):
    # 名片
    param1 = [dict(MAX_HORIZONTAL_GAP=50,
                   MIN_V_OVERLAPS=0.6,
                   MIN_SIZE_SIM=0.6,
                   TEXT_PROPOSALS_MIN_SCORE=0.1,
                   TEXT_PROPOSALS_NMS_THRESH=0.3),
              False,
              False,
              False,
              0.2]
    param2 = [dict(MAX_HORIZONTAL_GAP=50,
                   MIN_V_OVERLAPS=0.6,
                   MIN_SIZE_SIM=0.6,
                   TEXT_PROPOSALS_MIN_SCORE=0.1,
                   TEXT_PROPOSALS_NMS_THRESH=0.3),
              False,
              False,
              False,
              0.2]
    param3 = [dict(MAX_HORIZONTAL_GAP=50,
                   MIN_V_OVERLAPS=0.6,
                   MIN_SIZE_SIM=0.6,
                   TEXT_PROPOSALS_MIN_SCORE=0.1,
                   TEXT_PROPOSALS_NMS_THRESH=0.3),
              False,
              False,
              False,
              0.2]
    param4 = [dict(MAX_HORIZONTAL_GAP=50,
                   MIN_V_OVERLAPS=0.6,
                   MIN_SIZE_SIM=0.6,
                   TEXT_PROPOSALS_MIN_SCORE=0.1,
                   TEXT_PROPOSALS_NMS_THRESH=0.3),
              False,
              False,
              False,
              0.2]
    params = {1: param1, 2: param2, 3: param3, 4: param4}
    return params.get(img_type, None)


if __name__ == '__main__':
    a = select_config(1)
    print(a[0])
    print(a[2])
