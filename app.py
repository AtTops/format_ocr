# -*- coding: utf-8 -*-
# @Time    : 18-11-13 下午4:31
# @Author  : wanghai
# @Email   :
# @File    : app.py
# @Software: PyCharm Community Edition
# TODO add logs

import os
import core
import time
import json
from glob import glob
from PIL import Image
from flask import Flask, render_template, request, jsonify
from core_helper.type_config import select_config

app = Flask(__name__)

# UPLOAD_FOLDER = os.path.basename('/users/uploads') # last '/', so:uploads
app.config['UPLOAD_FOLDER'] = './users/uploads'
app.config['JSON_AS_ASCII'] = False  # 否则在web端显示的是utf-8的编码，而不是中文（但是不影响接口的读取）
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


# paths = ['./test_img/mp.jpg', './test_img/mp1.jpg', './test_img/mp2.jpg', './test_img/mp3.jpg']

def ocr(absolute_path, img_type, tune_angle=0):
    print("OCR Starting!")
    img = Image.open(absolute_path).convert("RGB")
    t = time.time()
    if tune_angle == 1:
        global_tune = True
        fine_tune = True
    else:
        global_tune = False
        fine_tune = False
    params = select_config(img_type)
    img, res_origin, res_sorted, angle = core.model(img,
                                                    global_tune=global_tune,  # 图片的整体大方向调整，逆时针旋转 镜像. 大约0.5s
                                                    fine_tune=fine_tune,
                                                    # 微调倾斜角（如果能保证图像水平，或者global_tune之后为水平，则不需要微调）. 大约2s
                                                    config=params[0],
                                                    if_im=params[1],
                                                    left_adjust=params[2],  # 对检测的文本行进行向左延伸
                                                    right_adjust=params[3],  # 对检测的文本行进行向右延伸
                                                    alpha=params[4],  # 对检测的文本行进行向右、左延伸的倍数
                                                    result_typeset_opotion=0,  # 结果排版方案 TODO: other two type
                                                    )
    print("检测识别1  总耗时:{}s\n".format(time.time() - t))
    json_str = ''
    # for index, _ in enumerate(re_origin):
    #     print(res_origin[index]["text"])
    if len(res_sorted) < 1:
        # TODO: return
        return 'we are sorry, no text detected!'
    else:
        basename = os.path.basename(absolute_path)
        print(basename)
        out_path = './users/out_result/' + basename[:-4] + '.txt'
        print(out_path)
        out = open(out_path, 'w')
        for row in range(res_sorted.shape[0]):
            for col in range(res_sorted.shape[1]):
                json_str += res_sorted[row][col]
                out.write(res_sorted[row][col])
            json_str += '\n'
            out.write('\n')
        out.close()
    print(json_str)
    return json_str


# 主页
@app.route('/')
def hello_world():
    return render_template('index.html')


# 上传图片，服务器下载, 识别
@app.route('/ocr/api/v1.0', methods=['POST'])
def upload_ocr():
    if request.method == 'POST':
        # TOTO: b64encode
        t0 = time.time()
        file = request.files['upl_img']
        store_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        try:
            file.save(store_path)
        except IOError:
            return jsonify(
                {'code': -1, 'data': '获取图像数据失败！', 'take_time': round(time.time() - t0, 3), 'desc': 'failure'})
        param_type = request.form.get("img_type", type=int)
        param_tune = request.form.get("tune_angle", type=int)
        if 0 < param_type < 5:
            # TODO: 更加细化的提示（前台解释）
            try:
                data = ocr(store_path, img_type=param_type, tune_angle=param_tune)
            except:  # TODO: the size
                return json.dumps(
                    {'code': 0, 'data': 'Illegal img.Please confirm that the image size is larger than ?',
                     'take_time': round(time.time() - t0, 3), 'desc': 'success'},
                    ensure_ascii=False)
            return json.dumps(
                {'code': 0, 'data': data, 'take_time': round(time.time() - t0, 3), 'desc': 'success'},
                ensure_ascii=False)
        else:
            return jsonify(
                {'code': -1, 'data': 'Illegal img_type, must be 1~4', 'take_time': round(time.time() - t0, 3),
                 'desc': 'failure'})
    else:
        return jsonify({'code': -1, 'data': 'Method not allowed, must be POST', 'take_time': 0, 'desc': 'failure'})


if __name__ == '__main__':
    app.run(debug=True, port=8008)
