import os


class cfg:
    """
       默认配置
    """
    global_tune = False  # 是否"大的"图片方向检测及调整,0.5s左右，水平图像可以False   逆时针旋转 镜像
    fine_tune = False  # 是否微调角度

    pwd = os.getcwd()

    # 文字方向检测
    angle_model_pb = os.path.join(pwd, "models", "Angle-model.pb")
    angle_model_pbtxt = os.path.join(pwd, "models", "Angle-model.pbtxt")
    # 图缩放
    scale = 900
    max_scale = 1500

    # True 启用opencv dnn 反之 darknet
    opencv_flag = True
    darknet_root = os.path.join(os.path.curdir, "darknet")

    # yolo 相关
    yolo_cfg = os.path.join(pwd, "models", "text.cfg")
    yolo_weights = os.path.join(pwd, "models", "text.weights")
    yolo_data = os.path.join(pwd, "models", "text.data")
    img_size = (1024, 1024)  # yolo3 输入图像尺寸(训练决定的)

    # 文本框相关
    MAX_HORIZONTAL_GAP = 70
    MIN_V_OVERLAPS = 0.6
    MIN_SIZE_SIM = 0.6
    TEXT_PROPOSALS_MIN_SCORE = 0.15
    TEXT_PROPOSALS_NMS_THRESH = 0.3
    left_adjust = False
    right_adjust = False
    alpha = 0.2

    # 是否show出框好的图
    display = True
    # 是否show出框
    if_im = False

    # ocr model
    lstm_flag = True  # OCR模型是否调用LSTM层
    GPU = True  # OCR 是否启用GPU
    chinese_model = True  # 模型选择 True:中英文模型 False:英文模型
    if chinese_model:
        if lstm_flag:
            ocr_model = os.path.join(pwd, "models", "ocr-lstm.pth")
        else:
            ocr_model = os.path.join(pwd, "models", "ocr-dense.pth")
    else:
        lstm_flag = True
        ocr_model = os.path.join(pwd, "models", "ocr-english.pth")
