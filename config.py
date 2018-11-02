import os

# True 启用opencv dnn 反之 darkent
opencvFlag = True
# yolo 安装目录
darknetRoot = os.path.join(os.path.curdir, "darknet")
pwd = os.getcwd()
yoloCfg = os.path.join(pwd, "models", "text.cfg")
yoloWeights = os.path.join(pwd, "models", "text.weights")
yoloData = os.path.join(pwd, "models", "text.data")

# 是否show出框好的图
DISPLAY = True

# 文字方向检测
AngleModelPb = os.path.join(pwd, "models", "Angle-model.pb")
AngleModelPbtxt = os.path.join(pwd, "models", "Angle-model.pbtxt")
# 图缩放
SCALE = 900
MAX_SCALE = 1500
# yolo3 输入图像尺寸
IMGSIZE = (1024, 1024)
# 是否启用LSTM crnn模型
DETECTANGLE = False  # 是否进行文字/图片方向检测及调整 TODO:目前是设置为关闭也会检测，可能耗时
LSTMFLAG = False  # OCR模型是否调用LSTM层
GPU = True  # OCR 是否启用GPU
chinsesModel = True  # 模型选择 True:中英文模型 False:英文模型
if chinsesModel:
    if LSTMFLAG:
        ocrModel = os.path.join(pwd, "models", "ocr-lstm.pth")
    else:
        ocrModel = os.path.join(pwd, "models", "ocr-dense.pth")
else:
    LSTMFLAG = True
    ocrModel = os.path.join(pwd, "models", "ocr-english.pth")
