from config import yoloCfg, yoloWeights
from config import AngleModelPb, AngleModelPbtxt
from config import IMGSIZE  # 输出图像的size
import numpy as np
import cv2
import time

textNet = cv2.dnn.readNetFromDarknet(yoloCfg, yoloWeights)
# 文字方向检测模型
angleNet = cv2.dnn.readNetFromTensorflow(AngleModelPb, AngleModelPbtxt)


def text_detect(img):
    thresh = 0.1
    h, w = img.shape[:2]
    # 二值图像几何形状提取，Blob几何调整
    inputBlob = cv2.dnn.blobFromImage(img, scalefactor=0.00390625, size=IMGSIZE, swapRB=True, crop=False)
    textNet.setInput(inputBlob)
    t = time.time()
    # look = np.squeeze(inputBlob)
    # cv2.imshow("image_blob", np.transpose(look, [1, 2, 0]))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    pred = textNet.forward()
    print("yolo3找出所有的本文框:{}s".format(time.time() - t))
    cx = pred[:, 0] * w
    cy = pred[:, 1] * h
    xmin = cx - pred[:, 2] * w / 2
    xmax = cx + pred[:, 2] * w / 2
    ymin = cy - pred[:, 3] * h / 2
    ymax = cy + pred[:, 3] * h / 2
    scores = pred[:, 4]
    indx = np.where(scores > thresh)[0]
    scores = scores[indx]
    boxes = np.array(list(zip(xmin[indx], ymin[indx], xmax[indx], ymax[indx])))
    return boxes, scores


def angle_detect(img):
    """
    总体大方向检测
    """

    ROTATE = [0, 90, 180, 270]
    # h, w = img.shape[:2]
    # thesh = 0.05
    # xmin, ymin, xmax, ymax = int(thesh * w), int(thesh * h), w - int(thesh * w), h - int(thesh * h)
    # 剪切图片边缘 TODO:如果周围有文字呢？
    # img = img[ymin:ymax, xmin:xmax]
    # swapRB，是选择是否交换R与B颜色通道，一般用opencv读取caffe的模型就需要将这个参数设置为false，读取tensorflow的模型，
    # 则默认选择True即可，这样才不会出现在opencv框架和tensorflow框架下，object detection检测效果不同。
    inputBlob = cv2.dnn.blobFromImage(img, scalefactor=1.0,
                                      size=(224, 224),
                                      swapRB=True,
                                      mean=[103.939, 116.779, 123.68], crop=False)
    angleNet.setInput(inputBlob)
    pred = angleNet.forward()
    index = np.argmax(pred, axis=1)[0]
    return ROTATE[index]
