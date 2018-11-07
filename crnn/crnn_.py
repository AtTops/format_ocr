# coding:utf-8
import sys
import torch
import torch.utils.data
from crnn.util import strLabelConverter
import crnn.dataset as dataset
import crnn.models.crnn as crnn
import crnn.keys as keys
from torch.autograd import Variable
from collections import OrderedDict
from config import ocrModel, LSTMFLAG, GPU, chinsesModel

sys.path.insert(1, "./crnn")


def crnnSource():
    if chinsesModel:
        alphabet = keys.alphabetChinese
    else:
        alphabet = keys.alphabetEnglish

    converter = strLabelConverter(alphabet)
    if torch.cuda.is_available() and GPU:
        model = crnn.CRNN(32, 1, len(alphabet) + 1, 256, 1, lstmFlag=LSTMFLAG).cuda()  ##LSTMFLAG=True crnn 否则 dense ocr
    else:
        model = crnn.CRNN(32, 1, len(alphabet) + 1, 256, 1, lstmFlag=LSTMFLAG).cpu()

    state_dict = torch.load(ocrModel, map_location=lambda storage, loc: storage)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')  # remove `module.` torch的版本问题
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    model.eval()

    return model, converter


# 加载模型
model, converter = crnnSource()


def crnnOcr(image):
    """
    crnn模型，自然场景端到端识别
    :param image: 'L' image
    :return: box识别后的text
    """
    scale = image.size[1] * 1.0 / 32
    w = int(image.size[0] / scale)
    transformer = dataset.resizeNormalize((w, 32))
    if torch.cuda.is_available() and GPU:
        image = transformer(image).cuda()
    else:
        image = transformer(image).cpu()

    image = image.view(1, *image.size())
    image = Variable(image)
    model.eval()
    preds = model(image)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    return sim_pred
