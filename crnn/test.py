# -*- coding: utf-8 -*-
"""
这个测试不建议使用，可使用demo.py
"""

import torch.utils.data
from torch.autograd import Variable
from crnn.util import strLabelConverter
import crnn.dataset as dataset
from PIL import Image
import crnn.models.crnn as crnn
import crnn.keys as keys

alphabet = keys.alphabetChinese
print("len(alphabet): %d" % len(alphabet))
converter = strLabelConverter(alphabet)
model = crnn.CRNN(32, 1, len(alphabet) + 1, 256, 1).cuda()
path = './samples/model_acc97.pth'
model.load_state_dict(torch.load(path))
print(model)

while 1:
    im_path = "./img/a.jpg"
    image = Image.open(im_path).convert('L')
    scale = image.size[1] * 1.0 / 32
    w = int(image.size[0] / scale)
    print(w)

    transformer = dataset.resizeNormalize((w, 32))
    image = transformer(image).cuda()
    image = image.view(1, *image.size())
    image = Variable(image)
    model.eval()
    preds = model(image)
    _, preds = preds.max(2)
    # preds = preds.squeeze(1)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    print('%-20s => %-20s' % (raw_pred, sim_pred))
