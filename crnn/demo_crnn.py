# -*- coding: utf-8 -*-
# @Time    : 18-11-6 下午6:22
# @Author  : wanghai
# @Email   : 
# @File    : demo_crnn.py
# @Software: PyCharm Community Edition
from glob import glob
from PIL import Image
from crnn.crnn_ import crnnOcr as crnnOcr
import csv

paths = glob('../testA_part/all/*.jpg')
# paths = glob('./img/b.jpg')

if __name__ == '__main__':
    with open('testA_part_result_only_crnn.csv', 'w') as csv_file:
        fieldheader = ['filename', 'content']
        writer = csv.DictWriter(csv_file, fieldheader)
        writer.writeheader()
        for path in paths:
            print(path[18:-4])
            img = Image.open(path).convert('L')
            simPred = crnnOcr(img)
            print(simPred)
            row = {}  # TODO 修改为多行一次写入
            row.update(filename=path[18:-4], content=simPred)
            writer.writerow(row)

