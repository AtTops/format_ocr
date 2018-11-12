# -*- coding: utf-8 -*-
# @Time    : 18-11-6 下午6:22
# @Author  : wanghai
# @Email   : 
# @File    : demo_crnn.py
# @Software: PyCharm Community Edition
# 文件分类以及识别“正常”尺寸的图像
from glob import glob
from PIL import Image
from crnn.crnn_ import crnnOcr as crnnOcr
import csv
import os, shutil

paths = glob('../testA_part/all/*.jpg')


# paths = glob('./img/b.jpg')

def movefile(srcfile, dstfile, file_type):
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(dstfile)  # 分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)  # 创建路径
        shutil.move(srcfile, dstfile + '.' + file_type)  # 移动文件
        print("move %s -> %s" % (srcfile, dstfile))


def copyfile(srcfile, dstfile, file_type):
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(dstfile)  # 分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)  # 创建路径
        shutil.copyfile(srcfile, dstfile + '.' + file_type)  # 复制文件
        print("copy %s -> %s" % (srcfile, dstfile))


if __name__ == '__main__':
    thin_dstfile_root = '../testA_part/thin/'
    none_dstfile_root = '../testA_part/none/'
    with open('test_not_none_crnn.csv', 'w') as csv_file:
        fieldheader = ['filename', 'content']
        writer = csv.DictWriter(csv_file, fieldheader)
        writer.writeheader()
        count = 0
        for path in paths:
            fname = path[18:-4]
            img = Image.open(path).convert('L')
            width, height = img.size
            # 避免太细长放入网络出错
            if height / width > 7:
                print("thin")
                # copy to thin directory
                copyfile(path, thin_dstfile_root + fname, 'jpg')
                #
            else:
                simPred = crnnOcr(img)
                if len(simPred) > 0:
                    print('==> ', simPred)
                    row = {}  # TODO 修改为多行一次写入
                    row.update(filename=path[18:-4], content=simPred)
                    writer.writerow(row)
                else:
                    count += 1
                    copyfile(path, none_dstfile_root + fname, 'jpg')
        print('done! ', count)
