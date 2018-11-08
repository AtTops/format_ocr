# format_ocr
Not ready yet

参考自https://github.com/chineseocr/chineseocr 主要目的是优化执行时间，优化识别结果的排版

代码梳理，比如去除一些没必要的判断，减少一些没必要的代码执行，改变传参路径
修正crnn模块bug，crnn模块可以单独运行
TODO: 从框检测筛选优化文本检测时间（源代码/模型在1080Ti上大概耗时1.6s）
TODO: 优化角度检测,角度检测调整约0.8s
TODO: 排版，不同的图，可能是横向、纵向、多列纵向等
