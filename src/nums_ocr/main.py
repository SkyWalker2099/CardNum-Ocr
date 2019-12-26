import os
import numpy as np
import random
import string


from keras.models import Model
from keras.layers import Input, Lambda, Activation, Conv2D, MaxPooling2D, ZeroPadding2D, Reshape, Concatenate,Flatten,Dense,Dropout,GRU,LSTM,Add
from keras.regularizers import l2
import keras.backend as K
from keras.callbacks import *
from keras.models import load_model
from keras.utils.vis_utils import plot_model

from src.nums_ocr.model_crnn_ctc import get_crnn_ctc_model
from src.nums_ocr.Train_Predict import *
from src.nums_ocr.bankOCR_DigitalDivide import *
import cv2 as cv
import numpy as np

# conv, base, ctc_model = get_crnn_ctc_model()
# plot_model(ctc_model, to_file='model/ctc_model.png', show_shapes=True)

# # 统计test文件夹里的图片识别率
#
# # 识别单张图片（4 numbers）
#


model = Train_Predict()
model.Load_model(filepath='model/crnn_ctc_base_model_v1.h5')

def predict(path):
    try:

        imgs, imgg, aimg = digit_imgs(
            path)

        images = to4imgs(imgs)

        result = ""

        for image in images:
            img = image.copy()
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            pred = model.preict_img(image)
            result += pred

        print(result)

        # imshow([imgg])
        cv.imshow(result, aimg)
        cv.waitKey()
        return result
    except Exception as e:
        print(path + "\n  error")

if __name__ == '__main__':

    result = predict("/home/external_zzh/PycharmProjects/CardNum/src/nums_ocr/datasets/test_images/7.jpeg")
    predict(result)