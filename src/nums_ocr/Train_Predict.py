import os

import cv2 as cv

import numpy as np
import random
import string

from keras.models import Model
from keras.layers import Input, Lambda, Activation, Conv2D, MaxPooling2D, ZeroPadding2D, Reshape, Concatenate,Flatten,Dense,Dropout,GRU,LSTM,Add
from keras.regularizers import l2
import keras.backend as K
from keras.callbacks import *
from keras.models import load_model


from src.nums_ocr.dataset_load import load_test, load_train, im_generator
from src.nums_ocr.model_crnn_ctc import get_crnn_ctc_model

char_set=string.digits
#识别字符串最大长度
seq_len=4
label_count=len(char_set)+1
image_size=(120, 46)

IMAGE_HEIGHT=image_size[1]
IMAGE_WIDTH=image_size[0]

base_dir=os.getcwd()


class Train_Predict:
    def __init__(self):
        self.conv_shape = None
        self.base_model = None
        self.ctc_model = None

    # def build_model(self):
    #     self.conv_shape, self.base_model, self.ctc_model = get_crnn_ctc_model()
    #
    # def predict(self):
    #     digits_blank = char_set + ' '
    #     x_test, y_test = load_test()
    #     y_pred = self.base_model.predict(x_test)
    #     y_pred = y_pred[:, 2:, :]
    #
    #     # test_index = 19
    #     x_len = len(x_test)
    #     print(x_len)
    #     out = K.get_value(K.ctc_decode(y_pred,
    #                                    input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1])[0][0])[:, :4]
    #
    #     # print wrong predictions
    #     err = 0
    #     for i in range(x_len):
    #         pred = ''.join([digits_blank[x] for x in out[i] if x != -1])
    #         true = ''.join([digits_blank[x] for x in y_test[i] if x != 11])
    #         if pred != true:
    #             err += 1
    #             print('true:%s pred:%s' % (true, pred))
    #     print('accuracy:', (x_len - err) / x_len)

    def preict_img(self, img):
        '''
        @:param
            img: image_file .jpg ...
        @:return
            predicted nums
        '''

        image = img.copy()
        # img = img.convert('RGB')
        # img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        img = cv.resize(img,(IMAGE_WIDTH, IMAGE_HEIGHT))

        img = np.asarray(img, dtype='float32') / 255.

        # print(img.shape)
        img = np.transpose(img, (1, 0, 2))
        # plt.imshow(img)
        # plt.show()
        digits_blank = char_set + ' '

        img = img[np.newaxis, :]

        # print(img.shape)

        y_pred = self.base_model.predict(img)
        y_pred = y_pred[:, 2:, :]
        out = K.get_value(K.ctc_decode(y_pred,
                input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1])[0][0])[:, :4]

        pred = ''.join([digits_blank[x] for x in out[0] if x != -1])

        # cv.imshow(pred, image)
        # cv.waitKey()
        # cv.destroyAllWindows()

        return pred

    # def train(self, batch_size=128, epochs=100):
    #     X_train, Y_train = load_train()
    #     # 设置训练集、验证集
    #     total = len(X_train)
    #     # print(total)
    #     max = 85760
    #     maxin = int(max * 0.8)
    #     x_valid = X_train[maxin:max]
    #     y_valid = Y_train[maxin:max]
    #     x_train = X_train[:maxin]
    #     y_train = Y_train[:maxin]
    #     # print(total, maxin)
    #     batch_size = batch_size
    #     train_steps = maxin // batch_size
    #     # print(train_steps)
    #     valid_steps = (total - maxin) // batch_size
    #     # print(valid_steps)
    #     conv_shape = self.conv_shape
    #
    #     history = self.ctc_model.fit_generator(generator=im_generator(x_train, y_train, batch_size, conv_shape),
    #                                            steps_per_epoch=train_steps,
    #                                            epochs=epochs,
    #                                            validation_data=im_generator(x_valid, y_valid, batch_size, conv_shape),
    #                                            validation_steps=valid_steps)
    #     return history
    #
    # def Save_model(self, filepath):
    #     self.base_model.save(filepath+'base')
    #     self.ctc_model.save(filepath+'ctc')
    #
    def Load_model(self, filepath):
        self.base_model = load_model(filepath)

    def Load_model_ctc(self, filepath):
        self.ctc_model = load_model(filepath)
