import os

# from PIL import Image

import numpy as np
import random
import string

from keras.models import Model
from keras.layers import Input, Lambda, Activation, Conv2D, MaxPooling2D, ZeroPadding2D, Reshape, Concatenate,Flatten,Dense,Dropout,GRU,LSTM,Add
from keras.regularizers import l2
import keras.backend as K
from keras.callbacks import *
from keras.models import load_model

#print(os.getcwd())

char_set=string.digits
#识别字符串最大长度
seq_len=4
label_count=len(char_set)+1
image_size=(120, 46)

IMAGE_HEIGHT=image_size[1]
IMAGE_WIDTH=image_size[0]

base_dir=os.getcwd()
train_dir=os.path.join(base_dir, 'datasets/bankcardnum')

def get_label(filename):
    label=[]
    num=filename[:4]
    for n in num:
        if n != '_':
            label.append(int(char_set.find(n)))
    if len(label) < seq_len:
        cur_seq_len = len(label)
        for i in range(seq_len-cur_seq_len):
            label.append(label_count)
    return label


def get_im_by_paths(paths, mydir=train_dir, height=IMAGE_HEIGHT, width=IMAGE_WIDTH):
    imgs = np.zeros((len(paths), width, height, 3))
    i = 0
    for path in paths:
        path = os.path.join(mydir, path)
        img = Image.open(path).convert('RGB')
        img = np.asarray(img, dtype='float32') / 255.
        img = np.transpose(img, (1, 0, 2))
        imgs[i, :, :, :] = img
        i += 1

    return imgs


def load_test():
    test_dir = os.path.join(base_dir, 'datasets/test')
    paths = os.listdir(test_dir)
    x_test = np.zeros((len(paths), IMAGE_WIDTH, IMAGE_HEIGHT, 3))
    y_test = []
    i = 0
    for path in paths:
        img = Image.open(os.path.join(test_dir, path)).convert('RGB')
        img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)
        img = np.asarray(img, dtype='float32') / 255.
        # print(img.shape)
        img = np.transpose(img, (1, 0, 2))
        # plt.imshow(img)
        # plt.show()
        x_test[i, :, :, :] = img
        y_test.append(get_label(path))
        i += 1

    return x_test, y_test


def load_train():
    x_train = os.listdir(train_dir)
    y_train = []
    for name in x_train:
        y_train.append(get_label(name))
    return x_train, y_train

def im_generator(x_train, y_train, batch_size, conv_shape):
    '''
        @:param
            x_train: image_path
            y_train: labels
            batch_size
        @:return
            generator: (x:imgs, y:labels)
        '''
    while 1:
        for i in range(0, len(x_train), batch_size):
            x = get_im_by_paths(x_train[i:i + batch_size])
            y = y_train[i:i + batch_size]
            y = np.asarray(y)
            yield [x, y, np.ones(batch_size) * int(conv_shape[1] - 2),
                   np.ones(batch_size) * seq_len], y