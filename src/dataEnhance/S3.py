import cv2
import os
import random
from numpy import *
def PepperandSalt(src, percetage):
    NoiseImg = src
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):
        randX = random.random_integers(0, src.shape[0] - 1)
        randY = random.random_integers(0, src.shape[1] - 1)
        if random.random_integers(0, 1) <= 0.5:
            NoiseImg[randX, randY] = 0
        else:
            NoiseImg[randX, randY] = 255
    return NoiseImg


def GaussianNoise(src, means, sigma):
    NoiseImg = src
    rows = NoiseImg.shape[0]
    cols = NoiseImg.shape[1]
    for i in range(rows):
        for j in range(cols):
            NoiseImg[i, j] = NoiseImg[i, j] + random.gauss(means, sigma)
            if NoiseImg[i, j] < 0:
                NoiseImg[i, j] = 0
            elif NoiseImg[i, j] > 255:
                NoiseImg[i, j] = 255
    return NoiseImg

path = os.listdir("D:/cards/S2/")
i=0
for picture in path:

    img = cv2.imread('D:/cards/S2/'+picture, 0)
    img1 = PepperandSalt(img, 0.03)
    picture_list=list(picture)
    nPos=picture_list.insert(5,str(i))
    picture=''.join(picture_list)
    cv2.imwrite('D:/cards/S3/'+picture, img1)
    # cv2.imshow('PepperandSalt', img1)
    # cv2.waitKey(0)
    i=i+1
    #第三次