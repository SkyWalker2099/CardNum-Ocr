import cv2
import os
import numpy as np
path = os.listdir("D:/cards/images/")
i = 0
for picture in path:
    pic = cv2.imread('D:/cards/images/'+picture) #读入图片
    j=1
    while(j<=10):
        contrast = 0.1*j        #对比度
        brightness = 100    #亮度
        pic_turn = cv2.addWeighted(pic,contrast,pic,0,brightness)
        #cv2.addWeighted(对象,对比度,对象,对比度)
        '''cv2.addWeighted()实现的是图像透明度的改变与图像的叠加'''
        picture_list = list(picture)
        nPos = picture_list.insert(5, '_'+str(i)+str(j))
        picture = ''.join(picture_list)
        cv2.imwrite('D:/cards/KreasReshape/' + picture, pic_turn)
        j=j+1
    i = i + 1
    if i > 2167:
        break
    # 第四次
    # 二十张图片