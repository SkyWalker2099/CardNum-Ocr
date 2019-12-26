import numpy as np
import cv2 as cv
import os
import math
from .Line2Point import *
import operator

def overreturn(image,max):
    x = image.shape[0]
    y = image.shape[1]

    one = np.ones(image.shape,np.uint8)*max

    # lu = image[0][0]
    # ru = image[0][y-1]
    # ld = image[x-1][0]
    # rd = image[x-1][y-1]

    # print(lu,ru,ld,rd)

    x = int(x/2)
    y = int(y/2)
    m = image[x-50 :x+50, y-50:y+50]
    print(m.shape)
    avg = m.sum()/10000
    print(avg)
    if(avg <= 127):
        image = one - image


    return image

def overreturn_2(image,max):
    avg = np.average(image)
    one = np.ones(image.shape, np.uint8) * max
    if(avg > 177):
        image = one - image

    return image

def highLuminate(image,threshold):
    """
    粗糙的算法检测高光
    :param image:
    :param threshold:
    :return:
    """
    x = image.shape[0]
    y = image.shape[1]
    gray = cv.cvtColor(image,cv.COLOR_RGB2GRAY)
    ret,thresh = cv.threshold(gray,threshold,255,cv.THRESH_BINARY)

    return thresh

def ColourDistance(rgb_1, rgb_2):
    R_1, G_1, B_1 = rgb_1
    R_2, G_2, B_2 = rgb_2
    rmean = (R_1 + R_2) / 2
    R = R_1 - R_2
    G = G_1 - G_2
    B = B_1 - B_2
    return math.sqrt((2 + rmean / 256) * (R ** 2) + 4 * (G ** 2) + (2 + (255 - rmean) / 256) * (B ** 2))

def pixelLuminate(pixel):
    b,g,r = pixel
    l = 0.3*r+0.6*g+0.1*b
    print(l)
    return l

def highLightPixelRid(image,th):
    img = image.copy()
    x = img.shape[0]
    y = img.shape[1]

    for i in range(1,x):
        for j in range(1,y):
            if pixelLuminate(image[i][j]) > th:
                img[i][j] = img[i][j-1]

    return img

def line_theta(line):
    x1,y1,x2,y2 = line
    if x2-x1 == 0:
        return 90
    theta = math.atan((y2-y1)/(x2-x1))
    theta = int(theta/math.pi * 180)
    return theta

def lines_merge(lines):
    alines = []
    alines.append(lines[0][0])

    for i in range(1,len(lines)):
        flag = False
        for j in range(len(alines)):

            if g2(alines[j],lines[i][0]) == True:
                alines[j] = merge(alines[j],lines[i][0])
                flag = True
                break
        if flag == False:
            alines.append(lines[i][0])

    return alines

def merge(l1,l2):
    _1x1, _1y1, _1x2, _1y2 = l1
    _2x1, _2y1, _2x2, _2y2 = l2
    x1 = int((_1x1 + _2x1)/2)
    y1 = int((_1y1 + _2y1) / 2)
    x2 = int((_1x2 + _2x2) / 2)
    y2 = int((_1y2 + _2y2) / 2)
    return [x1,y1,x2,y2]

def pfind(points,shape,th):
    mat = np.zeros((shape[0],shape[1]))

    for p in points:
        x = p.x
        y = p.y
        for i in range(x-3,x+3):
            for j in range(y-3,y+3):
                if(j >= shape[0] or i >= shape[1] or i<0 or j<0):
                    continue
                mat[j][i] = mat[j][i] + 1

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if (mat[i][j] > 0):
                # print(i, j, mat[i][j])
                mat[i][j] = max(0, mat[i][j] - 1)

    return mat

def CornerPoints(mat,points):

    y = mat.shape[0] # 400
    x = mat.shape[1] # 630


    lpoints = [point for point in points if point.x < (x/2-10)]
    rpoints = [point for point in points if point.x > (x/2+10)]

    lpoints.sort(key=operator.attrgetter("y","x"))
    rpoints.sort(key=operator.attrgetter("y","x"))

    plu = anglep(mat,lpoints[0])
    pld = anglep(mat,lpoints[-1])

    pru = anglep(mat,rpoints[0])
    prd = anglep(mat,rpoints[-1])

    # print(plu,pld,pru,prd)

    return (plu,pru,pld,prd)

def anglep(mat,point):
    x = point.x
    y = point.y
    m = -1
    mx = 0
    my = 0
    for i in range(x-10,x+10):
        for j in range(y-10,y+10):
            try:

                if(m<mat[j][i]):
                    mx = i
                    my = j
                    m = mat[j][i]
            except Exception as e:
                pass
    return Point(mx,my)



if __name__ == '__main__':

    pass