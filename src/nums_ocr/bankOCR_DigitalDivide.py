import cv2 as cv
import os
import numpy as np
from .NUMPY import *
from .Line2Point import *
import math
from .DigitFilter import *

def get_image(path, size):
    image = cv.imread(path)
    image = cv.resize(image,size)
    return image

def sharpening(image):
    """
    进行图像锐化操作
    :param image:
    :return:
    """
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    dst= cv.filter2D(image,-1,kernel=kernel)
    return dst

def GasussianBlur(image,a):
    """
    进行高斯模糊
    :param image: 输入图片
    :param a: 高斯矩阵长宽
    :return: 处理后的图
    """

    kernel_size = (a,a)
    o = cv.GaussianBlur(image,kernel_size,0)
    return o;

def sobel_demo(image):
    """
    求梯度
    :param image:
    :return:
    """
    grad_x = cv.Sobel(image, cv.CV_32F, 1, 0)  # 对x求一阶导
    grad_y = cv.Sobel(image, cv.CV_32F, 0, 1)  # 对y求一阶导
    gradx = cv.convertScaleAbs(grad_x)  # 用convertScaleAbs()函数将其转回原来的uint8形式
    grady = cv.convertScaleAbs(grad_y)
    # cv.imshow("gradient_x", gradx)  # x方向上的梯度
    # cv.imshow("gradient_y", grady)  # y方向上的梯度
    gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)  # 图片融合
    return gradxy

def findContours(image):
    """
    画出外轮廓
    :param image:
    :return:
    """

    img = cv.cvtColor(image.copy(),cv.COLOR_BGR2GRAY)
    # img = cv.equalizeHist(img)
    ret,thresh = cv.threshold(img,np.average(image),255,cv.THRESH_BINARY)
    thresh = overreturn_2(thresh,255)
    contours,hier = cv.findContours(thresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

    # for cnt in contours:
    #     epsilon = 0.01*cv.arcLength(cnt,True)
    #
    #     approx = cv.approxPolyDP(cnt,epsilon,True)
    #
    #     hull = cv.convexHull(cnt)
    #     cv.drawContours(image,[hull],-1,(0,0,255),2)



    return image,thresh

def linesp(image):
    # image = img.copy()
    # gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    # edges = cv2.Canny(gray,50,150)

    # lines = cv.HoughLinesP(image,1,np.pi/180,90,minLineLength=400,maxLineGap=20)
    lines = cv.HoughLinesP(image, 1, np.pi / 180, 90, minLineLength=250, maxLineGap=20)

    result = []
    for l in lines:
        linel = Line(l[0][0],l[0][1],l[0][2],l[0][3])
        k = findk(linel)
        if(abs(k)<0.3 and length(linel) <=300):
            continue
        result.append(l)


    return result

def digit_imgs(path):
    img = get_image(path, (630, 400))
    ai = img.copy()
    imgg = img.copy()

    image = GasussianBlur(img, 11)
    image2 = sobel_demo(image)
    # image2 = sobel_demo(image2)
    image3, th = findContours(image2.copy())
    lines = linesp(th)
    # lines = lines_merge(lines)

    # for l in lines:
    #     cv.line(imgg,(l[0][0],l[0][1]),(l[0][2],l[0][3]),(0,0,255),1)

    points = Line2angleP(lines)
    # print(sorted(points))
    mat = pfind(points, image.shape, 2)
    cornerp = CornerPoints(mat, points)

    SrcPoints = np.float32([[cornerp[0].x, cornerp[0].y],
                            [cornerp[1].x, cornerp[1].y],
                            [cornerp[2].x, cornerp[2].y],
                            [cornerp[3].x, cornerp[3].y]])
    CanvasPoints = np.float32([[0, 0], [500, 0], [0, 300], [500, 300]])

    PerspectiveMatrix = cv.getPerspectiveTransform(np.array(SrcPoints), np.array(CanvasPoints))

    img2 = cv.warpPerspective(img, PerspectiveMatrix, (500, 300))

    # imshow([imgg,img2])

    h = int(img2.shape[0] / 32)
    img3 = img2[h * 16:h * 23]

    result_imgs, all_d= digit_divide(img3)

    # imshow(result_imgs)
    return result_imgs, all_d,ai

def to4imgs(imgs):
    images = []
    i = 0
    while i+3<len(imgs):
        image = np.hstack((imgs[i],imgs[i+1],imgs[i+2],imgs[i+3]))
        images.append(image)
        i = i+4


    if(i<len(imgs)):
        image = imgs[i]
        for j in range(i+1,len(imgs)):
            image = np.hstack([image,imgs[j]])
            print(len(imgs),j)


        images.append(image)

    return images










if __name__ == '__main__':
    digit_imgs("/home/external_zzh/data/bank_cardOCR/test_images/11.jpeg")