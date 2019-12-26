import numpy as np
import cv2 as cv
import math
import operator
from .NUMPY import *

def imshow(images):
    s = 0
    for image in images:
        cv.imshow(str(s),image)
        s = s+1
    cv.waitKey(0)

def sharpening(image):
    """
    进行图像锐化操作
    :param image:
    :return:
    """
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    dst= cv.filter2D(image,-1,kernel=kernel)
    return dst

def piecewiseToGray(image,x1,x2,y1,y2):
    img = np.zeros(image.shape)
    x,y = image.shape

    for i in range(y):
        for j in  range(x):
            r = image[j][i]
            if(r < x1):
                img[j][i] = int(y1*r/x1)
            elif(r > x2 ):
                img[j][i] = int((r-x2)*(255-y2)/(255-x2)+y2)
            else:
                img[j][i] = int((r-x1)*(y2-y1)/(x2-x1) + y1)
    # print(img)
    return img

def edge(image):

    blurred = cv.GaussianBlur(image,(3,3),0)
    gray = cv.cvtColor(blurred,cv.COLOR_BGR2GRAY)
    edge_output = cv.Canny(gray,50,150)
    return edge_output

def bgr2yuv(image):
    image = sharpening(image)
    yuvimage = cv.cvtColor(image,cv.COLOR_RGB2YUV)
    y,u,v = cv.split(yuvimage)
    return y,u,v


def backgroundFilter(image):

    y,u,v = bgr2yuv(image)

    u = cv.Canny(u,40,120)
    v = cv.Canny(v,40,120)

    t = u+v
    p = sum_col(t)
    return p


def sum_row(image):
    h = image.shape[0]
    p = []
    for i in range(h):
        s = np.sum(image[i])
        # print(s)
        p.append(s)
    return p

def sum_col(image):
    w = image.shape[1]
    p = []
    for i in range(w):
        s = np.sum(image[:,i])
        p.append(s)
    return p

def row_locate(image):
    p = sum_row(image)
    h = image.shape[0]

    step = int(h/4)
    st = step
    ed = h - step

    max = -1
    loc = -1


    for i in range(st,ed):
        m = sum(p[i-step:i+step])
        if(m > max):
            max = m
            loc = i
    st = loc - step
    ed = loc + step
    return st,ed,p

def row_cut(image):
    st,ed,p = row_locate(image)
    image = image[st:ed]
    return st,ed,image

def is_black(image):
    """
    是凹凸型还是黑色数字
    :param image:
    :return:
    """
    hsv = cv.cvtColor(image,cv.COLOR_BGR2HSV)
    v = cv.split(hsv)[2]

    x,y = v.shape
    t = 0
    for i in range(x):
        for j in range(y):
            if v[i][j] <= 55:
                t+=1
    p = t/(x*y)
    # print(p)
    if(p>0.08 and p< 0.25):

        return 1

    else:
        return 2


def col_locate_black(image):
    img = image.copy()
    image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    def variance(image):
        aver = np.average(image)
        w,h = image.shape
        s = 0
        for i in range(w):
            for j in range(h):
                s+=math.pow(image[i][j] - aver,2)
        return s

    def black_count2(image):
        th = cv.threshold(image, np.average(image), 255, cv.THRESH_BINARY)[1]

        s = np.sum(th)
        ts = np.sum(np.ones(s.shape)*255)
        return s/ts


    def black_count(image,n):

        if(variance(image) < 30000):
            return False


        th = cv.threshold(image, np.average(image), 255, cv.THRESH_BINARY)[1]
        p = th[:,n]
        p = np.ones(p.shape)*255 - p

        p2 = th[:,-1-n]
        p2 = np.ones(p2.shape)*255 - p2

        ps = np.sum(p)
        # imshow([p])
        s = np.sum(th)
        ts = np.sum(np.ones(s.shape)*255)
        # print(ps,s/ts)
        if(ps < 255 and (s/ts > 150 and s/ts < 200)):

            # imshow([th,p,p2])
            return True
        else:
            return False


    x,y = image.shape
    st = 0
    ed = 0
    for i in range(0,y-20):
        # print(i)
        if black_count(image[:,i:i+10],0) and black_count2(image[:,i:i+30]) > 360:
            # print(black_count2(image[:,i:i+30]))
            st = i
            break

    for i in range(y,20,-1):
        # print(i)
        if black_count(image[:,i-10:i],-1) and black_count2(image[:, i - 30:i]) > 360 :
            #

            ed = i
            break



    return st,ed

def col_cut_black(image):
    img = image.copy()
    st,ed = col_locate_black(image)
    # imshow([image])
    image = image[:,st:ed]
    replenish = img[:,ed:ed+20]

    return image,replenish

def col_locate_concave(canny):
    def g(i,k):


        if(k == 1):
            s = np.sum(canny[:, i:i + 10])
            ts = np.sum(np.ones(canny[:, i:i + 10].shape) * 255)
        elif(k == 2):
            s = np.sum(canny[:, i - 10:i])
            ts = np.sum(np.ones(canny[:,i-10:i].shape)*255)

        # print(s,ts,s/ts)
        if(s/ts>=0.125):
            return True
        else:
            return False


    # print(canny.shape)
    thc = np.sum(canny)/(2*canny.shape[1])
    p = sum_col(canny)
    st = 0
    ed = 0
    for i in range(5,len(p)):
        if(p[i]>thc and g(i,1)):
          st = i
          break
    for i in range(len(p)-1,5,-1):
        if (p[i] > thc and g(i,2)):
            ed = i
            break
    canny = canny[:,st-1:ed+1]
    return st,ed,canny

def col_cut_concave(image):


    img = image.copy()
    y,u,v = bgr2yuv(image)
    _255 = np.ones(u.shape)*np.average(y)
    _255 = _255.astype("uint8")
    yuv = cv.merge([y,u,v])
    bgr = cv.cvtColor(yuv,cv.COLOR_YUV2RGB)

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    canny = cv.Canny(gray, 50, 150)

    u = cv.Canny(u,35,105)
    v = cv.Canny(v,35,105)

    kernel = cv.getStructuringElement(cv.MORPH_CROSS,(2,2))
    u = cv.morphologyEx(u,cv.MORPH_CLOSE,kernel)
    v = cv.morphologyEx(v,cv.MORPH_CLOSE,kernel)

    # canny = canny - u - v
    canny = cv.morphologyEx(canny,cv.MORPH_CLOSE,kernel)

    st,ed,canny = col_locate_concave(canny)
    image = image[:,st-1:ed+1]
    replenish = img[:, ed+1:ed+20]

    return image,replenish

def digit_cut_concave(image):
    # imshow([image])
    backp = backgroundFilter(image)
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (2, 2))
    canny = cv.Canny(gray,50,150,)
    canny = cv.morphologyEx(canny, cv.MORPH_CLOSE, kernel)
    p = sum_col(canny)


    result = []
    for i in range(len(p)):
        if(p[i] > backp[i]):
            result.append(p[i] - backp[i])
        else:
            result.append(0)



    min_step = 20

    digits = []

    last = 0
    now = last + min_step
    while now < len(result):
        if(result[now] <= 510):
            # print(now - last)
            img = image[:,last:now]
            digits.append(img)
            # imshow([img])
            last = now
            now = now + min_step
        else:
            now = now + 1


    if (len(result) - last)/min_step > 0.5:
        img = image[:,last:len(result)]
        digits.append(img)


    return digits

def digit_cut_black(image):

    def digit_check(image):
        gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        th = cv.threshold(gray,np.average(gray)-10,255,cv.THRESH_BINARY)[1]

        x,y = gray.shape

        ed = th[:,-1]


        edp = np.sum(ed)
        tedp = 255*x


        hstep = int(x/3)
        ts = int(tedp/3)

        e1 = np.sum(ed[0:hstep])/ts
        e2 = np.sum(ed[hstep:hstep*2])/ts
        e3 = np.sum(ed[hstep*2:hstep*3])/ts


        if( e1 < 1 and e2 == 1 and e3 < 1):
            return False


        # imshow([th])

        if  edp/tedp >= 0.75:
            # print(e1,e2,e3)
            # imshow([image])
            return True
        else:
            return False



    min_step = int(image.shape[1]/20)
    last = 0
    now = last + min_step
    imgs = []

    while now < image.shape[1]:
        if digit_check(image[:,last:now]):
            imgs.append(image[:,last:now])
            last = now
            now = now + min_step
        else:
            now = now + 1

    if(now - last != min_step):
        imgs.append(image[:,last:image.shape[1]-1])

    return imgs

def img_divide(img,n):
    y = img.shape[1]
    y = int(y/n)
    imgs = []
    for i in range(n):
        imgg = img[:,i*y:(i+1)*y]
        imgs.append(imgg)

    return imgs


def digit_divide(image):
    img = image.copy()
    image = edge(image)
    st,ed,image = row_cut(image)
    img = img[st:ed]

    imgs = []


    if is_black(img) == 1:
        imgs,replenish = col_cut_black(img)
        img2 = imgs.copy()
        imgs = digit_cut_black(imgs)


    elif is_black(img) == 2:
        imgs,replenish = col_cut_concave(img)
        img2 = imgs.copy()
        imgs = digit_cut_concave(imgs)


    result_imgs = []

    for i in imgs:
        x,y,_ = i.shape
        num = (y/x)/0.7
        num = round(num)
        rs = img_divide(i,num)
        for r in rs:
            result_imgs.append(r)

    while len(result_imgs)%4 !=0:
        result_imgs.append(replenish.copy())


    return result_imgs,img2



if __name__ == '__main__':
    pass