import cv2 as cv
import numpy
import math
import operator

class Line:
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0
    def __init__(self,x1,y1,x2,y2):

        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2


class Point:
    x = 0
    y = 0
    def __init__(self,x,y):
        self.x = int(x)
        self.y = int(y)

    def selfcheck(self,mx,my):
        if(self.x <= 0 or self.y <= 0 or self.x >= mx or self.y >= my):
            return False
        else:
            return True

    def __repr__(self):
        return str(self.x)+" "+str(self.y)

class PointOfTwolines(Point):
    def __init__(self,x,y,k1,k2):
        super(PointOfTwolines,self).__init__(x,y)
        self.k1 = k1
        self.k2 = k2


def findk(l):
    """
    倾斜度k
    :param l:
    :return:
    """
    x1,y1,x2,y2 = [l.x1,l.y1,l.x2,l.y2]

    return (y2-y1)/(x2-x1+0.00001)

def length(l):
    # print(l)
    x1, y1, x2, y2 = [l.x1, l.y1, l.x2, l.y2]

    return math.sqrt(pow(x1-x2,2)+pow(y1-y2,2))


def findPoint(l1,l2):
    k1 = findk(l1)
    k2 = findk(l2)

    if necessary(k1,k2) == False:
        return None
    else:
        x = (k1*l1.x1 - k2*l2.x1 + l2.y1 - l1.y1) / (k1 - k2)
        y = k1*(x - l1.x1) + l1.y1
        p = PointOfTwolines(x,y,k1,k2)
        return p

def necessary(k1,k2):
    """
    是否有必要找共同点
    :param k1:
    :param k2:
    :return:
    """

    cosT = math.fabs(1+k1*k2)/(math.sqrt(1+pow(k1,2)) * math.sqrt(1+pow(k2,2)))
    # print(cosT)
    if(cosT < 0.15):
        return True
    else:
        return False

def quadrant(point):

    x = int(point.x/315)
    y = int(point.y/200)
    print(x,y)
    if x == 0:
        if y == 0:
            return 0
        elif y == 1:
            return 1
    elif x == 1:
        if y == 0:
            return 2
        elif y == 1:
            return 3


def Line2angleP(lines):
    lines2 = lines.copy()
    points = []

    for i in range(len(lines)):
        for j in range(i,len(lines)):
            l1 = lines[i]
            l2 = lines2[j]
            p = findPoint(Line(l1[0][0], l1[0][1], l1[0][2], l1[0][3]), Line(l2[0][0], l2[0][1], l2[0][2], l2[0][3]))
            if (p == None or p.selfcheck(630, 400) == False):
                continue
            else:
                # points.append(quadrant(p))
                points.append(p)


    return points


if __name__ == '__main__':

    print(necessary(-1,11111111111))