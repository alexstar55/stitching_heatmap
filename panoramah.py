
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import copy
import time
import pickle
class Image:
    def __init__(self,name,img):
        def calculate(self):
            detector = cv2.AKAZE_create()
            keypoints, descriptors = detector.detectAndCompute(self.image,None)
            return keypoints, descriptors

        self.name = name
        self.image = img
        self.kp,self.des = calculate(self)

    def show(self):
        cv2.imshow(self.name, self.image)
        cv2.waitKey(0)

    def resize_mat(self, div):
        height, width = self.image.shape[0:2]
        d = [0, 0, width, height]
        if div[0][0] < 0:
            d[0] = div[0][0]
        if div[0][1] > width:
            d[2] = div[0][1]
        if div[1][0] < 0:
            d[1] = div[1][0]
        if div[1][1] > height:
            d[3] = div[1][1]
        T = np.array([[1.0, 0.0, -d[0]], [0.0, 1.0, -d[1]], [0.0, 0.0, 1.0]])
        self.image = cv2.warpPerspective(self.image, T, (int(-d[0] + d[2]), int(-d[1] + d[3])))
        return d

def resize_image(img):
    img = cv2.resize(img,(int(750),int(1000)))  # width, height
    for i in img:
        for j in i:
            if not j.all():
                j[0] += 1
                j[1] += 1
                j[2] += 1
    return img

def calc_dst4points(H, size):
    x = []
    y = []
    x.append(((H[0][0]*0 + H[0][1]*0 + H[0][2])/(H[2][0]*0 + H[2][1]*0 + H[2][2])))
    y.append(((H[1][0]*0 + H[1][1]*0 + H[1][2])/(H[2][0]*0 + H[2][1]*0 + H[2][2])))
    x.append(((H[0][0]*0 + H[0][1]*size[0] + H[0][2])/(H[2][0]*0 + H[2][1]*size[0] + H[2][2])))
    y.append(((H[1][0]*0 + H[1][1]*size[0] + H[1][2])/(H[2][0]*0 + H[2][1]*size[0] + H[2][2])))
    x.append(((H[0][0]*size[1] + H[0][1]*0 + H[0][2])/(H[2][0]*size[1] + H[2][1]*0 + H[2][2])))
    y.append(((H[1][0]*size[1] + H[1][1]*0 + H[1][2])/(H[2][0]*size[1] + H[2][1]*0 + H[2][2])))
    x.append(((H[0][0]*size[1] + H[0][1]*size[0] + H[0][2])/(H[2][0]*size[1] + H[2][1]*size[0] + H[2][2])))
    y.append(((H[1][0]*size[1] + H[1][1]*size[0] + H[1][2])/(H[2][0]*size[1] + H[2][1]*size[0] + H[2][2])))

    min_x = min(x)
    min_y = min(y)
    max_x = max(x)
    max_y = max(y)
    div = ((min_x, max_x),(min_y, max_y))
    return div

def write_blending(target, source, SrcMask):
    mask = cv2.cvtColor(SrcMask,cv2.COLOR_GRAY2RGB)
    target[(mask != [0,0,0])] = source[(mask != [0,0,0])]
    return target


def make_mask(target, src):
    CommonMaskRGB = np.zeros((target.shape[0], target.shape[1]), dtype=np.uint8)
    SrcMaskRGB = np.zeros((target.shape[0], target.shape[1]), dtype=np.uint8)
    TargetMaskRGB = np.zeros((target.shape[0], target.shape[1]), dtype=np.uint8)
    CommonMaskRGB[(cv2.cvtColor(target,cv2.COLOR_RGB2GRAY) != 0) * (cv2.cvtColor(src,cv2.COLOR_RGB2GRAY) != 0)] = 255
    SrcMaskRGB[(cv2.cvtColor(src,cv2.COLOR_RGB2GRAY) != 0) * (CommonMaskRGB == 0)] = 255
    TargetMaskRGB[(cv2.cvtColor(target,cv2.COLOR_RGB2GRAY) != 0)] = 255
    CommonMask = cv2.erode(CommonMaskRGB,np.ones((5,5),np.uint8),iterations = 3)
    SrcMask = cv2.dilate(SrcMaskRGB,np.ones((5,5),np.uint8),iterations = 1)
    TargetMask = cv2.dilate(TargetMaskRGB,np.ones((3,3),np.uint8),iterations = 1)
    return CommonMask, SrcMask, TargetMask

def arrange_rgb(mat, TargetMask):
    mat[TargetMask==0] = [0,0,0]
    gray = cv2.cvtColor(mat,cv2.COLOR_RGB2GRAY)
    mat[(TargetMask != 0) * (gray == 0)] = 1
    return mat

def get_center(mask):
    min_x = 10000
    max_x = -1
    min_y = 10000
    max_y = -1
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if(mask[y][x]):
                if(x<min_x):
                    min_x = x
                if(y<min_y):
                    min_y = y
                if(x>max_x):
                    max_x = x
                if(y>max_y):
                    max_y = y
    return (max_y+min_y)/2, (max_x+min_x)/2

def get_distance(querykeysi,trainkeysi,cvimg_w1):  #输入点坐标，输出距离。querykeys[pt_idx],trainkeys[pt_idx]输入
    v = list(map(lambda x: x[0]-x[1], zip(trainkeysi, querykeysi)))
    v[0]=v[0]+cvimg_w1
    return np.sqrt(v[0]**2+v[1]**2)

def make_panorama(original1,original2,count):
#     matcher = cv2.BFMatcher(cv2.NORM_HAMMING, False)
#     matches = matcher.knnMatch(original1.des,original2.des,2)
#     goodmatches = []
#     trainkeys = []
#     querykeys = []
#     maskArray = []

#     for i in matches:
#         if i[0].distance/i[1].distance < 0.7:
#             goodmatches.append(i[0])
#             querykeys.append((original1.kp[i[0].queryIdx].pt[0],original1.kp[i[0].queryIdx].pt[1]))
#             trainkeys.append((original2.kp[i[0].trainIdx].pt[0],original2.kp[i[0].trainIdx].pt[1]))
    
    ############以下添加最小距离匹配
#     dslst=[]
#     original1_cv_img = original1
#     original2_cv_img = original2
#     cvimg_h1, cvimg_w1 = original1_cv_img.image.shape[0:2]
#     cvimg_h2, cvimg_w2 = original2_cv_img.image.shape[0:2]
#     for i in range(len(querykeys)):
#         dslst.append(get_distance(querykeys[i],trainkeys[i],cvimg_w1))
       
#     Z1=zip(dslst,querykeys)
#     Z2=zip(dslst,trainkeys)
    
#     rst_lst1=sorted(Z1)
#     rst_lst2=sorted(Z2)
#     querykeys=[]
#     trainkeys=[]
#     querykeys=[i[1] for i in rst_lst1[:100]]
#     trainkeys=[i[1] for i in rst_lst2[:100]]
    #############    
#     H, status = cv2.findHomography(np.array(trainkeys),np.array(querykeys),cv2.RANSAC, 5.0)
    dumpname='dump'+str(count)+'.txt' #保存的H矩阵
    with open(dumpname,'rb') as f:
        H=pickle.load(f)
    
    
    div = calc_dst4points(H, original2.image.shape)
    d = original1.resize_mat(div)
    T_xy = [[1.0, 0.0, -d[0]],[0.0, 1.0, -d[1]],[0.0, 0.0, 1.0]]
    panorama = cv2.warpPerspective(original2.image,np.dot(T_xy,H),(original1.image.shape[1],original1.image.shape[0]))
    CommonMask, SrcMask, TargetMask = make_mask(panorama, original1.image)
    label = cv2.connectedComponentsWithStats(CommonMask)
    center = np.delete(label[3], 0, 0)
    test = get_center(CommonMask)
    blending = cv2.seamlessClone(original1.image, panorama, cv2.cvtColor,(CommonMask,cv2.COLOR_GRAY2BGR), (int(test[1]),int(test[0])), cv2.NORMAL_CLONE)
    blending = arrange_rgb(blending, TargetMask)
    blending = write_blending(blending, original1.image, SrcMask)


    # Convert PIL to opencv 
    # original1_cv_img = cv2.cvtColor(np.array(original1), cv2.COLOR_RGB2BGR)
    # original2_cv_img = cv2.cvtColor(np.array(original2), cv2.COLOR_RGB2BGR)
    
    original1_cv_img = original1
    original2_cv_img = original2
    cvimg_h1, cvimg_w1 = original1_cv_img.image.shape[0:2]
    cvimg_h2, cvimg_w2 = original2_cv_img.image.shape[0:2]
    
    print(cvimg_h1, cvimg_w1)
    print(cvimg_h2, cvimg_w2)

    # blank_img = np.zeros((max(cvimg_h1, cvimg_h2) ,(cvimg_w1+cvimg_w2), 3), np.uint8)
    #
    # blank_img[0:cvimg_h1,0:cvimg_w1 ] = original1_cv_img.image
    # blank_img[ 0:cvimg_h2,cvimg_w1:cvimg_w1 + cvimg_w2] = original2_cv_img.image
    #
    #
    # for pt_idx, _ in enumerate(querykeys[:25]):
    #     #print(pt)
    #
    #     cv2.line(blank_img,(int(querykeys[pt_idx][0]), int(querykeys[pt_idx][1])),(cvimg_w1 + int(trainkeys[pt_idx][0]), int(trainkeys[pt_idx][1])), (0,255,0),1)
    # cv2.imwrite('1111.jpg', blank_img)

    return blending

