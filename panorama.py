import cv2
import numpy as np
import copy
import time

RESIZE_W = 675
RESIZE_H = 900


class Image:
    def __init__(self, name, img):
        def calculate(self):
            detector = cv2.AKAZE_create()
            keypoints, descriptors = detector.detectAndCompute(self.image, None)
            return keypoints, descriptors

        self.name = name
        self.image = img
        self.kp, self.des = calculate(self)

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
    img = cv2.resize(img, (RESIZE_W, RESIZE_H))  # width, height
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
    x.append(((H[0][0] * 0 + H[0][1] * 0 + H[0][2]) / (H[2][0] * 0 + H[2][1] * 0 + H[2][2])))
    y.append(((H[1][0] * 0 + H[1][1] * 0 + H[1][2]) / (H[2][0] * 0 + H[2][1] * 0 + H[2][2])))
    x.append(((H[0][0] * 0 + H[0][1] * size[0] + H[0][2]) / (H[2][0] * 0 + H[2][1] * size[0] + H[2][2])))
    y.append(((H[1][0] * 0 + H[1][1] * size[0] + H[1][2]) / (H[2][0] * 0 + H[2][1] * size[0] + H[2][2])))
    x.append(((H[0][0] * size[1] + H[0][1] * 0 + H[0][2]) / (H[2][0] * size[1] + H[2][1] * 0 + H[2][2])))
    y.append(((H[1][0] * size[1] + H[1][1] * 0 + H[1][2]) / (H[2][0] * size[1] + H[2][1] * 0 + H[2][2])))
    x.append(((H[0][0] * size[1] + H[0][1] * size[0] + H[0][2]) / (H[2][0] * size[1] + H[2][1] * size[0] + H[2][2])))
    y.append(((H[1][0] * size[1] + H[1][1] * size[0] + H[1][2]) / (H[2][0] * size[1] + H[2][1] * size[0] + H[2][2])))

    min_x = min(x)
    min_y = min(y)
    max_x = max(x)
    max_y = max(y)
    div = ((min_x, max_x), (min_y, max_y))
    return div


def write_blending(target, source, SrcMask):
    mask = cv2.cvtColor(SrcMask, cv2.COLOR_GRAY2RGB)
    target[(mask != [0, 0, 0])] = source[(mask != [0, 0, 0])]
    return target


def make_mask(target, src):
    CommonMaskRGB = np.zeros((target.shape[0], target.shape[1]), dtype=np.uint8)
    SrcMaskRGB = np.zeros((target.shape[0], target.shape[1]), dtype=np.uint8)
    TargetMaskRGB = np.zeros((target.shape[0], target.shape[1]), dtype=np.uint8)
    CommonMaskRGB[(cv2.cvtColor(target, cv2.COLOR_RGB2GRAY) != 0) * (cv2.cvtColor(src, cv2.COLOR_RGB2GRAY) != 0)] = 255
    SrcMaskRGB[(cv2.cvtColor(src, cv2.COLOR_RGB2GRAY) != 0) * (CommonMaskRGB == 0)] = 255
    TargetMaskRGB[(cv2.cvtColor(target, cv2.COLOR_RGB2GRAY) != 0)] = 255
    CommonMask = cv2.erode(CommonMaskRGB, np.ones((5, 5), np.uint8), iterations=3)
    SrcMask = cv2.dilate(SrcMaskRGB, np.ones((5, 5), np.uint8), iterations=1)
    TargetMask = cv2.dilate(TargetMaskRGB, np.ones((3, 3), np.uint8), iterations=1)
    return CommonMask, SrcMask, TargetMask


def arrange_rgb(mat, TargetMask):
    mat[TargetMask == 0] = [0, 0, 0]
    gray = cv2.cvtColor(mat, cv2.COLOR_RGB2GRAY)
    mat[(TargetMask != 0) * (gray == 0)] = 1
    return mat


def get_center(mask):
    min_x = 10000
    max_x = -1
    min_y = 10000
    max_y = -1
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if (mask[y][x]):
                if (x < min_x):
                    min_x = x
                if (y < min_y):
                    min_y = y
                if (x > max_x):
                    max_x = x
                if (y > max_y):
                    max_y = y
    return (max_y + min_y) / 2, (max_x + min_x) / 2


def get_distance(querykeysi, trainkeysi, cvimg_w1):  # 输入点坐标，输出距离。querykeys[pt_idx],trainkeys[pt_idx]输入
    v = list(map(lambda x: x[0] - x[1], zip(trainkeysi, querykeysi)))
    v[0] = v[0] + cvimg_w1
    return np.sqrt(v[0] ** 2 + v[1] ** 2)


def in_interval(o, w0, w1, h0, h1):
    '''
    o is a tuple like (79,153). w0, w1, h0, h1 defines a square.
    if o in the square, return True
    '''
    return (o[0] >= w0) and (o[0] <= w1) and (o[1] >= h0) and (o[1] <= h1)


def grid_pick(querykeys, trainkeys):
    '''
    input:feature points(querykeys, trainkeys)
    output:selected points(querykeys, trainkeys)
    原理；对参与拼接的第二张图（因为第一张图有可能是多图拼的，图像区域不规则）
    划分grid_szw*grid_szh网格，在不同格子里提取(trainkeys)特征点，然后对应选出
    第一张图的相应点，最后返回筛选后的querykeys, trainkeys
    '''
    grid_szh = 5
    grid_szw = 5
    inter_h = [(RESIZE_H // grid_szh) * i for i in range(grid_szh)]
    inter_h.append(RESIZE_H)
    inter_w = [(RESIZE_W // grid_szw) * i for i in range(grid_szw)]
    inter_w.append(RESIZE_W)
    res_train = list()
    res_query = list()
    tmplist = np.zeros([grid_szh, grid_szw, 2])  # 记录像素点的位置flag
    for i in range(grid_szw):
        for j in range(grid_szh):
            for o in trainkeys:  # 下面的判断保证每个方格只取一个点
                if ((in_interval(o, inter_w[i], inter_w[i + 1], inter_h[j], inter_h[j + 1])) and (
                        np.array(tmplist[j][i]).all() == np.array([0, 0]).all())):
                    res_train.append(o)
                    tmplist[j][i] = o
                    res_query.append(querykeys[trainkeys.index(o)])  # 如果trainkeys的某格子里有一点，把对应位置querykeys也加上
    return (res_query, res_train)


def fgrid_pick(querykeys, trainkeys):
    '''
        input:feature points(querykeys, trainkeys)
        output:selected points(querykeys, trainkeys)
        原理；对参与拼接的第二张图（因为第一张图有可能是多图拼的，图像区域不规则）的特征点 的分布区域
        划分grid_szw*grid_szh网格，在不同格子里提取(trainkeys)特征点，然后对应选出
        第一张图的相应点，最后返回筛选后的querykeys, trainkeys
        '''
    grid_szh = 5
    grid_szw = 5
    wmin = np.array(trainkeys)[:, 0].min()
    wmax = np.array(trainkeys)[:, 0].max()
    hmin = np.array(trainkeys)[:, 1].min()
    hmax = np.array(trainkeys)[:, 1].max()
    inter_h = [(((hmax - hmin) // grid_szh) * i + hmin) for i in range(grid_szh)]
    inter_h.append(hmax)
    inter_w = [(((wmax - wmin) // grid_szw) * i + wmin) for i in range(grid_szw)]
    inter_w.append(wmax)
    res_train = list()
    res_query = list()
    tmplist = np.zeros([grid_szh, grid_szw, 2])  # 记录像素点的位置flag
    for i in range(grid_szw):
        for j in range(grid_szh):
            for o in trainkeys:  # 下面的判断保证每个方格只取一个点
                if ((in_interval(o, inter_w[i], inter_w[i + 1], inter_h[j], inter_h[j + 1])) and (
                        np.array(tmplist[j][i]).all() == np.array([0, 0]).all())):
                    res_train.append(o)
                    tmplist[j][i] = o
                    res_query.append(querykeys[trainkeys.index(o)])  # 如果trainkeys的某格子里有一点，把对应位置querykeys也加上
    return (res_query, res_train)


def make_panorama(original1, original2, count):
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, False)
    matches = matcher.knnMatch(original1.des, original2.des, 2)
    goodmatches = []
    trainkeys = []
    querykeys = []
    maskArray = []

    for i in matches: #这里可以对matches特征数量加以限制
        if i[0].distance / i[1].distance < 0.65:
            goodmatches.append(i[0])
            querykeys.append((original1.kp[i[0].queryIdx].pt[0], original1.kp[i[0].queryIdx].pt[1]))
            trainkeys.append((original2.kp[i[0].trainIdx].pt[0], original2.kp[i[0].trainIdx].pt[1]))

    ############以下添加最小距离匹配
    dslst = []
    original1_cv_img = original1
    original2_cv_img = original2
    cvimg_h1, cvimg_w1 = original1_cv_img.image.shape[0:2]
    cvimg_h2, cvimg_w2 = original2_cv_img.image.shape[0:2]
    ########取消距离匹配时保留上面这几段参数，注释下面的部分
    for i in range(len(querykeys)):
        dslst.append(get_distance(querykeys[i], trainkeys[i], cvimg_w1))

    Z1 = zip(dslst, querykeys)
    Z2 = zip(dslst, trainkeys)

    rst_lst1 = sorted(Z1)
    rst_lst2 = sorted(Z2)
    querykeys = []
    trainkeys = []
    querykeys = [i[1] for i in rst_lst1[:100]]  # [(430,775),(719,781),(922,685),(723,982),(507,958),(751,1323),(1036,1451),(709,1487),(599,659)]#这里修改为人工选点
    trainkeys = [i[1] for i in rst_lst2[:100]]  # [(79,153),(399,155),(583,74),(398,360),(175,358),(384,544),(560,571),(343,698),(273,19)]#
    #############
    #下面一段用于相对整个图片做网格筛选特征点
    querykeys, trainkeys = grid_pick(querykeys, trainkeys)

    ##############
    # 下面一段用于相对所有特征点分布的框 做网格筛选特征点
    # querykeys, trainkeys = fgrid_pick(querykeys, trainkeys)

    ##############
    # 下面这一段用于为绿线图提供参数，这段需要在H矩阵使用之前，以免图像扭曲，参数改变
    print(original1_cv_img.image.shape)
    cvimg_h1, cvimg_w1 = original1_cv_img.image.shape[0:2]
    cvimg_h2, cvimg_w2 = original2_cv_img.image.shape[0:2]
    blank_img = np.zeros((max(cvimg_h1, cvimg_h2), (cvimg_w1 + cvimg_w2), 3), np.uint8)

    blank_img[0:cvimg_h1, 0:cvimg_w1] = original1_cv_img.image
    blank_img[0:cvimg_h2, cvimg_w1:cvimg_w1 + cvimg_w2] = original2_cv_img.image
    ####################
    H, status = cv2.findHomography(np.array(trainkeys), np.array(querykeys), cv2.RANSAC, 5.0)
    div = calc_dst4points(H, original2.image.shape)
    d = original1.resize_mat(div)
    T_xy = [[1.0, 0.0, -d[0]], [0.0, 1.0, -d[1]], [0.0, 0.0, 1.0]]
    panorama = cv2.warpPerspective(original2.image, np.dot(T_xy, H),
                                   (original1.image.shape[1], original1.image.shape[0]))
    CommonMask, SrcMask, TargetMask = make_mask(panorama, original1.image)

    label = cv2.connectedComponentsWithStats(CommonMask)
    center = np.delete(label[3], 0, 0)
    test = get_center(CommonMask)
    blending = cv2.seamlessClone(original1.image, panorama, cv2.cvtColor(CommonMask, cv2.COLOR_GRAY2BGR),
                                 (int(test[1]), int(test[0])), cv2.NORMAL_CLONE)
    blending = arrange_rgb(blending, TargetMask)
    blending = write_blending(blending, original1.image, SrcMask)

    # Convert PIL to opencv
    # original1_cv_img = cv2.cvtColor(np.array(original1), cv2.COLOR_RGB2BGR)
    # original2_cv_img = cv2.cvtColor(np.array(original2), cv2.COLOR_RGB2BGR)
    #

    print(cvimg_h1, cvimg_w1)
    print(cvimg_h2, cvimg_w2)

    # print(original1_cv_img.image.shape)

    # loop over the matches

    # (hA, wA) = 1200,900#
    # (hB, wB) = 1200,900#
    # vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    # vis[0:hA, 0:wA] = cv2.imread("0.jpg")
    # vis[0:hB, wA:] = cv2.imread("04small.jpg")

    # loop over the matches

    green_name = "green" + str(count) + ".jpg"
    for pt_idx, _ in enumerate(querykeys[:200]):
        # print(pt)

        cv2.line(blank_img, (int(querykeys[pt_idx][0]), int(querykeys[pt_idx][1])),
                 (cvimg_w1 + int(trainkeys[pt_idx][0]), int(trainkeys[pt_idx][1])), (0, 255, 0), 1)
    cv2.imwrite(green_name, blank_img)

    return blending
