# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os.path
import copy


# 高斯噪声
def addGaussianNoise(image, percetage):
    G_Noiseimg = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    G_NoiseNum = int(percetage * image.shape[0] * image.shape[1])
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(0, h)
        temp_y = np.random.randint(0, w)
        G_Noiseimg[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0]
    return G_Noiseimg

# 旋转
def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    # If no rotation center is specified, the center of the image is set as the rotation center
    if center is None:
        center = (w / 2, h / 2)
    m = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, m, (w, h))
    return rotated


# 翻转
def flip(image):
    flipped_image = np.fliplr(image)
    return flipped_image


# 图片文件夹路径
file_dir = r'D:/deeplearning/sheep_status_dataset_forlabel/20240420-multi/socializing_enhance/'
for img_name in os.listdir(file_dir):
    img_path = file_dir + img_name
    img = cv2.imread(img_path)
    # cv2.imshow("1",img)
    # cv2.waitKey(5000)
    # 旋转
    # rotated_90 = rotate(img, 90)
    # cv2.imwrite(file_dir + img_name[0:-4] + '_r90.jpg', rotated_90)
    rotated_180 = rotate(img, 180)
    cv2.imwrite(file_dir + img_name[0:-4] + '_r180.jpg', rotated_180)

for img_name in os.listdir(file_dir):
    img_path = file_dir + img_name
    img = cv2.imread(img_path)
    # 镜像
    flipped_img = flip(img)
    cv2.imwrite(file_dir + img_name[0:-4] + '_fli.jpg', flipped_img)

    # 增加噪声
    # img_salt = SaltAndPepper(img, 0.3)
    # cv2.imwrite(file_dir + img_name[0:7] + '_salt.jpg', img_salt)
    img_gauss = addGaussianNoise(img, 0.3)
    cv2.imwrite(file_dir + img_name[0:-4] + '_noise.jpg', img_gauss)

    # # 变亮、变暗
    # img_darker = darker(img)
    # cv2.imwrite(file_dir + img_name[0:-4] + '_darker.jpg', img_darker)
    # img_brighter = brighter(img)
    # cv2.imwrite(file_dir + img_name[0:-4] + '_brighter.jpg', img_brighter)
    #
    # blur = cv2.GaussianBlur(img, (7, 7), 1.5)
    # #      cv2.GaussianBlur(图像，卷积核，标准差）
    # cv2.imwrite(file_dir + img_name[0:-4] + '_blur.jpg', blur)
