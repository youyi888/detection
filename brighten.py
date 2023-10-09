import cv2
import numpy as np
def gamma_trans(img,gamma):#gamma大于1时图片变暗，小于1图片变亮
	#具体做法先归一化到1，然后gamma作为指数值求出新的像素值再还原
	gamma_table = [np.power(x/255.0,gamma)*255.0 for x in range(256)]
	gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
	#实现映射用的是Opencv的查表函数
	return cv2.LUT(img,gamma_table)
img = cv2.imread('E:/mark10.7/1-1/tile/focus0/0/039x050.jpg')
img_corrected = gamma_trans(img, 0.5)
cv2.imshow('a1',img_corrected)
# cv2.imshow('as1',img)
# img_corrected = gamma_trans(img, 2)
# cv2.imshow('a1g',img_corrected)
cv2.waitKey(0)
