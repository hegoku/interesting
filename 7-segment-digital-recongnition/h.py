#!/usr/bin/python
#coding=utf-8
import cv2
import numpy as np
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import sys

#灰度最大最小值的均值+最小值
def maxMinAvarge(img):
	max=np.max(img)
	min=np.min(img)
	return (max-min)/2+min

def zhifangtu(img,pos,title):
	bins = np.arange(257)

	item = img
	hist,bins = np.histogram(item,bins)
	width = 0.7*(bins[1]-bins[0])
	center = (bins[:-1]+bins[1:])/2
	plt.subplot(pos),plt.bar(center, hist, align = 'center', width = width),plt.title(title)
	return

def sortByX(s):
	return s['x']

SZ = 50 # size of each digit is SZ x SZ
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

def getContoursRectImage(img1,img2):
	result_img=[]
	image, contours, hierarchy = cv2.findContours(img2.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	#img=cv2.drawContours(img,contours,-1,(255,0,0),3)
	for ctr in contours:
		rect=cv2.boundingRect(ctr)
		roi = img1[rect[1]:rect[3]+rect[1],rect[0]:rect[2]+rect[0]]
		blank_image = np.zeros((50,50), np.uint8)
		x=int(50/2-rect[2]/2)
		y=int(50/2-rect[3]/2)
		blank_image[y:y+rect[3],x:x+rect[2]]=roi
		result_img.append({'x':rect[0],'img':blank_image})
	return sorted(result_img,key = sortByX)

samples = np.loadtxt('./generalsamples.data',np.float32)
responses = np.loadtxt('./generalresponses.data',np.float32)
responses = responses.reshape((responses.size,1))

model = cv2.ml.KNearest_create()
model.train(samples,cv2.ml.ROW_SAMPLE,responses)

cap = cv2.VideoCapture(0)
while(True):
    ret,img=cap.read()

    width=int(50*len(img[0])/len(img))
    img = cv2.resize(img, (width, 50), interpolation=cv2.INTER_AREA)
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #灰度
    img_smooth=cv2.GaussianBlur(img_gray, (3,11), 0) #高斯平滑,纵向,为了让0,7这种上下连通

    #如果灰度的均值大于 maxMinAvarge 则说明背景偏亮,则要让背景变为黑色
    if(np.mean(img_gray)>maxMinAvarge(img_gray)):
    	ret,img_binary=cv2.threshold(img_gray,maxMinAvarge(img_gray),255,cv2.THRESH_BINARY_INV)
    	ret,img_smooth=cv2.threshold(img_smooth,maxMinAvarge(img_gray),255,cv2.THRESH_BINARY_INV)
    else:
    	ret,img_binary=cv2.threshold(img_gray,maxMinAvarge(img_gray),255,cv2.THRESH_BINARY)
    	ret,img_smooth=cv2.threshold(img_smooth,maxMinAvarge(img_gray),255,cv2.THRESH_BINARY)

    cv2.imshow('frame',img_smooth)

    slice_num=getContoursRectImage(img_binary,img_smooth)

    res_num=[]

    for i in slice_num:
    	roi=i['img']
    	roismall = cv2.resize(roi,(50,50))
    	roismall = roismall.reshape((1,2500))
    	roismall = np.float32(roismall)
    	retval, results, neigh_resp, dists = model.findNearest(roismall, k = 1)
    	string = str(chr((results[0][0])))
    	res_num.append(string)
    print res_num

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
