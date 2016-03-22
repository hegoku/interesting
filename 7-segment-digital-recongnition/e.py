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

number_mod=[
	cv2.threshold(cv2.cvtColor(cv2.imread('./number/0.jpg'),cv2.COLOR_BGR2GRAY),128,255,cv2.THRESH_BINARY)[1],
	cv2.threshold(cv2.cvtColor(cv2.imread('./number/1.jpg'),cv2.COLOR_BGR2GRAY),128,255,cv2.THRESH_BINARY)[1],
	cv2.threshold(cv2.cvtColor(cv2.imread('./number/2.jpg'),cv2.COLOR_BGR2GRAY),128,255,cv2.THRESH_BINARY)[1],
	cv2.threshold(cv2.cvtColor(cv2.imread('./number/3.jpg'),cv2.COLOR_BGR2GRAY),128,255,cv2.THRESH_BINARY)[1],
	cv2.threshold(cv2.cvtColor(cv2.imread('./number/4.jpg'),cv2.COLOR_BGR2GRAY),128,255,cv2.THRESH_BINARY)[1],
	cv2.threshold(cv2.cvtColor(cv2.imread('./number/5.jpg'),cv2.COLOR_BGR2GRAY),128,255,cv2.THRESH_BINARY)[1],
	cv2.threshold(cv2.cvtColor(cv2.imread('./number/6.jpg'),cv2.COLOR_BGR2GRAY),128,255,cv2.THRESH_BINARY)[1],
	cv2.threshold(cv2.cvtColor(cv2.imread('./number/7.jpg'),cv2.COLOR_BGR2GRAY),128,255,cv2.THRESH_BINARY)[1],
	cv2.threshold(cv2.cvtColor(cv2.imread('./number/8.jpg'),cv2.COLOR_BGR2GRAY),128,255,cv2.THRESH_BINARY)[1],
	cv2.threshold(cv2.cvtColor(cv2.imread('./number/9.jpg'),cv2.COLOR_BGR2GRAY),128,255,cv2.THRESH_BINARY)[1],
	cv2.threshold(cv2.cvtColor(cv2.imread('./number/dot.jpg'),cv2.COLOR_BGR2GRAY),128,255,cv2.THRESH_BINARY)[1],
]

#img=cv2.imread('./number/n.bmp')
#img=cv2.imread('./images/QQ20160317-1.png')
#img=cv2.imread('./images/QQ20160318-3.png')
#img=cv2.imread('./images/QQ20160322-1.png')
#img=cv2.imread('./images/QQ20160317-4.png') #车牌
#img=cv2.imread('./imagesp/hoto_2.jpg')
#img=cv2.imread('./images/photo_1.jpg')
img=cv2.imread(sys.argv[1])
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

slice_num=getContoursRectImage(img_binary,img_smooth)

res_num=[]

for i in slice_num:
	currentMin= 1.0
	currentIndex=0
	threshold=0.8
	print "---------------------------"
	for j in range(len(number_mod)):
		#tmp=cv2.GaussianBlur(number_mod[j], (1,11), 0)
		tmp=cv2.threshold(cv2.GaussianBlur(number_mod[j], (5,11), 0),128,255,cv2.THRESH_BINARY)[1]
		tmp=number_mod[j]
		w, h = tmp.shape[::-1]
		res=cv2.matchTemplate(i['img'],tmp,cv2.TM_SQDIFF_NORMED)
		min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
		print min_val
		print " "
		if(min_val<currentMin):
			currentMin=min_val
			currentIndex=j
	num=str(currentIndex)
	if(currentIndex==10):num="."
	#res_num.append(num)
	if(currentMin<=threshold):
		res_num.append(num)
print res_num

plt.figure(1)
plt.subplot(421),plt.imshow(img_gray,'gray'),plt.title('gray')
zhifangtu(img_gray,422,str(np.mean(img_gray)))

plt.subplot(423),plt.imshow(img_smooth,'gray'),plt.title('img_smooth')
zhifangtu(img,424,str(maxMinAvarge(img_smooth)))

plt.subplot(425),plt.imshow(img,'gray'),plt.title('final')

plt.figure(2)
j=1
for i in slice_num:
	plt.subplot(10,3,j),plt.imshow(i['img'],'gray')
	j=j+1

plt.figure(3)
j=1
for i in number_mod:
	tmp=cv2.threshold(cv2.GaussianBlur(i, (5,11), 0),128,255,cv2.THRESH_BINARY)[1]
	tmp=i
	plt.subplot(10,3,j),plt.imshow(tmp,'gray')
	j=j+1

plt.show()
