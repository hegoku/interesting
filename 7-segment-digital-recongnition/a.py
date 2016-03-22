#coding=utf-8
import cv2
import numpy as np
import matplotlib.image as mpimg
from matplotlib import pyplot as plt

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

def getContoursRectImage(img):
	result_img=[]
	result_hu=[]
	image, contours, hierarchy = cv2.findContours(img.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	img=cv2.drawContours(img,contours,-1,(255,0,0),3)
	for ctr in contours:
		rect=cv2.boundingRect(ctr)
		roi = img[rect[1]:rect[3]+rect[1],rect[0]:rect[2]+rect[0]]
	    width=int(32*rect[2]/rect[3])
	    roi = cv2.resize(roi, (width, 32), interpolation=cv2.INTER_AREA)
	    m=cv2.moments(roi)
	    hm=cv2.HuMoments(m);
	    result_hu.append(ctr);
	    result_img.append(roi)
	#rects = [cv2.boundingRect(ctr) for ctr in contours]

	#for rect in rects:
	    # Draw the rectangles
	#    roi = img[rect[1]:rect[3]+rect[1],rect[0]:rect[2]+rect[0]]
	#    width=int(32*rect[2]/rect[3])
#	    roi = cv2.resize(roi, (width, 32), interpolation=cv2.INTER_AREA)
	#    m=cv2.moments(roi)
	#    hm=cv2.HuMoments(m);
	#    result_hu.append(hm);
	#    result_img.append(roi)
	return [result_img,result_hu]

#number_mod=[
#	cv2.threshold(cv2.cvtColor(cv2.imread('/Users/jack/Pictures/number/0.jpg'),cv2.COLOR_BGR2GRAY),128,255,cv2.THRESH_BINARY)[1],
#	cv2.threshold(cv2.cvtColor(cv2.imread('/Users/jack/Pictures/number/1.jpg'),cv2.COLOR_BGR2GRAY),128,255,cv2.THRESH_BINARY)[1],
#	cv2.threshold(cv2.cvtColor(cv2.imread('/Users/jack/Pictures/number/2.jpg'),cv2.COLOR_BGR2GRAY),128,255,cv2.THRESH_BINARY)[1],
#	cv2.threshold(cv2.cvtColor(cv2.imread('/Users/jack/Pictures/number/3.jpg'),cv2.COLOR_BGR2GRAY),128,255,cv2.THRESH_BINARY)[1],
#	cv2.threshold(cv2.cvtColor(cv2.imread('/Users/jack/Pictures/number/4.jpg'),cv2.COLOR_BGR2GRAY),128,255,cv2.THRESH_BINARY)[1],
#	cv2.threshold(cv2.cvtColor(cv2.imread('/Users/jack/Pictures/number/5.jpg'),cv2.COLOR_BGR2GRAY),128,255,cv2.THRESH_BINARY)[1],
#	cv2.threshold(cv2.cvtColor(cv2.imread('/Users/jack/Pictures/number/6.jpg'),cv2.COLOR_BGR2GRAY),128,255,cv2.THRESH_BINARY)[1],
#	cv2.threshold(cv2.cvtColor(cv2.imread('/Users/jack/Pictures/number/7.jpg'),cv2.COLOR_BGR2GRAY),128,255,cv2.THRESH_BINARY)[1],
#	cv2.threshold(cv2.cvtColor(cv2.imread('/Users/jack/Pictures/number/8.jpg'),cv2.COLOR_BGR2GRAY),128,255,cv2.THRESH_BINARY)[1],
#	cv2.threshold(cv2.cvtColor(cv2.imread('/Users/jack/Pictures/number/9.jpg'),cv2.COLOR_BGR2GRAY),128,255,cv2.THRESH_BINARY)[1],
#]

number_mod=cv2.threshold(cv2.cvtColor(cv2.imread('/Users/jack/Pictures/number/n.bmp'),cv2.COLOR_BGR2GRAY),128,255,cv2.THRESH_BINARY)[1]
#number_mod=cv2.imread('./number/n.bmp')

#img=cv2.imread('./number/n.bmp')
#img=cv2.imread('./images/QQ20160317-1.png')
img=cv2.imread('./images/QQ20160318-2.png')
#img=cv2.imread('./images/QQ20160317-4.png') #车牌
#img=cv2.imread('./imagesp/hoto_2.jpg')
#img=cv2.imread('./images/photo_1.jpg')
width=int(32*len(img[0])/len(img))
img = cv2.resize(img, (width, 32), interpolation=cv2.INTER_AREA)
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #灰度
img_smooth=cv2.GaussianBlur(img_gray, (1,7), 0) #高斯平滑,纵向,为了让0,7这种上下连通
img_equ=img_smooth
#if( (np.max(img_smooth)-np.min(img_smooth))/255 <=0.3 ):
#img_equ=cv2.equalizeHist(img_smooth)
#如果灰度的均值大于 maxMinAvarge 则说明背景偏亮,则要让背景变为黑色
if(np.mean(img_smooth)>maxMinAvarge(img_smooth)):
	ret,img_binary=cv2.threshold(img_equ,maxMinAvarge(img_smooth),255,cv2.THRESH_BINARY_INV)
else:
	ret,img_binary=cv2.threshold(img_equ,maxMinAvarge(img_smooth),255,cv2.THRESH_BINARY)
#img_binary=cv2.adaptiveThreshold(img_binary,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
#kernel = np.ones((3,1),np.uint8)
#img_binary=cv2.erode(img_binary,kernel,iterations = 3)
#kernel = np.ones((1,3),np.uint8)
#img_binary=cv2.dilate(img_binary,kernel,iterations = 2)

image, contours, hierarchy = cv2.findContours(img_binary.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in contours]

# For each rectangular region, calculate HOG features and predict
# the digit using Linear SVM.

number_image=[]

for rect in rects:
    # Draw the rectangles
    cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 1)
    roi = img_binary[rect[1]:rect[3]+rect[1],rect[0]:rect[2]+rect[0]]
    width=int(32*rect[2]/rect[3])
    roi = cv2.resize(roi, (width, 32), interpolation=cv2.INTER_AREA)
    number_image.append(roi)

plt.subplot(421),plt.imshow(img_gray,'gray'),plt.title('gray')
zhifangtu(img_gray,422,str(np.mean(img_gray)))

plt.subplot(423),plt.imshow(img_binary,'gray'),plt.title('img_binary')
zhifangtu(img_binary,424,str(maxMinAvarge(img_equ)))

plt.subplot(425),plt.imshow(img,'gray'),plt.title('final')

plt.figure(2)
j=1
for i in number_image:
	plt.subplot(10,3,j),plt.imshow(i,'gray')
	j=j+1


plt.figure(3)
j=1
tmp=getContoursRectImage(number_mod)
plt.subplot(3,5,12),plt.imshow(number_image[3],'gray')
m=cv2.moments(number_image[3])
hm=cv2.HuMoments(m)
#image,contours,hierarchy = cv2.findContours(number_image[2].copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#hm=contours[0]
epsilon = 0.1*cv2.arcLength(contours[0],True)
hm = cv2.approxPolyDP(contours[0],epsilon,True)
for i in tmp[0]:
	plt.subplot(3,5,j),plt.imshow(i,'gray'),plt.title(str(cv2.matchShapes(hm,tmp[1][j-1],2,0.0)))
	j=j+1
#print cv2.matchShapes(tmp[1][1],tmp[1][4],1,0.0)
#print cv2.matchShapes(tmp[1][5],tmp[1][8],1,0.0)
#plt.subplot(111),plt.imshow(number_mod,'gray')

plt.show()
