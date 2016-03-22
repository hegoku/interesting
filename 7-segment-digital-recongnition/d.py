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

def bitand(img1,img2):
	bit=0.0;
	re=np.zeros((32+2,40), np.uint8)
	for i in range(len(img1)):
		for j in range(len(img1[i])):
			if(img1[i][j]!=img2[i][j]):
				re[i][j]=255
				bit=bit+1
	return [re,bit/(34*40)]

def getContoursRectImage(img):
	result_img=[]
	result_hu=[]
	image, contours, hierarchy = cv2.findContours(img.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	#img=cv2.drawContours(img,contours,-1,(255,0,0),3)
	for ctr in contours:
		rect=cv2.boundingRect(ctr)
		roi = img[rect[1]:rect[3]+rect[1],rect[0]:rect[2]+rect[0]]
		width=int(32*rect[2]/rect[3])
		roi = cv2.resize(roi, (width, 32), interpolation=cv2.INTER_AREA)
		#blank_image = np.zeros((32+2,width+2), np.uint8)
		#blank_image[1:32+1,1:width+1]=roi
		blank_image = np.zeros((32+2,40), np.uint8)
		tmpx=int(40/2-width/2)
		if(tmpx<=0):tmpx=0
		blank_image[1:32+1,tmpx-1:tmpx+width-1]=roi
		#blank_image=cv2.Canny(blank_image,1,2)
		m=cv2.moments(blank_image)
		hm=cv2.HuMoments(m)
		result_hu.append(hm)
		result_img.append(blank_image)
	return [result_img,result_hu]

number_mod=cv2.threshold(cv2.cvtColor(cv2.imread('./number/n5.bmp'),cv2.COLOR_BGR2GRAY),128,255,cv2.THRESH_BINARY)[1]
#number_mod=cv2.imread('./number/n2.jpg')
#number_mod=cv2.cvtColor(number_mod,cv2.COLOR_BGR2GRAY) #灰度
#number_mod=cv2.GaussianBlur(number_mod, (1,7), 0)
#ret,number_mod=cv2.threshold(number_mod,128,255,cv2.THRESH_BINARY)

#img=cv2.imread('./number/n.bmp')
#img=cv2.imread('./images/QQ20160317-1.png')
img=cv2.imread('./images/QQ20160318-1.png')
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

number_image=getContoursRectImage(img_binary)

plt.subplot(421),plt.imshow(img_gray,'gray'),plt.title('gray')
zhifangtu(img_gray,422,str(np.mean(img_gray)))

plt.subplot(423),plt.imshow(img_binary,'gray'),plt.title('img_binary')
zhifangtu(img_binary,424,str(maxMinAvarge(img_equ)))

plt.subplot(425),plt.imshow(img,'gray'),plt.title('final')

plt.figure(2)
j=1
for i in number_image[0]:
	plt.subplot(10,3,j),plt.imshow(i,'gray')
	j=j+1


plt.figure(3)
j=1
tmp=getContoursRectImage(number_mod)
plt.subplot(3,5,12),plt.imshow(number_image[0][4],'gray')
i=0
for i in range(len(tmp[0])):
	plt.subplot(3,5,j),plt.imshow(tmp[0][i],'gray'),plt.title(str(cv2.matchShapes(number_image[1][4],tmp[1][i],2,0.0)))
	#plt.subplot(3,5,j),plt.imshow(cv2.matchTemplate(tmp[0][i],number_image[0][4],cv2.TM_CCOEFF_NORMED),'gray'),plt.title( '' )
	#plt.subplot(3,5,j),plt.imshow(cv2.bitwise_and(number_image[0][4],tmp[0][i],False),'gray'),plt.title( '' )
	#aaa=cv2.add(tmp[0][i],number_image[0][4])
	#aaa=cv2.threshold(aaa,255,512,cv2.THRESH_BINARY)
	#aaa=bitand(tmp[0][i],number_image[0][4])
	#plt.subplot(3,5,j),plt.imshow(aaa[0],'gray'),plt.title( str(aaa[1]) )
	j=j+1
#print cv2.matchShapes(tmp[1][1],tmp[1][4],1,0.0)
#print cv2.matchShapes(tmp[1][5],tmp[1][8],1,0.0)
#plt.subplot(111),plt.imshow(number_mod,'gray')

plt.show()
