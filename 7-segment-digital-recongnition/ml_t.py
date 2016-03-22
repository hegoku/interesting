#coding=utf-8
import cv2
import numpy as np
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import sys

def maxMinAvarge(img):
	max=np.max(img)
	min=np.min(img)
	return (max-min)/2+min

im = cv2.imread('./number/train.jpg')
im3 = im.copy()

gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,11),0)
ret,thresh=cv2.threshold(blur,maxMinAvarge(blur),255,cv2.THRESH_BINARY)
#plt.subplot(1,1,1),plt.imshow(thresh,'gray')
#plt.show()
#################      Now finding Contours         ###################

images,contours,hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

samples =  np.empty((0,2500))
responses = []
keys = [i for i in range(48,58)]
keys.append(46)

for ctr in contours:
	rect=cv2.boundingRect(ctr)
	roi = thresh[rect[1]:rect[3]+rect[1],rect[0]:rect[2]+rect[0]]
	blank_image = np.zeros((50,50), np.uint8)
	x=int(50/2-rect[2]/2)
	y=int(50/2-rect[3]/2)
	blank_image[y:y+rect[3],x:x+rect[2]]=roi
	#plt.subplot(1,1,1),plt.imshow(blank_image,'gray')
	#plt.show()

	cv2.rectangle(im,(rect[0],rect[1]),(rect[2]+rect[0],rect[3]+rect[1]),(0,255,0),1)
	cv2.imshow('norm',im)
	key = cv2.waitKey(0)

	if key == 27:  # (escape to quit)
		sys.exit()
	elif key in keys:
		responses.append(key)
		#responses.append(chr(key))
		sample = blank_image.reshape((1,2500))
		samples = np.append(samples,sample,0)

responses = np.array(responses,np.float32)
responses = responses.reshape((responses.size,1))
print "training complete"

np.savetxt('./generalsamples.data',samples)
np.savetxt('./generalresponses.data',responses)
