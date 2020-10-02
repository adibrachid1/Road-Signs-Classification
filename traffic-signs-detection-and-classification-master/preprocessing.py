import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
from skimage import exposure
import csv
import time
import sys


def readTrafficSigns(datapath):
	'''
	read images and labels of the german traffic signs dataset
	'''
	images = []
	labels = []
	print('[*] Loading images ...')
	t = time.time()
	try:
		for c in range(0,43):
			classPath = datapath + '/' + format(c, '05d') + '/'
			with open(classPath + 'GT-'+ format(c, '05d') + '.csv') as annotationsFile:
				annotations = csv.reader(annotationsFile, delimiter=';')
				next(annotations)
				for row in annotations:
					filename = row[0]
					imageclass = row[7]
					images.append(cv2.imread(classPath + filename))
					labels.append(imageclass)
		print('[*] Done in {:.3f}s.'.format(time.time()-t))
	except Exception as e:
		print('[x] Failed loading images.')
		print('FileNotFoundError: {0}'.format(e))
	return images, labels

def loadData(label='features'):
	'''
	Load training and testing data
	label can be one of these: ['sizes', 'coords', 'features', 'labels'] (features for images)
	'''
	trainData = 'data/train.p'
	testData  = 'data/test.p'

	with open(trainData, mode='rb') as f:
		train = pickle.load(f)
	with open(testData, mode='rb') as f:
		test = pickle.load(f)

	return train[label],test[label]


def grayScale(img):
	'''
	Simply convert a given image to gray
	'''
	dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # or maybe COLOR_BGR2GRAY ?
	return dst

def eqHist(img):
	'''
	Histogram Equalization to improve the contrast of the image
	'''
	dst = grayScale(img)
	dst = cv2.equalizeHist(dst)
	return dst

def reshape(img):
	'''
	Resize given image to 32x32, still not sure to use this to resize
	'''
	try:
		rows, cols, channels = map(int, img.shape)
	except:
		rows, cols = map(int, img.shape)
	if rows == 32 and cols == 32:
		return img
	dst = cv2.resize(img, (32,32), interpolation=cv2.INTER_CUBIC)
	return dst




if __name__ == "__main__":
	datapath = "data/GTSRB/Final_Training/Images/"
	images, labels = readTrafficSigns(datapath)
	for img in images:
		cv2.imshow("TEST", img)
		cv2.waitKey(0)
		break
	cv2.destroyAllWindows()

	# TEST OK