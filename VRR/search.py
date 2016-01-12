#!/usr/local/bin/python2.7
#python search.py -i dataset/train/ukbench00000.jpg

import argparse as ap
import cv2
import imutils 
import numpy as np
import os
from sklearn.externals import joblib
from scipy.cluster.vq import *

from sklearn import preprocessing
import numpy as np

from pylab import *
from PIL import Image
# from rootsift import RootSIFT

# this is soooooo ugly
BASE_ENGINE_DIR = os.path.abspath("..")

def search_api(image_path):
	im, rank_ID, image_paths = do_search(image_path)
	return get_query(im, rank_ID, image_paths)

def do_search(image_path):
	pkl_path = os.path.join(".", "bof.pkl")
	# Load the classifier, class names, scaler, number of clusters and vocabulary 
	im_features, image_paths, idf, numWords, voc = joblib.load(pkl_path)
	    
	# Create feature extraction and keypoint detector objects
	# fea_det = cv2.FeatureDetector_create("SIFT")
	# des_ext = cv2.DescriptorExtractor_create("SIFT")
	sift = cv2.xfeatures2d.SIFT_create()


	# List where all the descriptors are stored
	des_list = []

	im = cv2.imread(image_path)
	g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	(kpts, des) = sift.detectAndCompute(g, None)
	print "# kps: {}, descriptors: {}".format(len(kpts), des.shape)

	# rootsift
	#rs = RootSIFT()
	#des = rs.compute(kpts, des)

	des_list.append((image_path, des))   
	    
	# Stack all the descriptors vertically in a numpy array
	descriptors = des_list[0][1]

	# 
	test_features = np.zeros((1, numWords), "float32")
	words, distance = vq(descriptors,voc)
	for w in words:
	    test_features[0][w] += 1

	# Perform Tf-Idf vectorization and L2 normalization
	test_features = test_features*idf
	test_features = preprocessing.normalize(test_features, norm='l2')

	score = np.dot(test_features, im_features.T)
	rank_ID = np.argsort(-score)

	return (im, rank_ID, image_paths)


def get_query(im, rank_ID, image_paths):
	image_list = []
	for i, ID in enumerate(rank_ID[0][0:20]):
		image_subpath = '/'.join(
			image_paths[ID].split('/')[2:]
			)
		image_list.append(image_subpath)
	return image_list


def visualize(im, rank_ID, image_paths):
	# Visualize the results
	figure()
	gray()
	subplot(5,4,1)
	imshow(im[:,:,::-1])
	axis('off')
	for i, ID in enumerate(rank_ID[0][0:16]):
		img_path = os.path.join(BASE_ENGINE_DIR, image_paths[ID])
		img = Image.open(img_path)
		gray()
		subplot(5,4,i+5)
		imshow(img)
		axis('off')

	show()  

if __name__ == '__main__':
		# Get the path of the training set
	parser = ap.ArgumentParser()
	parser.add_argument("-i", "--image", help="Path to query image", required="True")
	args = vars(parser.parse_args())

	# Get query image path
	image_path = os.path.abspath(args["image"])

	im, rank_ID, image_paths = do_search(image_path)
	visualize(im, rank_ID, image_paths)