#!/usr/local/bin/python2.7
#python findFeatures.py -t dataset/train/

import argparse as ap
import cv2
import numpy as np
import os
from sklearn.externals import joblib
from scipy.cluster.vq import *

from sklearn import preprocessing
from rootsift import RootSIFT
import math

import time
import cPickle

# Get the path of the training set
parser = ap.ArgumentParser()
parser.add_argument("-t", "--trainingSet", help="Path to Training Set", required="True")
args = vars(parser.parse_args())

# Get the training classes names and store them in a list
train_path = args["trainingSet"]
#train_path = "dataset/train/"

# training_names = os.listdir(train_path)
sub_folders = [ f for f in os.listdir(train_path) if not f.startswith(".")]
image_paths = []
for folder in sub_folders:
	folder_path = os.path.join(train_path, folder)
	for f in [ g for g in os.listdir(folder_path) if g.endswith(".jpg")]:
		image_path = os.path.join(folder_path, f)
		image_paths.append(image_path)


numWords = 1000

# Get all the path to the images and save them in a list
# image_paths and the corresponding label in image_paths
# image_paths = []
# for training_name in training_names:
#     image_path = os.path.join(train_path, training_name)
#     image_paths += [image_path]

# Create feature extraction and keypoint detector objects
# below 2 liens are original code.
# fea_det = cv2.FeatureDetector_create("SIFT")
# des_ext = cv2.DescriptorExtractor_create("SIFT")
sift = cv2.xfeatures2d.SIFT_create()

# List where all the descriptors are stored
des_list = []

try:

    for i, image_path in enumerate(image_paths):
        im = cv2.imread(image_path)
        print "Extract SIFT of %d of %d images" % (i, len(image_paths))
        # 2 lines of original code
        # kpts = fea_det.detect(im)
        # kpts, des = des_ext.compute(im, kpts)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        (kpts, des) = sift.detectAndCompute(gray, None)
        print "# kps: {}, descriptors: {}".format(len(kpts), des.shape)

        # rootsift
        #rs = RootSIFT()
        #des = rs.compute(kpts, des)
        des_list.append((image_path, des))
        
    # Stack all the descriptors vertically in a numpy array
    #downsampling = 1
    #descriptors = des_list[0][1][::downsampling,:]
    #for image_path, descriptor in des_list[1:]:
    #    descriptors = np.vstack((descriptors, descriptor[::downsampling,:]))


    # Stack all the descriptors vertically in a numpy array
    descriptors = des_list[0][1]
    for image_path, descriptor in des_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))  

    start = time.time()

    # Perform k-means clustering
    print "Start k-means: %d words, %d key points" %(numWords, descriptors.shape[0])

    voc, variance = kmeans(descriptors, numWords, 1)

    end = time.time()
    print end-start

    # Calculate the histogram of features
    im_features = np.zeros((len(image_paths), numWords), "float32")
    for i in xrange(len(image_paths)):
        words, distance = vq(des_list[i][1],voc)
        for w in words:
            im_features[i][w] += 1

    # Perform Tf-Idf vectorization
    nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
    idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

    # Perform L2 normalization
    im_features = im_features*idf
    im_features = preprocessing.normalize(im_features, norm='l2')

    joblib.dump((im_features, image_paths, idf, numWords, voc), "bof.pkl", compress=3)       

except:
    with open("descriptors", "wb") as f:
        cPickle.dump(descriptors, f)
    print "out of memory"
