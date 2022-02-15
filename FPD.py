#From here: https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/

import math
from os import listdir
from os.path import isfile, join, splitext
import pathlib
import traceback
import argparse
import os
import sys
import openPoseUtils
import poseUtils
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
#from numpy.random import random
from scipy.linalg import sqrtm
import random
import scipy.stats as stats
#import statistics

#DATASET1="dynamicData/H36M_ECCV18"
#DATASET1="/Volumes/ElementsDat/pose/H36M/ECCV2018/keyponts_generated_by_openpose_for_train_images_no_sufix"
#DATASET1="data/H36M_ECCV18_HOLLYWOOD_ORIGINAL_ONLY_THE_CROPPED"
DATASET1="/Volumes/ElementsDat/pose/H36M/H36M/H36M/"

#DATASET2="/Volumes/ElementsDat/pose/H36M/ECCV2018/keyponts_generated_by_openpose_for_train_images_no_sufix"
DATASET2="dynamicData/ECCV18OP_onlyused"
DATASET3="dynamicData/ECCV18OP_test"
DATASET4="data/output/ECCV2018v13/TEST/keypoints"

#DATASET2="/Volumes/ElementsDat/pose/COCO/ruben_structure/keypoints_openpose_format"
#DATASET2="data/output/COCOlite"
#DATASET2="dynamicData/H36M_ECCV18_HOLLYWOOD_test"
#DATASET2="data/output/ECCV2018v13/TEST/keypoints"
#DATASET2="data/output/ECCV2018v13/CHARADE/keypoints"
#DATASET2="dynamicData/H36M_ECCV18_HOLLYWOOD_original_test"

#DATASET2="/Volumes/ElementsDat/pose/H36M/ECCV2018/keyponts_generated_by_openpose_for_train_images_no_sufix"

SIZE1 = 35000
SIZE2 = 35000
DIMENSIONS = 75

'''
argv = sys.argv
try:
    DATASET1=argv[1]
    DATASET2=argv[2]

except ValueError:
    print("Wrong arguments. Expecting two paths.")
'''

# calculate frechet inception distance
def calculate_fid(act1, act2):
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid

def readDataset(path, size, dims):
	vectors = numpy.zeros(shape=(size, dims))
	scandirIterator = os.scandir(path)
	i = 0
	for item in scandirIterator:
		
		if i >= size:
			break

		filename = str(item.name)
		filename_without_extension = os.path.splitext(filename)[0]
		extension = os.path.splitext(filename)[1]
		if extension == ".json":
			try:
				keypoints = openPoseUtils.json2KeypointsFlat(join(path, filename))
				#print("dimensions = ", len(keypoints))
				vectors[i] = keypoints
				i += 1
			except Exception as e:
				print("WARNING: Error reading ", filename)
		     
				traceback.print_exc()
	print("Succesfully read "+str(i)+" files from "+path)		
	print("Variance: ", numpy.var(vectors))	
	normality_test_jarque_bera = stats.jarque_bera(vectors)
	print("probability normal Jarque-Bera: ", normality_test_jarque_bera[1])
	#normality_test_kstest = stats.kstest(vectors)
	#print("probability normal Kolmogorov-Smirnov: ", normality_test_kstest[1])

	return vectors, i
'''
def readDatasetBORRAR(path, pickFileFromThisPath, size, dims):
	vectors = numpy.zeros(shape=(size, dims))
	scandirIterator = os.scandir(path)
	i = 0
	for item in scandirIterator:
		
		if i >= size:
			break

		filename = str(item.name)
		filename_without_extension = os.path.splitext(filename)[0]
		extension = os.path.splitext(filename)[1]
		if extension == ".json":
			try:
				keypoints = openPoseUtils.json2KeypointsFlat(join(pickFileFromThisPath, filename))
				#print("dimensions = ", len(keypoints))
				vectors[i] = keypoints
				i += 1
			except Exception as e:
				print("WARNING: Error reading ", filename)
		     
				traceback.print_exc()
	print("Succesfully read "+str(i)+" files from "+path)
	print("Variance: ", statistics.variance(vectors))		
	return vectors, i
'''

def testWithSlices(vectors, size, sliceSize):
	for i in range(10):
		idx = random.randint(0, size-sliceSize)
		print("act1 = vectors["+str(idx)+":"+str(idx+sliceSize)+"]")
		act1 = vectors[idx:idx+sliceSize]
		#act2 = numpy.concatenate(total[:idx], total[idx+SIZE2:])
		act2 = numpy.array(list(vectors[:idx]) + list(vectors[idx+sliceSize:]))

		#act2 = total
		fid = calculate_fid(act1, act2)
		print('FID: %.3f' % fid)

d1, d1_num = readDataset(DATASET1, SIZE1, DIMENSIONS)
#d2, d2_num = readDatasetBORRAR(DATASET2, DATASET1, SIZE2, DIMENSIONS)
d2, d2_num = readDataset(DATASET2, SIZE2, DIMENSIONS)

d3, d3_num = readDataset(DATASET3, SIZE2, DIMENSIONS)

d4, d4_num = readDataset(DATASET4, SIZE2, DIMENSIONS)

#total = readDataset(DATASET1, SIZE1, DIMENSIONS)
#act1 = total[:int(SIZE1/8)]
#act2 = total[int(SIZE1/8):]
fid = calculate_fid(d1, d1)
print("SELF")
print('FID: %.3f' % fid)

testWithSlices(d1, d1_num, 1000)

fid = calculate_fid(d1[:int(d1_num/2)], d1[int(d1_num/2):])
print("HALF vs HALF")
print('FID: %.3f' % fid)

# fid between act1 and act2
fid = calculate_fid(d1, d2)
print(DATASET2+" size="+str(d2_num))
print('FID: %.3f' % fid)

fid = calculate_fid(d1, d3)
print(DATASET3+" size="+str(d3_num))
print('FID: %.3f' % fid)

fid = calculate_fid(d1, d4)
print(DATASET4+" size="+str(d4_num))
print('FID: %.3f' % fid)






