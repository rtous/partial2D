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

DATASET1="dynamicData/H36M_ECCV18"

DATASET2="data/output/ECCV2018v13/TEST/keypoints"
#DATASET2="data/output/COCOlite"
#DATASET2="dynamicData/H36M_ECCV18_HOLLYWOOD_test"
#DATASET2="data/output/ECCV2018v13/TEST/keypoints"
#DATASET2="data/output/ECCV2018v13/CHARADE/keypoints"
#DATASET2="dynamicData/H36M_ECCV18_HOLLYWOOD_original_test"
#DATASET2="/Volumes/ElementsDat/pose/H36M/ECCV2018/keyponts_generated_by_openpose_for_train_images_no_sufix"

SIZE1 = 35000
SIZE2 = 64
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
	return vectors

total = readDataset(DATASET1, SIZE1, DIMENSIONS)
for i in range(10):
	idx = random.randint(0, SIZE1)
	print("act1 = total["+str(idx)+":"+str(idx+SIZE2)+"]")
	act1 = total[idx:idx+SIZE2]
	#act2 = numpy.concatenate(total[:idx], total[idx+SIZE2:])
	act2 = total[idx:]
	fid = calculate_fid(act1, act2)
	print('FID: %.3f' % fid)
#total = readDataset(DATASET1, SIZE1, DIMENSIONS)
#act1 = total[:int(SIZE1/8)]
#act2 = total[int(SIZE1/8):]

'''
act1 = readDataset(DATASET1, SIZE1, DIMENSIONS)
act2 = readDataset(DATASET2, SIZE2, DIMENSIONS)


# fid between act1 and act2
fid = calculate_fid(act1, act2)
print('FID: %.3f' % fid)
'''



