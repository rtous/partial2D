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
from scipy.linalg import sqrtm
import random
import scipy.stats as stats
#import h36mIterator_tiny as h36mIterator #DEBUG import h36mIterator
import h36mIterator
import random
import BodyModelOPENPOSE15 #borrar
import traceback
import json

CRED = '\033[91m'
CGREEN  = '\33[32m'
CYELLOW = '\33[33m'
CBLUE   = '\33[34m'
CBOLD     = '\33[1m'
CEND = '\033[0m'

argv = sys.argv
try:
    DATASET_CANDIDATE=argv[1]
    DATASET_CROPPED=argv[2]
    DATASET_ORIIGNAL=argv[3]
    DATASET_CANDIDATE_MAX=int(argv[4])
    DATASET_REFERENCE=argv[5]
    DATASET_REFERENCE_MAX=int(argv[6])
    numJoints=int(argv[7])
    if argv[8]=="0":
        DISCARDINCOMPLETEPOSES=False
    else:
        DISCARDINCOMPLETEPOSES=True
    #NUMJOINTS NOT USED
    OUTPUTPATH=argv[9]
    #print("OUTPUTPATH=", OUTPUTPATH)
except ValueError:
    print("Wrong arguments.")
    traceback.print_exc()
    sys.exit()	

####### INITIAL WARNINGS ########
if not DATASET_CANDIDATE=="data/output/H36M/TEST/keypoints":
    print(CRED + "DATASET_CANDIDATE=" + str(DATASET_CANDIDATE) + CEND)
else:
    print(CGREEN + "DATASET_CANDIDATE=" + str(DATASET_CANDIDATE) + CEND)
if DATASET_REFERENCE_MAX<65536:
    print(CRED + "DATASET_REFERENCE_MAX=" + str(DATASET_REFERENCE_MAX) + CEND)
else:
    print(CGREEN + "DATASET_REFERENCE_MAX=" + str(DATASET_REFERENCE_MAX) + CEND)
if not DISCARDINCOMPLETEPOSES:
    print(CRED + "DISCARDINCOMPLETEPOSES=" + str(DISCARDINCOMPLETEPOSES) + CEND)
else:
    print(CGREEN + "DISCARDINCOMPLETEPOSES=" + str(DISCARDINCOMPLETEPOSES) + CEND)
###########


DIMENSIONS = numJoints*2

def removeConfidence(flatKeypoints):
	#return flatKeypoints[::3]
	#return numpy.delete(flatKeypoints, numpy.argwhere(numpy.mod(3)))
	#return flatKeypoints[2::3]
	num_keypoints = int(len(flatKeypoints)/3)
	res = numpy.empty(num_keypoints*2)
	for i in range(num_keypoints):
		res[i*2] = flatKeypoints[i*3]
		res[i*2+1] = flatKeypoints[i*3+1]
	return res


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

def readDatasetH36M(size, dims):
	vectors = numpy.zeros(shape=(size, dims))
	scandirIterator = h36mIterator.iterator(DATASET_REFERENCE)
	i = 0
	for keypoints in scandirIterator:
		if i == 0:
			print("First item from the iterator:", keypoints)
		if i >= size:
			break

		keypointsFlat = poseUtils.keypointsListFlatten(keypoints)
		keypointsFlat = removeConfidence(keypointsFlat)
		vectors[i] = keypointsFlat
		i += 1
			
	print("Succesfully read "+str(i)+" files")		
	print("Variance: ", numpy.var(vectors))	
	normality_test_jarque_bera = stats.jarque_bera(vectors)
	print("probability normal Jarque-Bera: ", normality_test_jarque_bera[1])
	#normality_test_kstest = stats.kstest(vectors)
	#print("probability normal Kolmogorov-Smirnov: ", normality_test_kstest[1])

	return vectors[:i], i

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
				keypointsFlat = removeConfidence(keypoints)
				
				#DEBUGGIN###
				'''
				keypoints_cropped, scaleFactor, x_displacement, y_displacement = openPoseUtils.json2normalizedKeypoints(join(path, filename), BodyModelOPENPOSE15, True)
				keypoints_cropped, confidence_values = openPoseUtils.removeConfidence(keypoints_cropped)
            	
            	#flatten
				#keypoints_cropped = [item for sublist in keypoints_cropped for item in sublist]
				#keypoints_cropped = [float(k) for k in keypoints_cropped]
				#keypoints_cropped = keypoints_cropped.flatten()
				#fakeReshapedAsKeypoints = np.reshape(keypoints_cropped, (15, 2))
				#fakeReshapedAsKeypoints = fakeReshapedAsKeypoints.numpy()
				
				denormalizedKeypoints = openPoseUtils.denormalize(fakeReshapedAsKeypoints, scaleFactor, x_displacement, y_displacement)
				#confidence lost
				keypointsFlat = poseUtils.keypointsListFlatten(denormalizedKeypoints)
				'''
				###########

				vectors[i] = keypointsFlat
				i += 1
			except Exception as e:
				print("WARNING: Error reading ", filename)
		     
				traceback.print_exc()
	print("Succesfully read "+str(i)+" files from "+path)		
	print("Variance: ", numpy.var(vectors))	
	#normality_test_jarque_bera = stats.jarque_bera(vectors)
	#print("probability normal Jarque-Bera: ", normality_test_jarque_bera[1])
	return vectors[:i], i

def testWithSlices(vectors, size, sliceSize):
	random.shuffle(vectors)
	for i in range(10):
		idx = random.randint(0, size-sliceSize)
		print("act1 = vectors["+str(idx)+":"+str(idx+sliceSize)+"]")
		act1 = vectors[idx:idx+sliceSize]
		#act2 = numpy.concatenate(total[:idx], total[idx+SIZE2:])
		act2 = numpy.array(list(vectors[:idx]) + list(vectors[idx+sliceSize:]))

		#act2 = total
		fid = calculate_fid(act1, act2)
		print('FID: %.3f' % fid)


def writeRunInfoFile(run_info_json, run_info_file_path):
    run_info_file = open(run_info_file_path, 'w')
    json.dump(run_info_json, run_info_file)
    run_info_file.flush()
    run_info_file.close()

d_ref, d_ref_num = readDatasetH36M(DATASET_REFERENCE_MAX, DIMENSIONS)

d_can, d_can_num = readDataset(DATASET_CANDIDATE, DATASET_CANDIDATE_MAX, DIMENSIONS)

if d_can_num>1000:
    print(CRED + "WARNING! " + DATASET_CANDIDATE + "contains "+str(d_can_num) + CEND)

if d_can_num==0:
    print(CRED + "WARNING! " + DATASET_CANDIDATE + "EMPTY!! "+ CEND)


d_crop, d_crop_num = readDataset(DATASET_CROPPED, DATASET_CANDIDATE_MAX, DIMENSIONS)

d_orig, d_orig_num = readDataset(DATASET_ORIIGNAL, DATASET_CANDIDATE_MAX, DIMENSIONS)

print("sample of "+DATASET_REFERENCE+":")
print(d_ref[0])
fid = calculate_fid(d_ref, d_ref)
print("SELF")
print('FID: %.3f' % fid)

testWithSlices(d_ref, d_ref_num, 1000)

fid = calculate_fid(d_ref[:int(d_ref_num/2)], d_ref[int(d_ref_num/2):])
print("HALF vs HALF")
print('FID: %.3f' % fid)

# fid between act1 and act2
fid = calculate_fid(d_ref, d_can)
print(DATASET_CANDIDATE+" size="+str(d_can_num))
print('FID: %.3f' % fid)

#write to file
f = open(OUTPUTPATH+"/run_info.json", 'r')
run_info_json = json.load(f)
f.close()
run_info_json["results"].append({'FID': fid})
writeRunInfoFile(run_info_json, OUTPUTPATH+"/run_info.json")

fid = calculate_fid(d_ref, d_crop)
print(DATASET_CROPPED+" size="+str(d_crop_num))
print('FID: %.3f' % fid)

fid = calculate_fid(d_ref, d_orig)
print(DATASET_ORIIGNAL+" size="+str(d_orig_num))
print('FID: %.3f' % fid)






