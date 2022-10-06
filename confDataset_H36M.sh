DATASET_NAME="H36M"
DATASET_CROPPED="NOTSPECIFIED"
#DATASET_ORIGINAL="/Volumes/ElementsDat/pose/H36M/ECCV2018/ECCV18OD_no_sufix"
#OUTPUTPATH="data/output/H36M_DAE"
DATASET_CHARADE="dynamicData/charade/input/keypoints"
DATASET_CHARADE_IMAGES="dynamicData/charade/input/images"
#DATASET_TEST="dynamicData/ECCV18OD_test_crop"
#DATASET_TEST_IMAGES="/Volumes/ElementsDat/pose/H36M/ECCV2018/ECCV18_Challenge/Train/IMG/"
BODY_MODEL=BodyModelOPENPOSE15
DATASET_MODULE="datasetH36M"
LEN_BUFFER_ORIGINALS=65536 #1000 65536
CROPPED_VARIATIONS=1 #1 (defalut) 0 to learn to copy
DISCARDINCOMPLETEPOSES=1 #1

#INFERENCE
if [ $DISCARDINCOMPLETEPOSES -eq 1 ]
then
    DATASET_TEST="dynamicData/H36Mtest_v2"
else
	DATASET_TEST="dynamicData/H36Mtest"
fi
#DATASET_TEST="dynamicData/H36Mtest_v2" #no null keypoints
DATASET_TEST_IMAGES="UNKNOWN"
ONLY15=1




