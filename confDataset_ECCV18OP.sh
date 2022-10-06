DATASET_NAME="ECCV18OP"
DATASET_CROPPED="data/ECCV18OP_crop"
DATASET_ORIGINAL="data/ECCV18OP_onlyused"
#OUTPUTPATH="data/output/ECCV18OP_DAE"
DATASET_CHARADE="dynamicData/charade/input/keypoints"
DATASET_CHARADE_IMAGES="dynamicData/charade/input/images"
#DATASET_TEST="dynamicData/ECCV18OD_test_crop"
#DATASET_TEST_IMAGES="/Volumes/ElementsDat/pose/H36M/ECCV2018/ECCV18_Challenge/Train/IMG/"
BODY_MODEL=BodyModelOPENPOSE25
DATASET_MODULE="datasetBasic"
LEN_BUFFER_ORIGINALS=65536 #1000 65536
CROPPED_VARIATIONS=1 #1 (defalut) 0 to learn to copy
DISCARDINCOMPLETEPOSES=1 #1

#INFERENCE
DATASET_TEST="dynamicData/ECCV18OD_test_crop"
#DATASET_TEST="dynamicData/H36Mtest_v2" #no null keypoints
DATASET_TEST_IMAGES="/Volumes/ElementsDat/pose/H36M/ECCV2018/ECCV18_Challenge/Train/IMG/"
ONLY15=0




