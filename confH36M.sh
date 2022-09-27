DATASET_CROPPED="NOTSPECIFIED"
#DATASET_ORIGINAL="/Volumes/ElementsDat/pose/H36M/ECCV2018/ECCV18OD_no_sufix"
OUTPUTPATH="data/output/H36M"
DATASET_CHARADE="dynamicData/charade/input/keypoints"
DATASET_CHARADE_IMAGES="dynamicData/charade/input/images"
#DATASET_TEST="dynamicData/ECCV18OD_test_crop"
#DATASET_TEST_IMAGES="/Volumes/ElementsDat/pose/H36M/ECCV2018/ECCV18_Challenge/Train/IMG/"
BODY_MODEL="OPENPOSE_15"
DATASET_MODULE="datasetH36M"
MODEL="models" #models models_mirror models_simple
NORMALIZATION="center_scale" #"center_scale", "basic", "none" 
KEYPOINT_RESTORATION=1
LEN_BUFFER_ORIGINALS=100000 #1000 65536
CROPPED_VARIATIONS=1 #1 (defalut) 0 to learn to copy
NZ=0 #100 #10 #0
DISCARDINCOMPLETEPOSES=1 #1
TRAINSPLIT=1 #0.8

#INFERENCE
DATASET_TEST="dynamicData/H36Mtest"
#DATASET_TEST="dynamicData/H36Mtest_v2" #no null keypoints
DATASET_TEST_IMAGES="UNKNOWN"
MODELPATH="data/output/H36M/model/model_epoch0_batch12000.pt"
ONLY15=1




