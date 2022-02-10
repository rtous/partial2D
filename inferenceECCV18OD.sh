DATASET_CROPPED="data/ECCV18OD_crop"
DATASET_ORIGINAL="/Volumes/ElementsDat/pose/H36M/ECCV2018/ECCV18OD_no_sufix"
OUTPUTPATH="data/output/ECCV18OD"
DATASET_CHARADE="/Users/rtous/DockerVolume/charade/input/keypoints"
DATASET_CHARADE_IMAGES="/Users/rtous/DockerVolume/charade/input/images"
DATASET_TEST="dynamicData/ECCV18OD_test_crop"
DATASET_TEST_IMAGES="/Volumes/ElementsDat/pose/H36M/ECCV2018/ECCV18_Challenge/Train/IMG/"
python inference.py $DATASET_CROPPED $DATASET_ORIGINAL $OUTPUTPATH $DATASET_CHARADE $DATASET_CHARADE_IMAGES $DATASET_TEST $DATASET_TEST_IMAGES

