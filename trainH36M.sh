DATASET_CROPPED="NOTSPECIFIED"
DATASET_ORIGINAL="/Volumes/ElementsDat/pose/H36M/H36M"
#DATASET_ORIGINAL="/Volumes/ElementsDat/pose/H36M/ECCV2018/ECCV18OD_no_sufix"
OUTPUTPATH="data/output/H36M"
DATASET_CHARADE="/Users/rtous/DockerVolume/charade/input/keypoints"
DATASET_CHARADE_IMAGES="/Users/rtous/DockerVolume/charade/input/images"
DATASET_TEST="dynamicData/ECCV18OD_test_crop"
DATASET_TEST_IMAGES="/Volumes/ElementsDat/pose/H36M/ECCV2018/ECCV18_Challenge/Train/IMG/"
python train.py $DATASET_CROPPED $DATASET_ORIGINAL $OUTPUTPATH $DATASET_CHARADE $DATASET_CHARADE_IMAGES $DATASET_TEST $DATASET_TEST_IMAGES
#python test13_plus_regression_like_context_encoder.py $DATASET_CROPPED $DATASET_ORIGINAL $OUTPUTPATH $DATASET_CHARADE $DATASET_CHARADE_IMAGES $DATASET_TEST $DATASET_TEST_IMAGES

