DATASET_CROPPED="data/H36M_ECCV18_HOLLYWOOD"
DATASET_ORIGINAL="/Volumes/ElementsDat/pose/H36M/ECCV2018/keyponts_generated_by_openpose_for_train_images_no_sufix"
OUTPUTPATH="data/output/ECCV2018v13"
DATASET_CHARADE="/Users/rtous/DockerVolume/charade/input/keypoints"
DATASET_CHARADE_IMAGES="/Users/rtous/DockerVolume/charade/input/images"
DATASET_TEST="dynamicData/H36M_ECCV18_HOLLYWOOD_test"
DATASET_TEST_IMAGES="/Volumes/ElementsDat/pose/H36M/ECCV2018/ECCV18_Challenge/Train/IMG/"

python test13_plus_regression_like_context_encoder.py $DATASET_CROPPED $DATASET_ORIGINAL $OUTPUTPATH $DATASET_CHARADE $DATASET_CHARADE_IMAGES $DATASET_TEST $DATASET_TEST_IMAGES

