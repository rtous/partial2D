dataroot_cropped="data/H36M_ECCV18_HOLLYWOOD"
dataroot_original="/Volumes/ElementsDat/pose/H36M/ECCV2018/keyponts_generated_by_openpose_for_train_images_no_sufix"
#dataroot_original="/Volumes/ElementsDat/pose/H36M/ECCV2018/keyponts_generated_by_openpose_for_train_images"


OUTPUTPATH="data/output/ECCV2018v13"
dataroot_validation="/Users/rtous/DockerVolume/charade/input/keypoints"
TEST_IMAGES_PATH="/Users/rtous/DockerVolume/charade/input/images"

python test13_plus_regression_like_context_encoder.py $dataroot_cropped $dataroot_original $OUTPUTPATH $dataroot_validation $TEST_IMAGES_PATH

