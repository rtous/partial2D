#dataroot_cropped="data/H36M_ECCV18_HOLLYWOOD"
dataroot_cropped="data/ECCV18OP_crop"

dataroot_original="data/ECCV18OP_onlyused"
#dataroot_original="/Volumes/ElementsDat/pose/H36M/ECCV2018/keyponts_generated_by_openpose_for_train_images_no_sufix"
#dataroot_original="/Volumes/ElementsDat/pose/H36M/ECCV2018/keyponts_generated_by_openpose_for_train_images"


#OUTPUTPATH="data/output/ECCV2018"
OUTPUTPATH="data/output/ECCV2018_DEBUG"
dataroot_validation="/Users/rtous/DockerVolume/charade/input/keypoints"
TEST_IMAGES_PATH="/Users/rtous/DockerVolume/charade/input/images"

python test12_poseCompletion4_CGAN_v3_justpatch_charade.py $dataroot_cropped $dataroot_original $OUTPUTPATH $dataroot_validation $TEST_IMAGES_PATH

