dataroot_cropped="data/keypoints_openpose_format_cropped_lite"
dataroot_original="data/ruben_structure/keypoints_openpose_format_lite"
#ORIGINAL_IMAGES_PATH = "/Volumes/ElementsDat/pose/COCO/train2017"
OUTPUTPATH="data/output2"
dataroot_validation="/home/users/jpoveda/charade/input/keypoints"
TEST_IMAGES_PATH="/home/users/jpoveda/charade/input/images"

python test12_poseCompletion4_CGAN_v3_justpatch_charade.py $dataroot_cropped $dataroot_original $OUTPUTPATH $dataroot_validation $TEST_IMAGES_PATH

