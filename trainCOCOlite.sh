dataroot_cropped="/Volumes/ElementsDat/pose/COCO/ruben_structure/keypoints_openpose_format_cropped_lite"
dataroot_original="/Volumes/ElementsDat/pose/COCO/ruben_structure/keypoints_openpose_format_lite"
#ORIGINAL_IMAGES_PATH = "/Volumes/ElementsDat/pose/COCO/train2017"
OUTPUTPATH="data/output/COCOlite"
dataroot_validation="/Users/rtous/DockerVolume/charade/input/keypoints"
TEST_IMAGES_PATH="/Users/rtous/DockerVolume/charade/input/images"

python test12_poseCompletion4_CGAN_v3_justpatch_charade.py $dataroot_cropped $dataroot_original $OUTPUTPATH $dataroot_validation $TEST_IMAGES_PATH

