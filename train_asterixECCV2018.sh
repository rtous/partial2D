dataroot_cropped="data/H36M_ECCV18/H36M_ECCV18_HOLLYWOOD"
dataroot_original="data/H36M_ECCV18/keyponts_generated_by_openpose_for_train_images_no_sufix"
#ORIGINAL_IMAGES_PATH = "/Volumes/ElementsDat/pose/COCO/train2017"
OUTPUTPATH="data/output/ECCV2018"
dataroot_validation="/home/users/jpoveda/charade/input/keypoints"
TEST_IMAGES_PATH="/home/users/jpoveda/charade/input/images"

python test12_poseCompletion4_CGAN_v3_justpatch_charade.py $dataroot_cropped $dataroot_original $OUTPUTPATH $dataroot_validation $TEST_IMAGES_PATH
#nohup python test12_poseCompletion4_CGAN_v3_justpatch_charade.py $dataroot_cropped $dataroot_original $OUTPUTPATH $dataroot_validation $TEST_IMAGES_PATH &
#nohup python test12_poseCompletion4_CGAN_v3_justpatch_charade.py $dataroot_cropped $dataroot_original $OUTPUTPATH $dataroot_validation $TEST_IMAGES_PATH >nohup.out 2>&1 &

