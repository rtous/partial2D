SETUP=1 #0=laptop, 1=office

if [ $SETUP -eq 0 ]
then   
    DATASET_ORIGINAL="/Volumes/ElementsDat/pose/H36M/H36M"
    #DATASET_ORIGINAL="/Volumes/ElementsDat/pose/H36M/ECCV2018/ECCV18OD_no_sufix"
else
    DATASET_ORIGINAL="/mnt/f/datasets/pose/H36M/H36M"
    #DATASET_ORIGINAL="/Volumes/ElementsDat/pose/H36M/ECCV2018/ECCV18OD_no_sufix"
fi


DATASET_CROPPED="NOTSPECIFIED"
OUTPUTPATH="data/output/H36M"
DATASET_CHARADE="dynamicData/charade/input/keypoints"
DATASET_CHARADE_IMAGES="dynamicData/charade/input/images"
#DATASET_TEST="dynamicData/H36Mtest_v2" #no null keypoints
#DATASET_TEST="dynamicData/H36Mtest"    #with null keypoints
DATASET_TEST="dynamicData/H36Mtest_original_v2_noreps" #debug copy
DATASET_TEST_IMAGES="UNKNOWN"



#DATASET_TEST="dynamicData/ECCV18OP_test_crop"
#DATASET_TEST_IMAGES="/Volumes/ElementsDat/pose/H36M/ECCV2018/ECCV18_Challenge/Train/IMG/"
#MODELPATH="data/output/H36M/model/model_epoch4_batch12000.pt"
#MODELPATH="data/output/H36M/model/model_epoch9_batch5000.pt"

MODELPATH="data/output/H36M/model/model_epoch99_batch0.pt"
#MODELPATH="dynamicData/models/H36M_GAN_epoch7_batch2000/H36M_GAN_epoch1_batch1000.pt"

MODEL="models" #models models_mirror models_simple
ONLY15=1
BODY_MODEL="OPENPOSE_15"
NORMALIZATION="center_scale" #"center_scale", "basic", "none" 
KEYPOINT_RESTORATION=0 #1 (defalut)
NZ=100 #100 #0



python inference.py $DATASET_CROPPED $DATASET_ORIGINAL $OUTPUTPATH $DATASET_CHARADE $DATASET_CHARADE_IMAGES $DATASET_TEST $DATASET_TEST_IMAGES $MODELPATH $MODEL $ONLY15 $BODY_MODEL $NORMALIZATION $KEYPOINT_RESTORATION $NZ


