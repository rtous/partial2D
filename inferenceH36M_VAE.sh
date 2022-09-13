SETUP=1 #0=laptop, 1=office

if [ $SETUP -eq 0 ]
then
    DATASET_CROPPED="NOTSPECIFIED"
    DATASET_ORIGINAL="/Volumes/ElementsDat/pose/H36M/H36M"
    #DATASET_ORIGINAL="/Volumes/ElementsDat/pose/H36M/ECCV2018/ECCV18OD_no_sufix"
    DATASET_TEST_IMAGES="/Volumes/ElementsDat/pose/H36M/ECCV2018/ECCV18_Challenge/Train/IMG/"
else
    DATASET_CROPPED="NOTSPECIFIED"
    DATASET_ORIGINAL="/mnt/f/datasets/pose/H36M/H36M"
    #DATASET_ORIGINAL="/Volumes/ElementsDat/pose/H36M/ECCV2018/ECCV18OD_no_sufix"
    DATASET_TEST_IMAGES="/Volumes/ElementsDat/pose/H36M/ECCV2018/ECCV18_Challenge/Train/IMG/"
fi

OUTPUTPATH="data/output/H36M_VAE"
DATASET_CHARADE="dynamicData/charade/input/keypoints"
DATASET_CHARADE_IMAGES="dynamicData/charade/input/images"
DATASET_TEST="dynamicData/H36Mtest"
DATASET_TEST_IMAGES="UNKNOWN"
#DATASET_TEST="dynamicData/ECCV18OP_test_crop"
#DATASET_TEST_IMAGES="/Volumes/ElementsDat/pose/H36M/ECCV2018/ECCV18_Challenge/Train/IMG/"
#MODELPATH="data/output/H36M/model/model_epoch4_batch12000.pt"
#MODELPATH="data/output/H36M/model/model_epoch9_batch5000.pt"

MODELPATH="data/output/H36M_VAE/model/model_epoch99_batch0.pt"
#MODELPATH="dynamicData/models/H36M_GAN_epoch7_batch2000/H36M_GAN_epoch7_batch2000.pt"

MODEL="models_VAE"
ONLY15=1
BODY_MODEL="OPENPOSE_15"

python inference_VAE.py $DATASET_CROPPED $DATASET_ORIGINAL $OUTPUTPATH $DATASET_CHARADE $DATASET_CHARADE_IMAGES $DATASET_TEST $DATASET_TEST_IMAGES $MODELPATH $MODEL $ONLY15 $BODY_MODEL

