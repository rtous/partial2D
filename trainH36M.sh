SETUP=0 #0=laptop, 1=office

if [ $SETUP -eq 0 ]
then   
    DATASET_ORIGINAL="/Volumes/ElementsDat/pose/H36M/H36M"
    #DATASET_ORIGINAL="/Volumes/ElementsDat/pose/H36M/ECCV2018/ECCV18OD_no_sufix"
else
    DATASET_ORIGINAL="/mnt/f/datasets/pose/H36M/H36M"
    #DATASET_ORIGINAL="/Volumes/ElementsDat/pose/H36M/ECCV2018/ECCV18OD_no_sufix"
fi

DATASET_CROPPED="NOTSPECIFIED"
#DATASET_ORIGINAL="/Volumes/ElementsDat/pose/H36M/ECCV2018/ECCV18OD_no_sufix"
OUTPUTPATH="data/output/H36M"
DATASET_CHARADE="dynamicData/charade/input/keypoints"
DATASET_CHARADE_IMAGES="dynamicData/charade/input/images"
#DATASET_CHARADE="/Users/rtous/DockerVolume/charade_full/input/keypoints"
#DATASET_CHARADE_IMAGES="/Users/rtous/DockerVolume/charade_full/input/images"
DATASET_TEST="dynamicData/ECCV18OD_test_crop"
DATASET_TEST_IMAGES="/Volumes/ElementsDat/pose/H36M/ECCV2018/ECCV18_Challenge/Train/IMG/"
BODY_MODEL="OPENPOSE_15"
DATASET_MODULE="datasetH36M"
MODEL="models"
python train.py $DATASET_CROPPED $DATASET_ORIGINAL $OUTPUTPATH $DATASET_CHARADE $DATASET_CHARADE_IMAGES $DATASET_TEST $DATASET_TEST_IMAGES $BODY_MODEL $DATASET_MODULE $MODEL
#python test13_plus_regression_like_context_encoder.py $DATASET_CROPPED $DATASET_ORIGINAL $OUTPUTPATH $DATASET_CHARADE $DATASET_CHARADE_IMAGES $DATASET_TEST $DATASET_TEST_IMAGES

