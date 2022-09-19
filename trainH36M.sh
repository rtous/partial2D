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
MODEL="models" #models models_mirror models_simple
NORMALIZATION="center_scale" #"center_scale", "basic", "none" 
KEYPOINT_RESTORATION=1
LEN_BUFFER_ORIGINALS=65536 #1000 65536
CROPPED_VARIATIONS=1
NZ=100 #100 #10 #0


python train.py $DATASET_CROPPED $DATASET_ORIGINAL $OUTPUTPATH $DATASET_CHARADE $DATASET_CHARADE_IMAGES $DATASET_TEST $DATASET_TEST_IMAGES $BODY_MODEL $DATASET_MODULE $MODEL $NORMALIZATION $KEYPOINT_RESTORATION $LEN_BUFFER_ORIGINALS $CROPPED_VARIATIONS $NZ
#python test13_plus_regression_like_context_encoder.py $DATASET_CROPPED $DATASET_ORIGINAL $OUTPUTPATH $DATASET_CHARADE $DATASET_CHARADE_IMAGES $DATASET_TEST $DATASET_TEST_IMAGES

