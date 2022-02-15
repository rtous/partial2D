DATASET_CROPPED="NOTSPECIFIED"
DATASET_ORIGINAL="/Volumes/ElementsDat/pose/H36M/H36M/H36M/"
OUTPUTPATH="data/output/H36M"
DATASET_CHARADE="/Users/rtous/DockerVolume/charade/input/keypoints"
DATASET_CHARADE_IMAGES="/Users/rtous/DockerVolume/charade/input/images"
DATASET_TEST="dynamicData/ECCV18OP_test_crop"
DATASET_TEST_IMAGES="/Volumes/ElementsDat/pose/H36M/ECCV2018/ECCV18_Challenge/Train/IMG/"
MODELPATH="data/output/H36M/model_epoch0_batch6000.pt"
python inference.py $DATASET_CROPPED $DATASET_ORIGINAL $OUTPUTPATH $DATASET_CHARADE $DATASET_CHARADE_IMAGES $DATASET_TEST $DATASET_TEST_IMAGES $MODELPATH

