DATASET_CROPPED="data/ECCV18OP_crop"
DATASET_ORIGINAL="/Volumes/ElementsDat/pose/H36M/ECCV2018/ECCV18OP_no_sufix"
OUTPUTPATH="data/output/ECCV18OP_II"
DATASET_CHARADE="/Users/rtous/DockerVolume/charade_full/input/keypoints"
DATASET_CHARADE_IMAGES="/Users/rtous/DockerVolume/charade_full/input/images"
DATASET_TEST="dynamicData/ECCV18OP_test_crop"
DATASET_TEST_IMAGES="/Volumes/ElementsDat/pose/H36M/ECCV2018/ECCV18_Challenge/Train/IMG/"
#MODELPATH="data/output/ECCV18OP_FINAL/model/model_epoch0_batch2000.pt"
MODELPATH="data/output/ECCV18OP_II/model/model_epoch0_batch2000.pt"
#MODEL="modelsECCV18OP"
MODEL="models"
ONLY15=0
BODY_MODEL="OPENPOSE_25"

python inference.py $DATASET_CROPPED $DATASET_ORIGINAL $OUTPUTPATH $DATASET_CHARADE $DATASET_CHARADE_IMAGES $DATASET_TEST $DATASET_TEST_IMAGES $MODELPATH $MODEL $ONLY15 $BODY_MODEL

