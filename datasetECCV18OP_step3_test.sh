#datasetECCV2018

INPUTPATH="data/H36M_ECCV18_HOLLYWOOD"
OUTPUTPATH="dynamicData/H36M_ECCV18_HOLLYWOOD_test"
INPUTPATH_ORIGINAL="/Volumes/ElementsDat/pose/H36M/ECCV2018/keyponts_generated_by_openpose_for_train_images_no_sufix"
OUTPUTPATH_ORIGINAL="dynamicData/H36M_ECCV18_HOLLYWOOD_original_test"
MAX=1000

python util_makeLite.py $INPUTPATH $OUTPUTPATH $INPUTPATH_ORIGINAL $OUTPUTPATH_ORIGINAL $MAX

