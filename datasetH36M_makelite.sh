#datasetH36M

SETUP=0 #0=laptop, 1=office

if [ $SETUP -eq 0 ]
then   
    DATASET_ORIGINAL="/Volumes/ElementsDat/pose/H36M/H36M"
    #DATASET_ORIGINAL="/Volumes/ElementsDat/pose/H36M/ECCV2018/ECCV18OD_no_sufix"
else
    DATASET_ORIGINAL="/mnt/f/datasets/pose/H36M/H36M"
    #DATASET_ORIGINAL="/Volumes/ElementsDat/pose/H36M/ECCV2018/ECCV18OD_no_sufix"
fi

OUTPUTPATH="dynamicData/H36Mtest_v2"
OUTPUTPATH_ORIGINAL="dynamicData/H36Mtest_original_v2"
MAX=1000

python util_makeLiteH36M.py $OUTPUTPATH $OUTPUTPATH_ORIGINAL $MAX $DATASET_ORIGINAL

