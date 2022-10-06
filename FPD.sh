if [ $# -eq 0 ]
then
    echo "No arguments supplied"
    echo "USAGE: ./FPD.sh DATASET_CONFIGURATION_FILE.sh DATASET_CONFIGURATION_FILE.sh [0/1]"
    echo "The optional param 0/1 signals laptop/office"
else
	######################
	if [ $# -eq 3 ]
	then
		SETUP=$3
	else
		SETUP=0 #0=laptop, 1=office
	fi
	######################
	if [ $SETUP -eq 0 ]
	then   
	    DATASET_REFERENCE="/Volumes/ElementsDat/pose/H36M/H36M"
	    #DATASET_ORIGINAL="/Volumes/ElementsDat/pose/H36M/ECCV2018/ECCV18OD_no_sufix"
	else
	    DATASET_REFERENCE="/mnt/f/datasets/pose/H36M/H36M"
	    #DATASET_ORIGINAL="/Volumes/ElementsDat/pose/H36M/ECCV2018/ECCV18OD_no_sufix"
	fi
	######################

	source $1

	source $2

	OUTPUTPATH="data/output/"$MODEL_NAME"_"$DATASET_NAME

	DATASET_CANDIDATE=$OUTPUTPATH"/TEST/keypoints"

	if [ $DISCARDINCOMPLETEPOSES -eq 0 ]
	then 
		
		DATASET_CROPPED="dynamicData/H36Mtest"
		DATASET_ORIGNAL="dynamicData/H36Mtest_original_noreps"
	else
		DATASET_CROPPED="dynamicData/H36Mtest_v2"
		DATASET_ORIGNAL="dynamicData/H36Mtest_original_v2_noreps"
	fi
	##############

	DATASET_CANDIDATE_MAX=35000
	DATASET_REFERENCE_MAX=200000
	NUMJOINTS=15 #TODO!

	echo "DATASET_CANDIDATE="$DATASET_CANDIDATE
	echo "DATASET_CROPPED="$DATASET_CROPPED
	echo "DATASET_ORIGNAL="$DATASET_ORIGNAL
	echo "DATASET_CANDIDATE_MAX="$DATASET_CANDIDATE_MAX
	echo "DATASET_REFERENCE="$DATASET_REFERENCE
	echo "DATASET_REFERENCE_MAX="$DATASET_REFERENCE_MAX
	echo "NUMJOINTS="$NUMJOINTS
	echo "DISCARDINCOMPLETEPOSES="$DISCARDINCOMPLETEPOSES
	echo "OUTPUTPATH="$OUTPUTPATH

	python FPD_H36M.py $DATASET_CANDIDATE $DATASET_CROPPED $DATASET_ORIGNAL $DATASET_CANDIDATE_MAX $DATASET_REFERENCE $DATASET_REFERENCE_MAX $NUMJOINTS $DISCARDINCOMPLETEPOSES $OUTPUTPATH
fi
