if [ $# -eq 0 ]
then
    echo "No arguments supplied"
    echo "USAGE: ./trainH36M CONFIGURATION_FILE.sh [0/1]"
    echo "The optional param 0/1 signals laptop/office"
else
	######################
	if [ $# -eq 2 ]
	then
		SETUP=$2
	else
		SETUP=0 #0=laptop, 1=office
	fi
	######################
	if [ $SETUP -eq 0 ]
	then   
	    DATASET_ORIGINAL="/Volumes/ElementsDat/pose/H36M/H36M"
	    #DATASET_ORIGINAL="/Volumes/ElementsDat/pose/H36M/ECCV2018/ECCV18OD_no_sufix"
	else
	    DATASET_ORIGINAL="/mnt/f/datasets/pose/H36M/H36M"
	    #DATASET_ORIGINAL="/Volumes/ElementsDat/pose/H36M/ECCV2018/ECCV18OD_no_sufix"
	fi
	######################

	source $1

	python train.py $DATASET_CROPPED $DATASET_ORIGINAL $OUTPUTPATH $DATASET_CHARADE $DATASET_CHARADE_IMAGES $DATASET_TEST $DATASET_TEST_IMAGES $BODY_MODEL $DATASET_MODULE $MODEL $NORMALIZATION $KEYPOINT_RESTORATION $LEN_BUFFER_ORIGINALS $CROPPED_VARIATIONS $NZ $DISCARDINCOMPLETEPOSES $TRAINSPLIT
	#python test13_plus_regression_like_context_encoder.py $DATASET_CROPPED $DATASET_ORIGINAL $OUTPUTPATH $DATASET_CHARADE $DATASET_CHARADE_IMAGES $DATASET_TEST $DATASET_TEST_IMAGES
fi
