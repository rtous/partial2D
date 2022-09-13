#./display.sh 

#Original test keypoints over white

INPUTPATHKEYPOINTS="/Users/rtous/DockerVolume/partial2D/data/H36Mtest_original_noreps"
OUTPUTPATH="/Users/rtous/DockerVolume/partial2D/data/H36Mtest_original_norepsImages"
INPUTPATHIMAGES="UNKNOWN"
OVER_IMAGE=0
SCALE=1
BODY_MODEL="BodyModelOPENPOSE15"

python displayTest.py $INPUTPATHKEYPOINTS $OUTPUTPATH $INPUTPATHIMAGES $OVER_IMAGE $SCALE $BODY_MODEL


#Cropped test keypoints over white

INPUTPATHKEYPOINTS="/Users/rtous/DockerVolume/partial2D/data/H36Mtest"
OUTPUTPATH="/Users/rtous/DockerVolume/partial2D/data/H36MtestImages"
INPUTPATHIMAGES="UNKNOWN"
OVER_IMAGE=0
SCALE=1
BODY_MODEL="BodyModelOPENPOSE15"

python displayTest.py $INPUTPATHKEYPOINTS $OUTPUTPATH $INPUTPATHIMAGES $OVER_IMAGE $SCALE $BODY_MODEL

#Fixed test keypoints over white

INPUTPATHKEYPOINTS="/Users/rtous/DockerVolume/partial2D/data/output/H36M/TEST/keypoints"
OUTPUTPATH="/Users/rtous/DockerVolume/partial2D/data/output/H36M/TEST/imagesBlank"
INPUTPATHIMAGES="UNKNOWN"
OVER_IMAGE=0
SCALE=1
BODY_MODEL="BodyModelOPENPOSE15"

python displayTest.py $INPUTPATHKEYPOINTS $OUTPUTPATH $INPUTPATHIMAGES $OVER_IMAGE $SCALE $BODY_MODEL

