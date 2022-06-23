#./display.sh 

#Original openpose keypoints over white

INPUTPATHKEYPOINTS="/Users/rtous/DockerVolume/charade/results/openpose/2D_keypoints"
OUTPUTPATH="/Users/rtous/DockerVolume/charade/result/2D/blank_imagesOPENPOSE"
INPUTPATHIMAGES="/Users/rtous/DockerVolume/charade/input/images"
OVER_IMAGE=0

python display.py $INPUTPATHKEYPOINTS $OUTPUTPATH $INPUTPATHIMAGES $OVER_IMAGE

#Original openpose keypoints over image

INPUTPATHKEYPOINTS="/Users/rtous/DockerVolume/charade/results/openpose/2D_keypoints"
OUTPUTPATH="/Users/rtous/DockerVolume/charade/result/2D/imagesOPENPOSE"
INPUTPATHIMAGES="/Users/rtous/DockerVolume/charade/input/images"
OVER_IMAGE=1

python display.py $INPUTPATHKEYPOINTS $OUTPUTPATH $INPUTPATHIMAGES $OVER_IMAGE


#Fixed keypoints over white

INPUTPATHKEYPOINTS="/Users/rtous/DockerVolume/partial2D/data/output/H36M/CHARADE/keypoints"
OUTPUTPATH="/Users/rtous/DockerVolume/partial2D/data/output/H36M/CHARADE/imagesBlank"
INPUTPATHIMAGES="/Users/rtous/DockerVolume/charade/input/images"
OVER_IMAGE=0

python display.py $INPUTPATHKEYPOINTS $OUTPUTPATH $INPUTPATHIMAGES $OVER_IMAGE

#Fixed keypoints over image

INPUTPATHKEYPOINTS="/Users/rtous/DockerVolume/partial2D/data/output/H36M/CHARADE/keypoints"
OUTPUTPATH="/Users/rtous/DockerVolume/partial2D/data/output/H36M/CHARADE/images"
INPUTPATHIMAGES="/Users/rtous/DockerVolume/charade/input/images"
OVER_IMAGE=1

python display.py $INPUTPATHKEYPOINTS $OUTPUTPATH $INPUTPATHIMAGES $OVER_IMAGE


