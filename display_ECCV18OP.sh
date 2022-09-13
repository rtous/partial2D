#Fixed keypoints over white

INPUTPATHKEYPOINTS="/Users/rtous/DockerVolume/partial2D/data/output/ECCV18OP/CHARADE/keypoints"
OUTPUTPATH="/Users/rtous/DockerVolume/partial2D/data/output/ECCV18OP/CHARADE/imagesBlank"
INPUTPATHIMAGES="/Users/rtous/DockerVolume/charade/input/images"
OVER_IMAGE=0
SCALE=1

python display.py $INPUTPATHKEYPOINTS $OUTPUTPATH $INPUTPATHIMAGES $OVER_IMAGE $SCALE

#Fixed keypoints over image

INPUTPATHKEYPOINTS="/Users/rtous/DockerVolume/partial2D/data/output/ECCV18OP/CHARADE/keypoints"
OUTPUTPATH="/Users/rtous/DockerVolume/partial2D/data/output/ECCV18OP/CHARADE/images"
INPUTPATHIMAGES="/Users/rtous/DockerVolume/charade/input/images"
OVER_IMAGE=1
SCALE=0

python display.py $INPUTPATHKEYPOINTS $OUTPUTPATH $INPUTPATHIMAGES $OVER_IMAGE $SCALE


