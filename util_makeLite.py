import pathlib
import os
from shutil import copyfile
from os.path import isfile, join, splitext
import sys


argv = sys.argv
try:
    INPUTPATH=argv[1]
    OUTPUTPATH=argv[2]
    INPUTPATH_ORIGINAL=argv[3]
    OUTPUTPATH_ORIGINAL=argv[4]
    MAX=int(argv[5])

except ValueError:
    print("Wrong arguments. Expecting two paths.")

'''
INPUTPATH = "/Volumes/ElementsDat/pose/COCO/ruben_structure/keypoints_openpose_format_cropped"
OUTPUTPATH = "/Volumes/ElementsDat/pose/COCO/ruben_structure/keypoints_openpose_format_cropped_lite"

INPUTPATH_ORIGINAL = "/Volumes/ElementsDat/pose/COCO/ruben_structure/keypoints_openpose_format"
OUTPUTPATH_ORIGINAL = "/Volumes/ElementsDat/pose/COCO/ruben_structure/keypoints_openpose_format_lite"

MAX = 1024
'''

pathlib.Path(OUTPUTPATH).mkdir(parents=True, exist_ok=True) 
pathlib.Path(OUTPUTPATH_ORIGINAL).mkdir(parents=True, exist_ok=True) 

scandirIterator = os.scandir(INPUTPATH)

n = 0
print("Reading files from "+INPUTPATH)
for file in scandirIterator:
	copyfile(join(INPUTPATH, str(file.name)), join(OUTPUTPATH, str(file.name)))
	indexUnderscore = file.name.find('_')
	filenameoriginal = file.name[:indexUnderscore]+".json"#+"_keypoints.json" 	
	filenameoriginalpath = join(OUTPUTPATH_ORIGINAL, filenameoriginal)
	if not os.path.isfile(filenameoriginalpath):
		copyfile(join(INPUTPATH_ORIGINAL, filenameoriginal), join(OUTPUTPATH_ORIGINAL, filenameoriginal))
	n += 1
	if n >= MAX:
		break
scandirIterator.close()