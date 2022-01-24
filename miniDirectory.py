import pathlib
import os
from shutil import copyfile
from os.path import isfile, join, splitext



INPUTPATH = "/Volumes/ElementsDat/pose/COCO/ruben_structure/keypoints_openpose_format_cropped"
OUTPUTPATH = "/Volumes/ElementsDat/pose/COCO/ruben_structure/keypoints_openpose_format_cropped_lite"

INPUTPATH_ORIGINAL = "/Volumes/ElementsDat/pose/COCO/ruben_structure/keypoints_openpose_format"
OUTPUTPATH_ORIGINAL = "/Volumes/ElementsDat/pose/COCO/ruben_structure/keypoints_openpose_format_lite"

MAX = 1024

pathlib.Path(OUTPUTPATH).mkdir(parents=True, exist_ok=True) 
pathlib.Path(OUTPUTPATH_ORIGINAL).mkdir(parents=True, exist_ok=True) 

scandirIterator = os.scandir(INPUTPATH)

n = 0
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