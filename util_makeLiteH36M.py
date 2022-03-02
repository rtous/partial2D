import pathlib
import os
from shutil import copyfile
from os.path import isfile, join, splitext
import sys
import h36mIterator
import openPoseUtils
import random


argv = sys.argv
try:
    OUTPUTPATH=argv[1]
    OUTPUTPATH_ORIGINAL=argv[2]
    MAX=int(argv[3])
except ValueError:
    print("Wrong arguments. Expecting two paths.")


pathlib.Path(OUTPUTPATH).mkdir(parents=True, exist_ok=True) 
pathlib.Path(OUTPUTPATH_ORIGINAL).mkdir(parents=True, exist_ok=True) 
pathlib.Path(OUTPUTPATH_ORIGINAL+"_noreps").mkdir(parents=True, exist_ok=True) 

i = 0
i_originals = 0
buffer_originals = []
buffer_variations = []
originals_idx = []

scandirIterator = h36mIterator.iterator()
for keypoints_original in scandirIterator:
    #We fill first a buffer of originals
    buffer_originals.append(keypoints_original)
    len_buffer_originals = len(buffer_originals)
    if len_buffer_originals == 65536:#65536:
        #Once the buffer is filled, we shuffle and obtain variations
        #for all the originals till obtaining MAX items
        print("shuffle buffer original full: ", len_buffer_originals)
        print("sorting buffer originals...")
        random.shuffle(buffer_originals)
        print("generating variations...")
        for o_idx, buffered_keypoints_original in enumerate(buffer_originals):
            variations = openPoseUtils.crop(buffered_keypoints_original)               
            for v_idx, keypoints_cropped in enumerate(variations):    
                buffer_variations.append((buffered_keypoints_original, keypoints_cropped, o_idx))
        
        len_variations = len(buffer_variations)
        print("Variations generated: ", len_variations)
        print("Shuffling variations...")
        random.shuffle(buffer_variations)
        print("Saving variations...")
        for tup in buffer_variations:
            if i >= MAX:
            	sys.exit()
            keypoints_original = tup[0]
            keypoints_cropped = tup[1]
            original_idx = tup[2]
            if original_idx not in originals_idx:
                openPoseUtils.keypoints2json(buffer_originals[original_idx], join(OUTPUTPATH_ORIGINAL+"_noreps", str(i_originals)+".json"))
                originals_idx.append(original_idx)
                i_originals += 1
            #print(keypoints_original)
            openPoseUtils.keypoints2json(keypoints_original, join(OUTPUTPATH_ORIGINAL, str(i)+".json"))
            openPoseUtils.keypoints2json(keypoints_cropped, join(OUTPUTPATH, str(i)+"_keypoints.json"))
            i += 1
        print("Not enough items in the buffer, filling the buffer again...")

        buffer_variations = []
        buffer_originals = [] 
        originals_idx = []

scandirIterator.close()
print("Closed h36mIterator.")