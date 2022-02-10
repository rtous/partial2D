# Human Pose Estimation in Movies with Conditional Generative Adversarial Networks


## Setup

The datasets I'm using are:

DATASET_CROPPED="data/H36M_ECCV18_HOLLYWOOD"

DATASET_ORIGINAL="/Volumes/ElementsDat/pose/H36M/ECCV2018/keyponts_generated_by_openpose_for_train_images_no_sufix"


## Data preparation

NOTE: size of directory: ls -f DIR | wc -l

1) ECCV18 Train images


2) openpose

   /Volumes/ElementsDat/pose/H36M/ECCV2018/keyponts_generated_by_openpose_for_train_images 
      images (35834)
      result (35834)

   (= keyponts_generated_by_openpose_for_train_images_no_sufix)

   dynamicData/H36M_ECCV18 (35834)

3) crop

   3.1) crop image and then openpose
      /Volumes/ElementsDat/pose/H36M/ECCV2018/keyponts_generated_by_openpose_for_train_images_cropped (71666)
      
      (not sure about this one!?)

   3.2) directly crop pose 

      data/H36M_ECCV18_HOLLYWOOD (315264) 

      =

       /Volumes/ElementsDat/pose/H36M/ECCV2018/H36M_ECCV18_HOLLYWOOD (315264)

4) filter


WARNING: Només 24794 orignals s'han usat

5) generate a test set

INPUTPATH="data/H36M_ECCV18_HOLLYWOOD"
OUTPUTPATH="dynamicData/H36M_ECCV18_HOLLYWOOD_test"
INPUTPATH_ORIGINAL="/Volumes/ElementsDat/pose/H36M/ECCV2018/keyponts_generated_by_openpose_for_train_images_no_sufix"
OUTPUTPATH_ORIGINAL="dynamicData/H36M_ECCV18_HOLLYWOOD_original_test"
MAX=1000

$ ./datasetECCV18_step3_test.sh


## Train

Testing:

 - CHARADE
 - TEST

## Monitoring training

tensorboard --logdir runs

http://localhost:6006/
