# Human pose completion in partial body camera shots

## 1. Introduction

This repository contains the code related to the paper ["Human pose completion in partial body camera shots"](https://upcommons.upc.edu/bitstream/handle/2117/394207/main.pdf;jsessionid=F7BEA81F9053C26DE28BE39BCAD8FAF5?sequence=1)

## 2 Acknowledgements

If you find this repository useful for your research, please cite the original publication:

   @article{Tous28072023},
      author = {Ruben Tous, Jordi Nin and Laura Igual},
      title = {Human pose completion in partial body camera shots},
      journal = {Journal of Experimental \& Theoretical Artificial Intelligence},
      volume = {0},
      number = {0},
      pages = {1--11},
      year = {2023},
      publisher = {Taylor \& Francis},
      doi = {10.1080/0952813X.2023.2241575},
      URL = { https://doi.org/10.1080/0952813X.2023.2241575}
   }

## Setup

git clone https://github.com/rtous/partial2D.git
cd partial2D

python3.11 -m venv myvenv
source myvenv/bin/activate

pip install torch==2.3.1 torchvision==0.18.1
pip install matplotlib==3.9.2
pip install opencv-python==3.4.17.61
pip install numpy==1.26.0
pip install tensorboard==2.18.0
pip install cdflib==1.3.2
pip install scipy==1.14.1

## Dataset

## Test

./datasetH36M_makelite.sh
./train.sh conf_DAE.sh confDataset_H36M.sh 0 
./inference.sh conf_DAE.sh confDataset_H36M.sh 0
./FPD.sh conf_DAE.sh confDataset_H36M.sh 0







pip install gdown
mkdir weights
cd weights
gdown https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ




Dataset:
https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ

##################

#install other libraries
pip install matplotlib==3.9.2
pip install opencv-python==3.4.17.61
pip install numpy==1.26.0



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
