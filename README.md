# Human pose completion in partial body camera shots (CompletePose)

## 1. Introduction

This repository contains the code related to the paper ["Human pose completion in partial body camera shots"](https://upcommons.upc.edu/bitstream/handle/2117/394207/main.pdf;jsessionid=F7BEA81F9053C26DE28BE39BCAD8FAF5?sequence=1)

## 2 Acknowledgements

If you find this repository useful for your research, please cite the original publication:
```
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
```

## Setup
```
git clone https://github.com/rtous/partial2D.git
cd partial2D

python3.11 -m venv myvenv
source myvenv/bin/activate

#pip install torch==2.3.1 torchvision==0.18.1
pip install torch==2.0.1 torchvision==0.15.2 
pip install matplotlib==3.9.2
pip install opencv-python==3.4.17.61
pip install numpy==1.26.0
pip install tensorboard==2.18.0
pip install cdflib==1.3.2
pip install scipy==1.14.1
```
## Dataset

### H36M (training dataset and used for quantitative evaluation)

Download H36M 2D poses:

   - Request access to https://vision.imar.ro/human3.6m
   - Download by subject (for all of them): Poses_D2_Positions
   - Decompress the files into a data/H36M folder (asuming you're in the repo's root)
```
      mkdir -p data/H36M/2D
      mv $HOME/Downloads/Poses* data/H36M/2D
      tar -xvf data/H36M/Poses_D2_Positions_S1.tgz -C data/H36M/2D
      tar -xvf data/H36M/Poses_D2_Positions_S5.tgz -C data/H36M/2D
      tar -xvf data/H36M/Poses_D2_Positions_S6.tgz -C data/H36M/2D
      tar -xvf data/H36M/Poses_D2_Positions_S7.tgz -C data/H36M/2D
      tar -xvf data/H36M/Poses_D2_Positions_S8.tgz -C data/H36M/2D
      tar -xvf data/H36M/Poses_D2_Positions_S9.tgz -C data/H36M/2D
      tar -xvf data/H36M/Poses_D2_Positions_S11.tgz -C data/H36M/2D
      rm data/H36M/2D/*.tgz
```

### CHARADE (test dataset for qualitative evaluation)

```
      mkdir -p data/CHARADE/keypoints
      wget https://github.com/rtous/charade/raw/refs/heads/main/keypointsOpenPose.zip
      unzip keypointsOpenPose.zip -d data/CHARADE/keypoints  
      rm keypointsOpenPose.zip
      mkdir -p data/CHARADE/images
      wget https://github.com/rtous/charade/raw/refs/heads/main/images.zip
      unzip images.zip -d data/CHARADE/images  
      rm images.zip
```


git clone ../https://github.com/rtous/charade
unzip ../keypointsOpenPose.zip -d data

## Test

```
   ./datasetH36M_makelite.sh
   ./train.sh conf_GAN.sh confDataset_H36M.sh 0 
   ./inference.sh conf_GAN.sh confDataset_H36M.sh 0
   ./FPD.sh conf_GAN.sh confDataset_H36M.sh 0
```

```
   ./datasetH36M_makelite.sh
   ./train.sh conf_DAE.sh confDataset_H36M.sh 0 
   ./inference.sh conf_DAE.sh confDataset_H36M.sh 0
   ./FPD.sh conf_DAE.sh confDataset_H36M.sh 0
```


## Monitoring training

tensorboard --logdir runs

http://localhost:6006/
