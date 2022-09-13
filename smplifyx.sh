#INPUTPATH="/Users/rtous/DockerVolume/charade/results/H36M_autoencoder"
#OUTPUTPATH="/Users/rtous/DockerVolume/charade/results/H36M_autoencoder/3D"
INPUTPATH="/Users/rtous/DockerVolume/partial2D/data/output/ECCV18OP_FINAL/CHARADE"
OUTPUTPATH="/Users/rtous/DockerVolume/partial2D/data/output/ECCV18OP_FINAL/CHARADE/3D"



#mkdir -p /Users/rtous/DockerVolume/charade/results/H36M_autoencoder/keypoints
#mkdir -p /Users/rtous/DockerVolume/charade/results/H36M_autoencoder/images

#cp /Users/rtous/DockerVolume/partial2D/data/output/H36M/CHARADE/keypoints/*.json /Users/rtous/DockerVolume/charade/results/H36M_autoencoder/keypoints 
#cp /Users/rtous/DockerVolume/charade/input/images/*.png /Users/rtous/DockerVolume/charade/results/H36M_autoencoder/images

cd $HOME/DockerVolume/flashback_smplify-x

source venv/bin/activate 

# 1) Infer the pose
python smplifyx/main.py --config cfg_files/fit_smplx.yaml \
    --data_folder $INPUTPATH \
    --use_cuda="False" \
    --output_folder $OUTPUTPATH \
    --visualize="True" \
    --model_folder models \
    --vposer_ckpt models/vposer_v1_0 \
    --part_segm_fn smplx_parts_segm.pkl

# 2) .pkl -> .json
python smplifyx/smplifyx2smplxRUBEN_v4_manyMeshes.py --config cfg_files/fit_smplx.yaml \
    --data_folder $OUTPUTPATH \
    --output_folder $OUTPUTPATH/poses \
    --visualize="False" \
    --model_folder models \
    --vposer_ckpt models/vposer_v1_0 \
    --part_segm_fn smplx_parts_segm.pkl

cd /Users/rtous/DockerVolume/partial2D
