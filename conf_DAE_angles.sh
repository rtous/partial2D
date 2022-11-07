MODEL_NAME="DAE_angles"
MODEL="models_DAE_angles" #models models_mirror models_simple
NORMALIZATION="angles" #"center_scale", "basic", "none" 
KEYPOINT_RESTORATION=1
LEN_BUFFER_ORIGINALS=65536 #1000 65536
CROPPED_VARIATIONS=1 #1 (defalut) 0 to learn to copy
NZ=100 #100 #10 #0
DISCARDINCOMPLETEPOSES=1 #1
TRAINSPLIT=1 #0.8
PIXELLOSS_WEIGHT=1 #It's a DAE

#INFERENCE
MODELFILE="model_epoch1_batch4000.pt"




