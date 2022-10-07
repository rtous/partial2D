MODEL_NAME="DAE_heatmaps"
MODEL="models_DAE_heatmaps" #models models_mirror models_simple
NORMALIZATION="heatmaps" #"center_scale", "basic", "none" 
KEYPOINT_RESTORATION=1
LEN_BUFFER_ORIGINALS=256 #1000 65536
CROPPED_VARIATIONS=1 #1 (defalut) 0 to learn to copy
NZ=0 #100 #10 #0
DISCARDINCOMPLETEPOSES=1 #1
TRAINSPLIT=1 #0.8
PIXELLOSS_WEIGHT=1 #It's a DAE

#INFERENCE
MODELFILE="model_epoch0_batch0.pt"



