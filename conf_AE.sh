MODEL_NAME="AE"
MODEL="models_AE" #models models_mirror models_simple models_AE
NORMALIZATION="center_scale" #"center_scale", "basic", "none" 
KEYPOINT_RESTORATION=0
LEN_BUFFER_ORIGINALS=65536 #1000 65536
CROPPED_VARIATIONS=0 #1 (defalut) 0 to learn to copy
NZ=32 #100 #100 #10 #0
DISCARDINCOMPLETEPOSES=1 #1
TRAINSPLIT=1 #0.8
PIXELLOSS_WEIGHT=1 #It's a AE

#INFERENCE
MODELFILE="model_epoch2_batch0.pt"


