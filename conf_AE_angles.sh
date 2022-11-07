MODEL_NAME="AE_ANGLES"
MODEL="models_AE_angles" #models models_mirror models_simple models_AE
NZ=32 #100 #10 #0


PIXELLOSS_WEIGHT=1 #It's a AE

#DATA SPECIFICS
NORMALIZATION="angles" #"center_scale", "basic", "none" 
#LEN_BUFFER_ORIGINALS=1000 #1000 65536
CROPPED_VARIATIONS=0 #1 (defalut) 0 to learn to copy
DISCARDINCOMPLETEPOSES=1 #1
TRAINSPLIT=1 #0.8
KEYPOINT_RESTORATION=0

#INFERENCE
MODELFILE="model_epoch10_batch0.pt"


