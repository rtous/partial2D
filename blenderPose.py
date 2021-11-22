#Apply HMR output to the smpl model loaded in blender

#NOTE: You need to select the Armature object first?

#HELP from https://chalmers.instructure.com/courses/7620/assignments/9230

import bpy
import cv2
import numpy as np
from mathutils import Matrix, Vector, Quaternion, Euler
from bpy import context
import math
import json
import os
import sys


### import Flashback library
import importlib.util
spec = importlib.util.spec_from_file_location("module.name", "/Users/rtous/DockerVolume/flashback/tools/poseUtils.py")
poseUtils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(poseUtils)
####

POSES_PATH="/Users/rtous/DockerVolume/partial2D/data/output/json"

bpy.ops.wm.open_mainfile(filepath="/Users/rtous/DockerVolume/flashback/3_3dmodel/info/SMPLX/models/library/7_smplx_SMPLXMESHANDRIG_generatedbyscript.blend")

#Need to select select the SMPL object and switch to EDIT mode
bpy.ops.object.mode_set(mode='OBJECT')
bpy.context.view_layer.objects.active = bpy.context.scene.objects['SMPL']
bpy.ops.object.mode_set(mode='EDIT')



#Rotate the entire body (-180 if you revreted a pose but not the rotation)
smpl = bpy.context.scene.objects['SMPL']
#smpl.rotation_mode = 'XYZ'
#smpl.rotation_euler = (Matrix.Rotation(np.radians(-180), 3, 'X') @ smpl.rotation_euler.to_matrix()).to_euler()
poseUtils.clearPose(smpl)

##############
################

cam = bpy.data.objects['Camera']
n = 0
for filename in sorted(os.listdir(POSES_PATH)):
    
    if filename != ".DS_Store":
        print("Processing "+filename)
         
        joints, global_orientation, cameraData = poseUtils.importPose(os.path.join(POSES_PATH, filename))
        
        if joints is not None:
            poseUtils.clearPose(smpl)    
            poseUtils.applyPose(smpl, joints)
            poseUtils.update_camera(cam, cameraData)
            
bpy.ops.wm.save_as_mainfile(filepath="/Users/rtous/DockerVolume/partial2D/data/output/render.blend")

