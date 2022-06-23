"""Functions to visualize human poses"""

#FROM https://github.com/una-dinosauria/3d-pose-baseline

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import openPoseUtils

# Joints in H3.6M -- data has 32 joints, but only 17 that move; these are the indices.
H36M_NAMES = ['']*32
H36M_NAMES[0]  = 'Hip'
H36M_NAMES[1]  = 'RHip'
H36M_NAMES[2]  = 'RKnee'
H36M_NAMES[3]  = 'RFoot'
H36M_NAMES[6]  = 'LHip'
H36M_NAMES[7]  = 'LKnee'
H36M_NAMES[8]  = 'LFoot'
H36M_NAMES[12] = 'Spine'
H36M_NAMES[13] = 'Thorax'
H36M_NAMES[14] = 'Neck/Nose'
H36M_NAMES[15] = 'Head'
H36M_NAMES[17] = 'LShoulder'
H36M_NAMES[18] = 'LElbow'
H36M_NAMES[19] = 'LWrist'
H36M_NAMES[25] = 'RShoulder'
H36M_NAMES[26] = 'RElbow'
H36M_NAMES[27] = 'RWrist'

POSE_BODY_25_BODY_PARTS_DICT = {
    0:"Nose",
    1:"Neck",
    2:"RShoulder",
    3:"RElbow",
    4:"RWrist",
    5:"LShoulder",
    6:"LElbow",
    7:"LWrist",
    8:"MidHip",
    9:"RHip",
    10:"RKnee",
    11:"RAnkle",
    12:"LHip",
    13:"LKnee",
    14:"LAnkle",
    15:"REye",
    16:"LEye",
    17:"REar",
    18:"LEar",
    19:"LBigToe",
    20:"LSmallToe",
    21:"LHeel",
    22:"RBigToe",
    23:"RSmallToe",
    24:"RHeel",
    #25:"Background"
}

def show3Dpose(channels, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=False): # blue, orange
  """
  Visualize a 3d skeleton

  Args
    channels: 96x1 vector. The pose to plot.
    ax: matplotlib 3d axis to draw on
    lcolor: color for left part of the body
    rcolor: color for right part of the body
    add_labels: whether to add coordinate labels
  Returns
    Nothing. Draws on ax.
  """

  assert channels.size == len(H36M_NAMES)*3, "channels should have 96 entries, it has %d instead" % channels.size
  vals = np.reshape( channels, (len(H36M_NAMES), -1) )

  I   = np.array([1,2,3,1,7,8,1, 13,14,15,14,18,19,14,26,27])-1 # start points
  J   = np.array([2,3,4,7,8,9,13,14,15,16,18,19,20,26,27,28])-1 # end points
  LR  = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

  # Make connection matrix
  for i in np.arange( len(I) ):
    x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
    ax.plot(x, y, z, lw=2, c=lcolor if LR[i] else rcolor)

  RADIUS = 750 # space around the subject
  xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
  ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
  ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])
  ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])

  if add_labels:
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

  # Get rid of the ticks and tick labels
  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_zticks([])

  ax.get_xaxis().set_ticklabels([])
  ax.get_yaxis().set_ticklabels([])
  ax.set_zticklabels([])

  # Get rid of the panes (actually, make them white)
  white = (1.0, 1.0, 1.0, 0.0)
  ax.w_xaxis.set_pane_color(white)
  ax.w_yaxis.set_pane_color(white)
  # Keep z pane

  # Get rid of the lines in 3d
  ax.w_xaxis.line.set_color(white)
  ax.w_yaxis.line.set_color(white)
  ax.w_zaxis.line.set_color(white)

def show2DposeOPENPOSE(channels, ax):
  I  = np.array([1,1,1,2,3,5,6, 8, 9,10, 8,12,13, 1, 0,15, 0,16,14,19,14,11,22,11]) # start points
  J  = np.array([8,2,5,3,4,6,7, 9,10,11,12,13,14, 0,15,17,16,18,19,20,21,22,23,24]) # end points
  #0 = red (real right) 1 = blue (left)
  LR = np.array([0,0,1,0,0,1,1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

  show2Dpose(I, J, LR, POSE_BODY_25_BODY_PARTS_DICT, channels, ax)

def show2DposeH36M(channels, ax):
  I  = np.array([1,2,3,1,7,8,1, 13,14,14,18,19,14,26,27])-1 # start points
  J  = np.array([2,3,4,7,8,9,13,14,16,18,19,20,26,27,28])-1 # end points
  LR = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
  show2Dpose(I, J, LR, H36M_NAMES, channels, ax)

def show2Dpose(I, J, LR, DICT, channels, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=False):
  """Visualize a 2d skeleton

  Args
    channels: 64x1 vector. The pose to plot.
    ax: matplotlib axis to draw on
    lcolor: color for left part of the body
    rcolor: color for right part of the body
    add_labels: whether to add coordinate labels
  Returns
    Nothing. Draws on ax.
  """

  assert channels.size == len(DICT)*2, "channels should have %d entries, it has %d instead" % (len(DICT)*2, channels.size)
  vals = np.reshape( channels, (len(DICT), -1) )

  #I  = np.array([1,2,3,1,7,8,1, 13,14,14,18,19,14,26,27])-1 # start points
  #J  = np.array([2,3,4,7,8,9,13,14,16,18,19,20,26,27,28])-1 # end points
  #LR = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

  # Make connection matrix
  for i in np.arange( len(I) ):
    x, y = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(2)]
    if (x[0] != 0 and x[1] != 0 and y[0] != 0 and y[1] != 0):
      ax.plot(x, y, lw=1.5, c=lcolor if LR[i] else rcolor)

  # Get rid of the ticks
  ax.set_xticks([])
  ax.set_yticks([])

  # Get rid of tick labels
  ax.get_xaxis().set_ticklabels([])
  ax.get_yaxis().set_ticklabels([])

  #RADIUS = 350 # space around the subject
  RADIUS = 150 # space around the subject
 
  xroot, yroot = vals[0,0], vals[0,1]
  ax.set_xlim([-RADIUS+xroot, RADIUS+xroot])
  ax.set_ylim([-RADIUS+yroot, RADIUS+yroot])
  if add_labels:
    ax.set_xlabel("x")
    ax.set_ylabel("z")

  ax.set_aspect('equal')

def visualizeOne(keypoints, format, savePath):
  # Visualize random samples
  import matplotlib.gridspec as gridspec

  # 1080p = 1,920 x 1,080
  #fig = plt.figure( figsize=(19.2, 10.8) )
  fig = plt.figure( figsize=(5, 5) )

  #gs1 = gridspec.GridSpec(5, 9) # 5 rows, 9 columns
  gs1 = gridspec.GridSpec(1, 1) # 5 rows, 9 columns
  gs1.update(wspace=-0.00, hspace=0.05) # set the spacing between axes.
  plt.axis('off')
  subplot_idx, exidx = 1, 1
  ax1 = plt.subplot(gs1[subplot_idx-1])
  ax1.axis('off')
  #keypointsNumpy = np.array(keypoints)
  if format == "OPENPOSE":
    show2DposeOPENPOSE(keypoints, ax1 )
  elif format == "OPENPOSE15":
    show2DposeOPENPOSE15(keypoints, ax1 )
  elif format == "H36M":
    show2DposeH36M(keypoints, ax1 )
  else:
    print("ERROR: Unknown format ", format)
    sys.exit()
  ax1.invert_yaxis()
  if savePath is not None:
    plt.savefig(savePath)
  else:
    plt.show()
  





def visualizeMany(keypointsList, format):
  # Visualize random samples
  import matplotlib.gridspec as gridspec

  # 1080p = 1,920 x 1,080
  fig = plt.figure( figsize=(19.2, 10.8) )

  gs1 = gridspec.GridSpec(5, 9) # 5 rows, 9 columns
  gs1.update(wspace=-0.00, hspace=0.05) # set the spacing between axes.
  plt.axis('off')

  subplot_idx, exidx = 1, 1
  nsamples = 15
  for i in np.arange( nsamples ):

    # Plot 2d pose
    ax1 = plt.subplot(gs1[subplot_idx-1])
    #p2d = enc_in[exidx,:]
    #viz.show2Dpose( p2d, ax1 )
    if format == "OPENPOSE":
      show2DposeOPENPOSE(keypointsList[exidx], ax1 )
    elif format == "H36M":
      show2DposeH36M(keypointsList[exidx], ax1 )
    ax1.invert_yaxis()

    '''
    # Plot 3d gt
    ax2 = plt.subplot(gs1[subplot_idx], projection='3d')
    p3d = dec_out[exidx,:]
    viz.show3Dpose( p3d, ax2 )

    # Plot 3d predictions
    ax3 = plt.subplot(gs1[subplot_idx+1], projection='3d')
    p3d = poses3d[exidx,:]
    viz.show3Dpose( p3d, ax3, lcolor="#9b59b6", rcolor="#2ecc71" )
    '''
    exidx = exidx + 1
    subplot_idx = subplot_idx + 3

  plt.show()
