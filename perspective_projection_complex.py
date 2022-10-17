#from https://stackoverflow.com/questions/67517809/mapping-3d-vertex-to-pixel-using-pyreder-pyglet-opengl

from numpy.linalg import inv
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pyrender
import math

#coordinate systems: https://medium.com/comerge/what-are-the-coordinates-225f1ec0dd78
#blender is right handed (thumb is x, index is y, middle is z) shown with z up
#unity is left handed (thumb is x, index is y, middle is z) shown with y up
#pyrender camera:
#the camera z-axis points away from the scene, 
#the x-axis points right in image space, 
#and the y-axis points up in image space.
#it's like openGL

#projective (pinhole) model of a camera.: https://www.imatest.com/support/docs/pre-5-2/geometric-calibration-deprecated/projective-camera/
#optical axis:
#focal pane: where the 2D is projected
#5 parameters: focal length in the x and y directions, principle point in the x and y directions, and skew between the x and y directions.
#pinhole to the focal plane along the optical axis
#the principle point (pp) which is the pixel coordinate of the intersection of the optical axis with the focal plane. 
#intrinsic matrix is an upper-triangular matrix that transforms a world coordinate relative to the camera into a homogeneous image coordinate
'''
K=
fx 0  ppx
0  fy ppy
0  0  1
'''
#homogeneous image point (vertical vector of three) = K * 3D point (three values)
#typically the 3dpoint is transposed to become vertical


#here
#https://stackoverflow.com/questions/724219/how-to-convert-a-3d-point-into-2d-perspective-projection
#homogenous coordinates are explained

focal_pane_width=10
focal_pane_height=10

def map_to_pixel(point3d,w,h,projection,view):
    p=projection@inv(view)@point3d.T
    p=p/p[3]
    p[0]=(w/2*p[0]+w/2)    #tranformation from [-1,1] ->[0,width]
    p[1]=h-(h/2*p[1]+h/2)  #tranformation from [-1,1] ->[0,height] (top-left image)
    return p

#yfov (float) – The floating-point vertical field of view in radians.
#• znear (float) – The floating-point distance to the near clipping plane. If not specified, defaults to 0.05.
#• zfar (float, optional) – The floating-point distance to the far clipping plane. zfar must be greater than znear. If None, the camera uses an infinite projection matrix.
#aspectRatio (float, optional) – The floating-point aspect ratio of the field of view. 
#If not specified, the camera uses the viewport’s aspect ratio.
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, znear=0.05, zfar=None, aspectRatio=1.0)


#trimesh camera_transform is world to camera transformation matrix (or reverse, I can't tell to be honest, try inverse if it doesn't work)


projection = camera.get_projection_matrix([10, 10])


view=       [[ 0.96592583, -0.0669873 ,  0.25     ,  3.        ],
                   [ 0.        ,  0.96592583,  0.25881905,  4.        ],
                   [-0.25881905, -0.25      ,  0.9330127 , 10.        ],
                   [ 0.        ,  0.        ,  0.        ,  1.        ]]

'''
projection= [[ 2.4142135,  0.        ,  0.        ,  0.        ],
                   [ 0.        ,  2.41421356,  0.        ,  0.        ],
                   [ 0.        ,  0.        , -1.0010005 , -0.10005003],
                   [ 0.        ,  0.        , -1.        ,  0.        ]]

'''
pixel_plot = plt.figure()
plt.rcParams["figure.figsize"] = [10, 10]
#plt.plot(x, y, marker="o", markersize=20, markeredgecolor="red", markerfacecolor="green")
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.grid()

#A cube
points3dhomogeneous = np.array([[1,1,1,1], [1,1,3,1], [1,3,1,1], [1,3,3,1], [3,1,1,1], [3,1,3,1], [3,3,1,1], [3,3,3,1]])
#points3d = np.array([[0,0,0], [0,0,3], [0,3,0], [0,3,3], [3,0,0], [3,0,3], [3,3,0], [3,3,3]])

focal_distance = 0.4#1
scalex = 1
scaley = 1
offsetx = 1
offsety = 1
angle = 1
for point3d in points3d:
	pixel = map_to_pixel(points3dhomogeneous,10,10,projection,view )
	plt.plot(pixel[0],pixel[1], marker='o', markersize=5, markeredgecolor="red", markerfacecolor="red") 

plt.show()
