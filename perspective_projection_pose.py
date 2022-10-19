'''
- 3D to 2D perspective projection
- https://towardsdatascience.com/camera-extrinsic-matrix-with-example-in-python-cfe80acab8dd
'''	


from numpy.linalg import inv
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#import pyrender
import math
import perspective_projection_extrinsics as pp

def angleTwoVectors(vector1, vector2):
    #θ = cos-1 [ (a · b) / (|a| |b|) ]
    unit_vector1 = vector1 / np.linalg.norm(vector1)
    unit_vector2 = vector2 / np.linalg.norm(vector2)
    angle = np.arccos(np.dot(unit_vector1, unit_vector2)) #angle in radian
    return angle

def rotationBoneAWithRespectToBoneB(vector1, vector2):
    """
    Computes the rotation that would enable to align boneA to boneB.
    Works on the global axis (the coordinates are global).
    The resulting axis+angle rotation cannot be applied directly in pose mode
    because in pose mode works with local axis
    """
    rot_angle =  angleTwoVectors(vector1, vector2)
    rot_axis = np.cross(vector1, vector2)
    return rot_axis, rot_angle

def rotation_matrix(axis, angle):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(angle / 2.0)
    b, c, d = -axis * math.sin(angle / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def rotateVectorAxisangle(vector, rot_angle, rot_axis):
    #assumes unit vector I guess
    mrot_aux = rotation_matrix(rot_axis, rot_angle)
    rotatedVector = np.dot(mrot_aux, vector)
    return rotatedVector

def axisangleCompact(rot_angle, rot_axis):
    #divide vector by its norm (lenght)
    rot_axis_unit = rot_axis / np.linalg.norm(rot_axis)

    #and multiply for the angle
    rot_axis_with_angle_as_magnitude = rot_axis_unit * rot_angle
    return rot_axis_with_angle_as_magnitude

#Matplotlib 10x10 image
pixel_plot, ax = plt.subplots()
plt.rcParams["figure.figsize"] = [10, 10]
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.grid()

#A cube
points3d = np.array([[1,1,1], [1,5,1], [5,5,1], [5,1,1],  #fron square
	                 [1,1,5], [1,5,5], [5,5,5], [5,1,5]]) #back square

#Lines between the 3d points
lines = np.array([[0,1],[1,2],[2,3],[3,0], #fron square
	             [4,5],[5,6],[6,7],[7,4], #back square
	             [0,4],[1,5],[2,6],[3,7]]
	             )

#3D pose -> lenghts
for l, line in enumerate(lines):
    from_3dpoint = points3d[line[0]]
    to_3dpoint = points3d[line[1]]
    squared_dist = np.sum((from_3dpoint-to_3dpoint)**2, axis=0)
    dist = np.sqrt(squared_dist)
    print("dist=", dist)

#3D pose -> angles

#we first compute the two vectors
#they share a point but this does not matter to compute their angle (will be useful to reconstruct)

#line0
from_3dpoint1 = points3d[lines[0][0]]
to_3dpoint1 = points3d[lines[0][1]]
vector1 = to_3dpoint1-from_3dpoint1
print(from_3dpoint1)
print(to_3dpoint1)
print(vector1)

#line1
from_3dpoint2 = points3d[lines[1][0]]
to_3dpoint2 = points3d[lines[1][1]]
vector2 = to_3dpoint2-from_3dpoint2
print(from_3dpoint2)
print(to_3dpoint2)
print(vector2)

rot_axis, rot_angle = rotationBoneAWithRespectToBoneB(vector1, vector2)
print("rot_axis=", rot_axis)
print("rot_angle=", rot_angle)

compact_axis_angle = axisangleCompact(rot_angle, rot_axis)
print("compact_axis_angle=", compact_axis_angle)





#Apply camera pose (extrinsics)
rotation_angles = [np.pi/4]
rotation_order = 'z'#'y'
translation_offset = np.array([7, 2, 0])
points3d_cam = pp.points3d_relative_to_camera(points3d, rotation_angles, rotation_order, translation_offset)

#Apply camera intrinsics
focal_pane_width=10
focal_pane_height=10
focal_distance = 0.8#1
scalex = 1
scaley = 1
ppx=focal_pane_width/2
ppy=focal_pane_height/2
points2d = pp.points3d_cam_to_pixels(points3d_cam, focal_distance, ppx, ppy, scalex, scaley)

#Show points 2d
for i, pixel in enumerate(points2d):
	plt.plot(pixel[0],pixel[1], marker='o', markersize=5, markeredgecolor="red", markerfacecolor="red") 
	ax.annotate(str(i), (pixel[0], pixel[1]))
	print("point3d("+str(i)+"):"+str(pixel[0])+","+str(pixel[1]))

#Show lines
for l, line in enumerate(lines):
	from_pixel = points2d[line[0]]
	to_pixel = points2d[line[1]]
	plt.plot([from_pixel[0],to_pixel[0]], [from_pixel[1],to_pixel[1]], linewidth=1, markersize=0)

#plt.show()
plt.savefig('perspective_projection_extrinsics.png')

