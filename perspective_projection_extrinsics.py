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

def homogenize(cartesian_vector: np.array):
    #Convert a vector from cartesian coordinates into homogenious coordinates.
    shape = cartesian_vector.shape
    homogenious_vector = np.ones((*shape[:-1], shape[-1] + 1))
    homogenious_vector[..., :-1] = cartesian_vector
    return homogenious_vector

def get_rot_x(angle):
    '''
    transformation matrix that rotates a point about the standard X axis
    '''
    Rx = np.zeros(shape=(3, 3))
    Rx[0, 0] = 1
    Rx[1, 1] = np.cos(angle)
    Rx[1, 2] = -np.sin(angle)
    Rx[2, 1] = np.sin(angle)
    Rx[2, 2] = np.cos(angle)
    
    return Rx

def get_rot_y(angle):
    '''
    transformation matrix that rotates a point about the standard Y axis
    '''
    Ry = np.zeros(shape=(3, 3))
    Ry[0, 0] = np.cos(angle)
    Ry[0, 2] = -np.sin(angle)
    Ry[2, 0] = np.sin(angle)
    Ry[2, 2] = np.cos(angle)
    Ry[1, 1] = 1
    
    return Ry

def get_rot_z(angle):
    '''
    transformation matrix that rotates a point about the standard Z axis
    '''
    Rz = np.zeros(shape=(3, 3))
    Rz[0, 0] = np.cos(angle)
    Rz[0, 1] = -np.sin(angle)
    Rz[1, 0] = np.sin(angle)
    Rz[1, 1] = np.cos(angle)
    Rz[2, 2] = 1
    
    return Rz

def rotation_matrix(angles, order):
    '''
    Create a matrix that rotates a vector through the given angles in the given order
    wrt the standard global axes (extrinsic rotation)
    Note: The rotation is carried out anti-clockwise in a left handed axial system
    E.g.: rotation_angles = [np.pi/4]
	E.g.: rotation_order = 'z'#'y'
    '''
    fn_mapping = {'x': get_rot_x, 'y': get_rot_y, 'z': get_rot_z}
    net = np.identity(3)
    for angle, axis in list(zip(angles, order))[::-1]:
        if fn_mapping.get(axis) is None:
            raise ValueError("Invalid axis")
        R = fn_mapping.get(axis)
        net = np.matmul(net, R(angle))
        
    return net

def translation_matrix(offset):
    #Create a transformation matrix that translates a vetor by the given offset
    #E.g.: offset = np.array([7, 2, 0])
    T = np.identity(4)
    T[:3, 3] = offset
    return T

def point3d_relative_to_camera(point3d, rotation_angles, rotation_order, translation_offset):
	#ONE point3d in world coords to coords relative to camera 
	# create rotation matrix
	R = rotation_matrix(rotation_angles, rotation_order)
	R_ = np.identity(4)
	R_[:3, :3] = R

	# create translation transformation matrix
	T_ = translation_matrix(translation_offset)
	E = np.linalg.inv(R_ @ T_)
	# remove last row of E
	E = E[:-1, :]
	
	#apply camera extrinsics to world coords to obtain camera relative coords
	points3d_cam = E @ homogenize(point3d)
	return points3d_cam

def points3d_relative_to_camera(points3d, rotation_angles, rotation_order, translation_offset):
	#for MANY points
	points3d_cam = np.zeros((points3d.shape[0], 3))
	for i, point3d in enumerate(points3d):
		#apply camera extrinsics (pose)
		point3d_cam = point3d_relative_to_camera(point3d, rotation_angles, rotation_order, translation_offset)
		points3d_cam[i]=point3d_cam
	return points3d_cam

def point3d_relative_to_cam_to_2d(point3d,focal_distance, ppx, ppy, scale_x, scale_y):
	#Apply camera intrinsics to ONE point
	fx = focal_distance*scale_x
	fy = focal_distance*scale_y
	camera_intrinsic_matrix = [[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]]
	homogeneous_point = camera_intrinsic_matrix @ point3d.T
	inhomogeneous_point = [homogeneous_point[0] / homogeneous_point[2], homogeneous_point[1] / homogeneous_point[2]]
	return inhomogeneous_point

def points3d_cam_to_pixels(points3d_cam, focal_distance, ppx, ppy, scale_x, scale_y):
	#Apply camera intrinsics to MANY points
	points2d = np.zeros((points3d.shape[0], 2))
	for i, point3d_cam in enumerate(points3d_cam):
		#apply camera extrinsics (pose)
		point2d = point3d_relative_to_cam_to_2d(point3d_cam, focal_distance, ppx, ppy, scale_x, scale_y)
		points2d[i]=point2d
	return points2d

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

#Apply camera pose (extrinsics)
rotation_angles = [np.pi/4]
rotation_order = 'z'#'y'
translation_offset = np.array([7, 2, 0])
points3d_cam = points3d_relative_to_camera(points3d, rotation_angles, rotation_order, translation_offset)

#Apply camera intrinsics
focal_pane_width=10
focal_pane_height=10
focal_distance = 0.8#1
scalex = 1
scaley = 1
ppx=focal_pane_width/2
ppy=focal_pane_height/2
points2d = points3d_cam_to_pixels(points3d_cam, focal_distance, ppx, ppy, scalex, scaley)

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

