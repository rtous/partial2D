'''
- Simple 3D to 2D perspective projection
- Pinhole model from https://www.imatest.com/support/docs/pre-5-2/geometric-calibration-deprecated/projective-camera/
- no skew, etc.
'''	


from numpy.linalg import inv
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pyrender
import math


#projective (pinhole) model of a camera.: 
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

focal_pane_width=10
focal_pane_height=10

def map_to_pixel(point3d,focal_distance, ppx, ppy, scale_x, scale_y):
	fx = focal_distance*scale_x
	fy = focal_distance*scale_y
	camera_intrinsic_matrix = [[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]]
	homogeneous_point = camera_intrinsic_matrix @ point3d.T
	inhomogeneous_point = [homogeneous_point[0] / homogeneous_point[2], homogeneous_point[1] / homogeneous_point[2]]
	return inhomogeneous_point

pixel_plot, ax = plt.subplots()
#pixel_plot = plt.figure()
plt.rcParams["figure.figsize"] = [10, 10]
#plt.plot(x, y, marker="o", markersize=20, markeredgecolor="red", markerfacecolor="green")
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.grid()

#A cube
#points3d = np.array([[0,0,0], [0,0,3], [0,3,0], [0,3,3], [3,0,0], [3,0,3], [3,3,0], [3,3,3]])
points3d = np.array([[1,1,1], [1,5,1], [5,5,1], [5,1,1],  #fron square
	                 [1,1,5], [1,5,5], [5,5,5], [5,1,5]]) #back square

lines = np.array([[0,1],[1,2],[2,3],[3,0], #fron square
	             [4,5],[5,6],[6,7],[7,4], #back square
	             [1,5], [3,7]]
	             )

focal_distance = 0.8#1
scalex = 1
scaley = 1

for i, point3d in enumerate(points3d):
	pixel = map_to_pixel(point3d, focal_distance,focal_pane_width/2,focal_pane_height/2, scalex, scaley)
	plt.plot(pixel[0],pixel[1], marker='o', markersize=5, markeredgecolor="red", markerfacecolor="red") 
	ax.annotate(str(i), (pixel[0], pixel[1]))

for line in lines:
	from_point3d = points3d[line[0]]
	print("from_point3d:", from_point3d)
	to_point3d = points3d[line[1]]
	print("to_point3d:", to_point3d)
	from_pixel = map_to_pixel(from_point3d, focal_distance,focal_pane_width/2,focal_pane_height/2, scalex, scaley)
	to_pixel = map_to_pixel(to_point3d, focal_distance,focal_pane_width/2,focal_pane_height/2, scalex, scaley)
	#plt.plot(pixel[0],pixel[1], marker='o', markersize=5, markeredgecolor="red", markerfacecolor="red") 
	print("line from "+str(line[0])+" to "+str(line[1]))
	plt.plot(from_pixel, to_pixel, 'go--', linewidth=2, markersize=0)

plt.show()
