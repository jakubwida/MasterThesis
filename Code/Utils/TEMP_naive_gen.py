import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches


fig_disk_num = 4
fig_radiuses = np.array([0.125,0.125,0.125,0.125])
fig_distances = np.array([0.125*3,0.125,0.125,0.125*3])
fig_angles = np.array([np.pi,np.pi,0.0,0.0])


"""
fig_disk_num = 1
fig_radiuses = np.array([0.5])
fig_distances = np.array([0.0])
fig_angles = np.array([0.0])
"""

"""
fig_disk_num = 2
fig_radiuses = np.array([0.25,0.25])
fig_distances = np.array([0.25,0.25])
fig_angles = np.array([0.0,np.pi])
"""

"""
fig_disk_num = 3
fig_radiuses = np.array([0.125,0.25,0.125])
fig_distances = np.array([0.125*3,0.0,0.125*3])
fig_angles = np.array([np.pi,0.0,0.0])
"""

voxel_size = 1.0
voxel_angular_size = np.pi * 2.0

world_size= np.array([3.0,3.0])

#==========helpers================#

def figure_to_xy(figure):
	rotated_angles = fig_angles + figure[2]
	xs = fig_distances * np.cos(rotated_angles) + figure[0]
	ys = fig_distances * np.sin(rotated_angles) + figure[1]
	return np.column_stack((xs,ys))


def draw(voxels,figures):
	plt.figure(figsize=[12,12])
	ax = plt.gca()
	ax.set_xlim([world_size[0] * -0.1, world_size[0] * 1.1])
	ax.set_ylim([world_size[0] * -0.1, world_size[1] * 1.1])

	for f in figures:
		xys = figure_to_xy(f)
		for i,xy in enumerate(xys):
			circle = plt.Circle(tuple(xy), fig_radiuses[i], edgecolor='red',facecolor='none')
			ax.add_artist(circle)

	for v in voxels:
		rect = patches.Rectangle([v[0],v[1]], voxel_size,voxel_size, edgecolor='black',facecolor='grey',alpha=0.25)
		ax.add_patch(rect)

	plt.show()
#==========useful functions================#

#returns true if collide
def figures_collide(fig1,fig2):
	xys1 = figure_to_xy(fig1)
	xys2 = figure_to_xy(fig2)
	for i in range(fig_disk_num):
		for j in range(fig_disk_num):
			if np.linalg.norm(xys1[i]-xys2[j]) < (fig_radiuses[i] + fig_radiuses[j]):
				return True
	return False


def min_max_sin(angle,angle_delta):
	while angle>2*np.pi:
		angle -= 2*np.pi
	while angle<0.0:
		angle += 2*np.pi

	v1 = np.sin(angle);
	v2 = np.sin(angle+angle_delta);

	result= [min(v1,v2),max(v1,v2)]

	if angle < 0.5*np.pi and angle+angle_delta > 0.5*np.pi:
		result[1] = 1.0
	elif angle < 1.5*np.pi and angle+angle_delta > 1.5*np.pi:
		result[0] = -1.0
	return result

def min_max_cos(angle,angle_delta):
	while angle>2*np.pi:
		angle -= 2*np.pi
	while angle<0.0:
		angle += 2*np.pi

	v1 = np.cos(angle);
	v2 = np.cos(angle+angle_delta);

	result= [min(v1,v2),max(v1,v2)]

	if  (angle < 0 and angle+angle_delta > 0 ) or (angle < 2.0*np.pi and angle+angle_delta > 2.0*np.pi):
		result[1] = 1.0
	elif angle < np.pi and angle+angle_delta > np.pi:
		result[0] = -1.0

	return result

def disk_voxel_intersect(ef_index,ef_xy,ef_angle,vf_index,voxel_xy,voxel_angle):

	minmax = min_max_cos(fig_angles[vf_index]+voxel_angle,voxel_angular_size)
	minmaxX = [
		voxel_xy[0]+fig_distances[vf_index]*minmax[0],
		voxel_xy[0]+fig_distances[vf_index]*minmax[1]+voxel_size
		]
	minmax = min_max_sin(fig_angles[vf_index]+voxel_angle,voxel_angular_size)
	minmaxY = [
		voxel_xy[1]+fig_distances[vf_index]*minmax[0],
		voxel_xy[1]+fig_distances[vf_index]*minmax[1]+voxel_size
		]

	maxDistance2 = 0.0;
	x0 = ef_xy[0] + fig_distances[ef_index]*np.cos(fig_angles[ef_index]+ef_angle);
	if x0 < minmaxX[0]:
		maxDistance2 += (x0-minmaxX[1])*(x0-minmaxX[1])
	elif x0 >= minmaxX[0] and x0 <= minmaxX[1]:
		maxDistance2 += max((x0-minmaxX[0])*(x0-minmaxX[0]), (x0-minmaxX[1])*(x0-minmaxX[1]))
	else:
		maxDistance2 += (x0-minmaxX[0])*(x0-minmaxX[0])

	y0 = ef_xy[1] + fig_distances[ef_index]*np.sin(fig_angles[ef_index]+ef_angle)
	if y0 < minmaxY[0]:
		maxDistance2 += (y0-minmaxY[1])*(y0-minmaxY[1])
	elif y0 >= minmaxY[0] and y0 <= minmaxY[1]:
		maxDistance2 += max((y0-minmaxY[0])*(y0-minmaxY[0]), (y0-minmaxY[1])*(y0-minmaxY[1]))
	else:
		maxDistance2 += (y0-minmaxY[0])*(y0-minmaxY[0]);

	maxDistance2 -= (fig_radiuses[ef_index] + fig_radiuses[vf_index])*(fig_radiuses[ef_index] + fig_radiuses[vf_index])
	if(maxDistance2<0.0):
		print(maxDistance2<0.0)
	return (maxDistance2<0.0)



def reject_voxel(fig,voxel):
	#print("rejecting fig:",fig," voxel:",voxel)
	for i in range(fig_disk_num):
		for j in range(fig_disk_num):
			if disk_voxel_intersect(i,fig,fig[2],j,voxel,voxel[2]):
				return True
	return False


#===========actual implementation===============#

voxels = []
for x in range(3):
	for y in range(3):
		voxels.append(np.array([x,y,0.0]))
figures = []

for iter in range(10):

	for i in range(100):
		voxel = random.choice(voxels)
		x = np.random.uniform(voxel[0],voxel[0]+voxel_size)
		y = np.random.uniform(voxel[1],voxel[1]+voxel_size)
		angle = np.random.uniform(voxel[2],voxel[2]+voxel_angular_size)
		fig = np.array([x,y,angle])
		doadd = True
		for f in figures:
			if figures_collide(f,fig):
				doadd = False
		if doadd:
			figures.append(fig)

	draw(voxels,figures)
	voxel_size = 0.5 * voxel_size
	voxel_angular_size = 0.5 * voxel_angular_size

	new_voxels = []
	for v in voxels:
		x1 = v[0]
		x2 = x1 + voxel_size
		y1 = v[1]
		y2 = y1 + voxel_size
		angle1 = v[2]
		angle2 = angle1 + voxel_angular_size
		new_voxels.append(np.array([x1,y1,angle1]))
		new_voxels.append(np.array([x1,y2,angle1]))
		new_voxels.append(np.array([x2,y1,angle1]))
		new_voxels.append(np.array([x2,y2,angle1]))
		new_voxels.append(np.array([x1,y1,angle2]))
		new_voxels.append(np.array([x1,y2,angle2]))
		new_voxels.append(np.array([x2,y1,angle2]))
		new_voxels.append(np.array([x2,y2,angle2]))

	voxels = []
	for count,v in enumerate(new_voxels):
		#print("voxel rejection:",count)
		doreject = False
		for f in figures:
			if reject_voxel(f,v):
				doreject = True
		if not doreject:
			voxels.append(v)

	draw(voxels,figures)

#OK. this sorta works, but the voxel rejection is fuckled. Fix it today. like serious, this aint working the way its intended to
