import numpy as np

#draw TEMP
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def two_circle_bounding_circle_middle_radius(pos_0,rad_0,pos_1,rad_1):
	pos_0_pos_1_dist = np.linalg.norm(pos_0-pos_1)
	max_radius = (pos_0_pos_1_dist+rad_0+rad_1)/2.0

	pos_0_to_center = max_radius - rad_0
	t_ratio = pos_0_to_center/pos_0_pos_1_dist
	weighted_center = (((1.0-t_ratio)*pos_0) + (t_ratio * pos_1))
	return weighted_center,max_radius

#APPROXIMATES NOT SOLVES
#creates a figure configuration where origin point lies in the centre of the smallest bounding circle of the polydisk
#radiuses = [r1,r2...] circle radius list
#positions = [(x1,y1),(x2,y2),...] positions of circles from initial origin
#RETURNS: figure radius,new circle positions
def balance_figure(radiuses,positions):
	circle_num = len(radiuses)
	radiuses = np.array(radiuses)
	positions = np.array(positions)

	#distance_matrix = np.zeroes((circle_num,circle_num))
	max_dist = 0
	first_indexes = [0,0]
	for i in range(circle_num):
		for j in range(circle_num):
			new_dist =  np.linalg.norm(positions[i]-positions[j])+radiuses[i]+radiuses[j]
			if new_dist > max_dist:
				first_indexes = [i,j]
				max_dist = new_dist

	pos_0 = positions[first_indexes[0]]
	pos_1 = positions[first_indexes[1]]
	rad_0 = radiuses[first_indexes[0]]
	rad_1 = radiuses[first_indexes[1]]

	weighted_center,max_radius = two_circle_bounding_circle_middle_radius(pos_0,rad_0,pos_1,rad_1)

	second_index = 0
	max_dist_1 = 0
	for i in range(circle_num):
		new_dist = np.linalg.norm(weighted_center-positions[i]) + radiuses[i]
		if new_dist > max_dist_1:
			second_index = i
			max_dist_1 = new_dist
	pos_2 = positions[second_index]
	rad_2 = radiuses[second_index]

	print(weighted_center)
	print(pos_0,pos_1,pos_2)
	print("max_dist_1",max_dist_1,"max_radius",max_radius)
	if max_dist_1 > max_radius+0.00000000000001:
		print("!!!3!!!")
		c_01,r_01 = two_circle_bounding_circle_middle_radius(pos_0,rad_0,pos_1,rad_1)
		c_02,r_02 = two_circle_bounding_circle_middle_radius(pos_0,rad_0,pos_2,rad_2)
		c_12,r_12 = two_circle_bounding_circle_middle_radius(pos_1,rad_1,pos_2,rad_2)

		weighted_center = (c_01+c_02+c_12)/3.0
		max_radius = max(np.linalg.norm(pos_0-weighted_center)+rad_0, np.linalg.norm(pos_1-weighted_center)+rad_1, np.linalg.norm(pos_2-weighted_center)+rad_2)


	#setting the centers:
	positions = positions - weighted_center
	weighted_center = np.array([0.0,0.0])

	return max_radius,positions
