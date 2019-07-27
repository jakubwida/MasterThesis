import numpy as np

#monte carlo approximation of the figure area
def fig_area(radiuses,positions):

	size = len(radiuses)

	highs = positions[:,1]+radiuses
	lows = positions[:,1]-radiuses
	lefts = positions[:,0]-radiuses
	rights = positions[:,0]+radiuses

	left_bound = lefts.min()
	right_bound = rights.max()
	upper_bound = highs.max()
	lower_bound = lows.min()
	rect_area = (right_bound - left_bound) * (upper_bound - lower_bound)


	point_num = 10000
	random_x = np.random.uniform(left_bound,right_bound,(point_num))
	random_y = np.random.uniform(lower_bound,upper_bound,(point_num))

	random_pts = np.column_stack((random_x,random_y))

	counter = 0
	for i in range(point_num):
		for j in range(size):
			if np.linalg.norm(random_pts[i]-positions[j]) < radiuses[j]:
				counter+=1
				break


	return (rect_area * (float(counter)/float(point_num)))
