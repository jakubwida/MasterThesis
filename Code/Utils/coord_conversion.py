import numpy as np


def cart2pol(x, y):
    dist = np.sqrt(x**2 + y**2)
    angle = np.arctan2(y, x)
    return(dist, angle)

def pol2cart(dist, angle):
    x = dist * np.cos(angle)
    y = dist * np.sin(angle)
    return(x, y)

#returns angles,distances
def convert_figure(positions_xy):
	size = positions_xy.shape[0]
	angles = np.zeros((size))
	distances = np.zeros((size))
	for i in range(size):
		distances[i],angles[i]=cart2pol(positions_xy[i][0],positions_xy[i][1])
	return angles,distances
