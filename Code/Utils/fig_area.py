import numpy as np
import shapely
from shapely.geometry import point
from shapely.geometry import linestring
from shapely import ops

import matplotlib.pyplot as plt
from descartes import PolygonPatch

def split_disjoint(radiuses,positions):
	size = len(radiuses)
	collision_groups = [{i} for i in range(size)]
	for i in range(size):
		for j in range(size):
			if i!=j:
				if np.linalg.norm(positions[i]-positions[j]) < (radiuses[i]+radiuses[j]):
					newset = collision_groups[i] | collision_groups[j]
					collision_groups[i] = newset
					collision_groups[j] = newset
	out = {frozenset(i) for i in collision_groups}
	out = [list(i) for i in out]
	return out

def circle_group_area(radiuses,positions):
	circles = []
	for i in range(len(radiuses)):
		circles.append(point.Point(positions[i][0],positions[i][1]).buffer(radiuses[i]))

	union = ops.unary_union(circles)
	result = [geom for geom in ops.polygonize(union)]

	completeareas = [list(ops.polygonize(g.exterior))[0].area for g in result]
	max_index = np.argmax(completeareas)
	result_area = result[max_index].area

	return result_area


def single_circle_area(radius):
	return np.pi * radius * radius

def fig_area(radiuses,positions):
	groups = split_disjoint(radiuses,positions)
	totalarea = 0.0
	for group in groups:
		if len(group) == 1:
			totalarea += single_circle_area(radiuses[group[0]])
		else:
			g_radiuses = radiuses[np.array(group)]
			g_positions = positions[np.array(group)]
			totalarea += circle_group_area(radiuses,positions)
	return totalarea


radiuses = np.array([0.5,5.0])
positions = np.array([(0.0,0.0),(15.0,15.0)])
print(fig_area(radiuses,positions))
