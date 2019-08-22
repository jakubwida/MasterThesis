import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np



def draw(ax,world_size,positions,radius):
	for pos in positions:
		circle = plt.Circle(tuple(pos), radius, edgecolor='red',facecolor='red',alpha=0.2)
		ax.add_artist(circle)

def draw_edges(ax,world_size,positions,radius):
	for translation in [(world_size,0),(-world_size,0)]:
		translation = np.array(translation)
		for pos in positions:
			circle = plt.Circle(tuple(np.array(pos)+translation), radius, edgecolor='blue',facecolor='blue',alpha=0.2)
			ax.add_artist(circle)

def prepare(world_size):
	plt.figure(figsize=[12,12])
	ax = plt.gca()
	plt.xticks([])
	plt.yticks([])
	ax.set_xlim([-world_size*0.5,world_size*1.5])
	ax.set_ylim([-world_size*0.5,world_size*1.5])
	rect = patches.Rectangle([0,-5.0], world_size,world_size+10.0, edgecolor='black',facecolor='black',alpha=0.1)
	ax.add_patch(rect)
	return ax

world_size = 10.0
radius = 2.0
positions = [(1.5,0.0),(3.0,5.0),(10.0,9.0),(8.0,4.0)]


ax = prepare(world_size)
draw(ax,world_size,positions,radius)
plt.show()

ax = prepare(world_size)
draw(ax,world_size,positions,radius)
draw_edges(ax,world_size,positions,radius)
plt.show()
