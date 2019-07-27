import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def _draw_figure(world,fig,ax,color='r'):
	fig_xys = world.figure_to_xy(fig)
	for i,xy in enumerate(fig_xys):
		circle = plt.Circle(tuple(xy), world.fig_radiuses[i], edgecolor=color,facecolor='none')
		ax.add_artist(circle)

def draw(world):

	plt.figure(figsize=[24,12])
	ax = plt.gca()
	ax.set_xlim([-10,world.world_size_float[0]+10.0])
	ax.set_ylim([-10,world.world_size_float[1]+10.0])

	figs = world.gpu_figs.get()

	for i in range(world.fig_num):
		fig = figs[i]
		_draw_figure(world,fig,ax)

		cell_pos = np.floor(np.array(fig)/world.cell_size)
		fpos2 = np.copy(fig)
		if cell_pos[0]==0 or cell_pos[0]==world.cell_num_x-1 or cell_pos[1]==0 or cell_pos[1]==world.cell_num_y-1:
			if cell_pos[0]==0:
				fpos2[0]+=world.world_size_float[0]
			if cell_pos[0]==world.cell_num_x-1:
				fpos2[0]-=world.world_size_float[0]
			if cell_pos[1]==0:
				fpos2[1]+=world.world_size_float[1]
			if cell_pos[1]==world.cell_num_y-1:
				fpos2[1]-=world.world_size_float[1]
			_draw_figure(world,fpos2,ax,'green')

	voxels = world.gpu_voxels.get()
	for i in range(world.voxel_num):
		voxel_pos = voxels[i]
		rect = patches.Rectangle([voxel_pos[0],voxel_pos[1]], world.voxel_size,world.voxel_size, edgecolor='black',facecolor='grey',alpha=0.25)
		ax.add_patch(rect)

	plt.show()
