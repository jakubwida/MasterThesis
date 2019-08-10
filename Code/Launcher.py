import numpy as np
from World import World
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

#generates a dimer with a given x, so that it fits to 1.0/1.0 cell
def _generate_dimer(x):
	r = 1.0/(2.0*x+2.0)
	d = 2.0 * x * r
	return [(r,0.0,0.0),(r,d,0.0)]

def _generate_fibrinogen():
	r_edge = 3.35
	r_mid = 2.65
	r_small = 0.75

	a = r_edge * 4.0 + 40.0* r_small + r_mid * 2.0
	r_edge = r_edge/a
	r_mid = r_mid/a
	r_small = r_small/a

	arr = [(r_edge,0.0,0.0)]
	for i in range(10):
		arr.append((r_small,r_edge+(i*2.0*r_small)+r_small,0.0))
	midpos = 20.0*r_small + r_edge + r_mid
	arr.append((r_mid,midpos,0.0))
	for i in range(10):
		arr.append((r_small,midpos+r_mid+(i*2.0*r_small)+r_small,0.0))
	arr.append((r_edge,2*midpos,0.0))
	return arr

def _draw_figure(figure_config):

	min_x = min([f[1]-f[0] for f in figure_config])
	max_x = max([f[1]+f[0] for f in figure_config])
	min_y = min([f[2]-f[0] for f in figure_config])
	max_y = max([f[2]+f[0] for f in figure_config])
	xsize = max_x - min_x
	ysize = max_y - min_y

	#plt.figure(figsize=[12,12])
	ax = plt.gca()
	ax.set_xlim([min_x-0.2*xsize,max_x+0.2*xsize])
	ax.set_ylim([min_y-0.2*ysize,max_y+0.2*ysize])
	for f in figure_config:
		circle = plt.Circle((f[1],f[2]), f[0], edgecolor='red',facecolor='none')
		ax.add_artist(circle)
	plt.show()

common_configs = {
	"dimer_x01":_generate_dimer(0.1),
	"dimer_x05":_generate_dimer(0.5),
	"dimer_x09":_generate_dimer(0.9),
	"fibrinogen":_generate_fibrinogen()
	}

#function for one use, fills up the result json for one particular config
# does 10 trials for each parameter combo
def fill_up_results(config):
	for fig_added_treshold in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
		for added_fig_num in [512*1,512*2,512*4,512*8]:
			for voxel_added_treshold in [10000,100000,100000]:
				for cell_num in [10,50,100]:
					w = World(config,1.0,(cell_num,cell_num),added_fig_num,fig_added_treshold,voxel_added_treshold)
					w.perform_rsas(10)


#TODO: rejangle the fill_up_results, so that it saves to it's own json
# and saves only the total time



#fill_up_results(common_configs["dimer_x01"])
#_draw_figure(common_configs["dimer_x=0.1"])
#_draw_figure(common_configs["dimer_x=0.5"])
#_draw_figure(common_configs["dimer_x=0.9"])
#_draw_figure(common_configs["fibrinogen"])

w = World(common_configs["dimer_x05"],1.0,(10,5),512,0.5,10000000)
w.perform_rsa(draw="ITERATION")
