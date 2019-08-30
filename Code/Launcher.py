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

	counter = 0
	start_c = 45
	tresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
	tresholds2 = [0.1,0.3,0.5,0.7,0.9]
	cell_nums = [10,30,50,70,90]
	cell_nums = [25,50,75,100,125,150,175,200,225,250]
	added_fig_nums = [512*1,512*2,512*4,512*8,512*16]
	res_f = open("Data/results.csv","a+")
	for fig_added_treshold in [0.98]:
		for added_fig_num in [512*32]:
			for cell_num in cell_nums:
				w = World(config,1.0,(cell_num,cell_num),(cell_num/25)*2*512,fig_added_treshold,(10 ** 5)*5)
				for i in range(5):
					if counter >= start_c:
						print("START",counter,"/",5*10)
						results = w.perform_rsa(save_summary=False)

						res_f.write(str(fig_added_treshold)+","+str(added_fig_num)+","+str(cell_num)+","+str(results["summary"]["total_time"])+"\n")
						print("END:",results["summary"]["total_time"])

					counter +=1

#TODO: rejangle the fill_up_results, so that it saves to it's own json
# and saves only the total time



#fill_up_results(common_configs["dimer_x05"])
#_draw_figure(common_configs["dimer_x=0.1"])
#_draw_figure(common_configs["dimer_x=0.5"])
#_draw_figure(common_configs["dimer_x=0.9"])
#_draw_figure(common_configs["fibrinogen"])
w = World(common_configs["dimer_x05"],1.0,(200,200),512*8,0.98,1000000)
w.perform_rsa(print_times="ALL",draw="NONE")
#for i in range(1000):
#	print(i)
#	w.perform_rsa()
