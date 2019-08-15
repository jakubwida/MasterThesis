import numpy as np
from World import World
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def _generate_dimer(x):
	r = 1.0/(2.0*x+2.0)
	d = 2.0 * x * r
	return [(r,0.0,0.0),(r,d,0.0)]

common_configs = {
	"dimer_x01":_generate_dimer(0.1),
	"dimer_x05":_generate_dimer(0.5),
	"dimer_x09":_generate_dimer(0.9)
	}


w = World(common_configs["dimer_x01"],1.0,(50,50),512*2,0.95,10**6)
resdict = w.perform_rsa(print_times="ALL")
#print(resdict)
data = {
	"timers":{
		"generation":[],
		"reject_vs_existing":[],
		"reject_vs_new":[],
		"split_voxels":[],
		"reject_voxels":[]
	},
	"data":{
		"voxel_num":[],
		"voxel_fraction":[],
		"fig_num":[],
		"density":[]
	}
}
for iteration in resdict['iterations']:
	for k in data:
		for kk in data[k]:
			data[k][kk].append(iteration[k][kk])

for k in data:
	for kk in data[k]:
		data[k][kk] = np.array(data[k][kk])

iter_num = len(resdict['iterations'])
totals = np.zeros(iter_num)

#summary writing =========================

summary_times = {}
summary_percentages = {}

for k in data['timers']:
	summary_times[k] = np.sum(data['timers'][k])

total = np.sum([summary_times[k] for k in summary_times])

for k in data['timers']:
	summary_percentages[k] = summary_times[k]/total

for k in summary_times:
	print("total time at ",k,":",summary_times[k]," proportionally:",summary_percentages[k])

#plotting ================================

fig, ax1 = plt.subplots()
xes = np.arange(iter_num)

for k in data['timers']:
	ax1.bar(xes,data['timers'][k],bottom=totals,label=k)
	totals = totals + data['timers'][k]

ax2 = ax1.twinx()

ax2.plot(xes,data['data']['voxel_num'],label="voxel_num",color='black')
ax2.plot(xes,data['data']['fig_num'],label="fig_num",color='red')

ax1.legend(loc='upper left')
ax2.legend()
plt.show()
