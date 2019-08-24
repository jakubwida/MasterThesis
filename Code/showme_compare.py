import numpy as np
from World import World
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator
import numpy as np
import csv

def _generate_dimer(x):
	r = 1.0/(2.0*x+2.0)
	d = 2.0 * x * r
	return [(r,0.0,0.0),(r,d,0.0)]

dimer_x05=_generate_dimer(0.5)

def fill_up_results(config):

	counter = 0
	start_c = 0

	cell_nums = [25,50,75,100,125,150,175,200,225,250]

	res_f = open("Data/results.csv","a+")
	for fig_added_treshold in [0.98]:
		for added_fig_num in [512*32]:
			for cell_num in cell_nums:
				w = World(config,1.0,(cell_num,cell_num),(cell_num/25)*2*512,fig_added_treshold,(10 ** 5)*5)
				for i in range(1):
					if counter >= start_c:
						print("START",counter,"/",5*10)
						results = w.perform_rsa(save_summary=False)
						gen_t = []
						rej_e_t = []
						rej_n_t = []
						split_t = []
						rej_v_t = []
						total_t=[]
						for t in results['iterations']:
							ts = t['timers']
							gen_t.append(ts['generation'])
							rej_e_t.append(ts['reject_vs_existing'])
							rej_n_t.append(ts['reject_vs_new'])
							split_t.append(ts['split_voxels'])
							rej_v_t.append(ts['reject_voxels'])
							total_t.append(ts['iteration'])

						gen_t = sum(gen_t)
						rej_e_t = sum(rej_e_t)
						rej_n_t = sum(rej_n_t)
						split_t = sum(split_t)
						rej_v_t = sum(rej_v_t)
						total_t= sum(total_t)

						res_f.write(str(cell_num)+","+str(gen_t)+","+str(rej_e_t)+","+str(split_t)+","+str(rej_v_t)+","+str(total_t)+"\n")

						print("END:",results["summary"]["total_time"])

					counter +=1

#fill_up_results(dimer_x05)

f = open("Data/results.csv")
csv_reader = csv.reader(f)

sides = []
gen_t = []
rej_e_t = []
rej_n_t = []
split_t = []
rej_v_t = []
total_t=[]

totals = [0 for i in range(10)]

for result in csv_reader:
	print(result)
	sides.append(result[0])
	gen_t.append(float(result[1]))
	rej_e_t.append(float(result[2]))
	rej_n_t.append(float(result[3]))
	split_t.append(float(result[4]))
	rej_v_t.append(float(result[5]))
	total_t.append(float(result[5])+float(result[1])+float(result[2])+float(result[3])+float(result[4]))

total_t=np.array(total_t)

gen_t = np.array(gen_t)/total_t
rej_e_t = np.array(rej_e_t)/total_t
rej_n_t = np.array(rej_n_t)/total_t
split_t = np.array(split_t)/total_t
rej_v_t = np.array(rej_v_t)/total_t

print(gen_t)
plt.figure(figsize=[12,6])
ax1 = plt.gca()

totals = np.zeros(10)

ax1.bar(sides,gen_t,bottom=totals,label='A: generation')
totals = totals + gen_t
ax1.bar(sides,rej_e_t,bottom=totals,label='B: rejection of shapes against existing')
totals = totals + rej_e_t
ax1.bar(sides,rej_n_t,bottom=totals,label='C: rejection of shapes against new')
totals = totals + rej_n_t
ax1.bar(sides,split_t,bottom=totals,label='D: splitting voxels')
totals = totals + split_t
ax1.bar(sides,rej_v_t,bottom=totals,label='E: rejecting voxels')
totals = totals + rej_v_t

plt.xlabel("packing side length")
plt.ylabel("execution time (as proportion of total)")
plt.legend()
plt.show()
