import csv
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=[12,6])

f = open("Data/results_precise_75.csv")
csv_reader = csv.reader(f)

voxel_removal_tresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
voxel_removal_tresholds = [0.1,0.3,0.5,0.7,0.9]
#voxel_removal_tresholds = [0.85,0.875,0.9,0.925,0.95,0.975]

data_dict={512*1:{},512*2:{},512*4:{},512*8:{},512*16:{}}
#data_dict={512*8:{}}
for k in data_dict:
	data_dict[k]={i:[] for i in voxel_removal_tresholds}


for result in csv_reader:
	data_dict[int(result[1])][float(result[0])].append(float(result[3]))


for k in data_dict:
	for i in data_dict[k]:
		arr = data_dict[k][i]
		data_dict[k][i] = {"mean":np.mean(arr),"stddev":np.std(arr)}

new_data = {k:{} for k in data_dict}

for k in new_data:
	new_data[k]["means"] = [data_dict[k][i]["mean"] for i in data_dict[k]]
	new_data[k]["stddev"] = [data_dict[k][i]["stddev"] for i in data_dict[k]]

#plt.plot(voxel_removal_tresholds,)
for k in new_data:
	if k != 3072:
		plt.plot(voxel_removal_tresholds, new_data[k]["means"],label=str(int(k/512))+" * 512",marker='o')
plt.legend()
plt.xlabel("voxel split treshold")
plt.ylabel("execution time (seconds)")

plt.show()
