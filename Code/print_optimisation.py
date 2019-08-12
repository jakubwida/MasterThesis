import csv
import matplotlib.pyplot as plt
import numpy as np

f = open("Data/results.csv")
csv_reader = csv.reader(f)

voxel_removal_tresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

data_dict={512*1:{},512*2:{},512*4:{},512*8:{}}
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
	plt.errorbar(voxel_removal_tresholds, new_data[k]["means"], new_data[512]["stddev"], linestyle='None', marker='^')
plt.show()
