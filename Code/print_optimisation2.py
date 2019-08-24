import csv
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=[12,6])

f = open("Data/results_sizes.csv")
ft = open("Data/results_sizes_avg.csv",'w+')
csv_reader = csv.reader(f)

data_dict = {}

for result in csv_reader:
	k = int(result[2])
	if k in data_dict:
		data_dict[k].append(float(result[3]))
	else:
		data_dict[k]=[float(result[3])]

for k in data_dict:
	data_dict[k] = sum(data_dict[k])/len(data_dict[k])
	ft.write(str(k)+","+str(data_dict[k])+"\n")
