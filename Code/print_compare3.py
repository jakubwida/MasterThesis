import csv
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=[12,6])

f1 = open("Data/results_sizes_avg.csv",'r')
f2 = open("Data/rsa3d.csv",'r')
csv_reader1 = csv.reader(f1)
csv_reader2 = csv.reader(f2)
gpursa_times = [float(result[1]) for result in csv_reader1]
rsa3d_times = [float(result[1]) for result in csv_reader2]
sizes=[25,50,75,100,125,150,175,200,225,250]
surfaces = [s**2 for s in sizes]
sizes2 = [str(s**2)+"\n("+str(s)+")" for s in sizes]
print(sizes)
print(rsa3d_times)
print(gpursa_times)

plt.plot(sizes2,rsa3d_times,label='CPU',marker='o')
plt.plot(sizes2,gpursa_times,label='GPU',marker='o')
#plt.xticks(sizes2)
plt.xlabel("packing surface area, non-linear scale \n(side length)")
plt.ylabel("execution time (seconds)")

plt.legend()
plt.show()
