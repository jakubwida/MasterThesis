import json
import csv

def record_run(summary_dict):
	json_file_read = open('Data/results.json',"r")
	data = json.load(json_file_read)
	data['runs'].append(summary_dict)
	json_file_write = open('Data/results.json',"w")
	json.dump(data, json_file_write, indent=3)
	json_file_read.close()
	json_file_write.close()

def save_output(figures,fig_num,file_path):
	with open(file_path, 'w', newline='') as csvfile:
		spamwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
		for i in range(fig_num):
			spamwriter.writerow(list(figures[i]))
