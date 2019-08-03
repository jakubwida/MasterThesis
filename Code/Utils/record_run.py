import json

#TEMP - currently only a sketch, not usable



#summary_dict:
#
# figure:{radiuses:[r1,r2,...], positions:[(x,y),...]}
# world:{cell_size:[float_x,float_y],world_size:[cell_num_x,cell_num_y],added_fig_num:int}
# total_time: float time
# density:float //area covered vs original
# iterations: [
#	{voxel_size:fraction_of_original,voxel_num:int,fig_num:int,density:float}
#   ,...
# ]

def record_summary(summary_dict):
	pass
#TODO:
# - add adding/using configuration
# -
#


data = {}
data['people'] = []
data['people'].append({
    'name': 'Scott',
    'website': 'stackabuse.com',
    'from': 'Nebraska'
})
data['people'].append({
    'name': 'Larry',
    'website': 'google.com',
    'from': 'Michigan'
})
data['people'].append({
    'name': 'Tim',
    'website': 'apple.com',
    'from': 'Alabama'
})

with open('data.json', 'w') as outfile:
    json.dump(data, outfile, indent=3, sort_keys=True)
