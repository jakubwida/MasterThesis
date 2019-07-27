import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray

import numpy as np
import math
import time

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from CUDASourceModule import get_module as get_module

#functions:

#figure = [x,y,angle]
def figure_to_xy(figure,figure_disk_num,figure_distances,figure_angles):
	rotated_angles = figure_angles + figure[2]
	xs = figure_distances * np.cos(rotated_angles) + figure[0]
	ys = figure_distances * np.sin(rotated_angles) + figure[1]
	return np.column_stack((xs,ys))

#returns 1 if they collide
#fig_1_xy,fig_2_xy = outputs of figure_to_xy(...)
def figure_collide(fig_1_xy,fig_2_xy,figure_radiuses):
	for n1,d1 in enumerate(fig_1_xy):
		for n2,d2 in enumerate(fig_2_xy):
			if np.linalg.norm(d1-d2) < figure_radiuses[n1]+figure_radiuses[n2]:
				return True
	return False

def _draw_figure(figure,figure_disk_num,figure_radiuses,figure_distances,figure_angles,ax,color='r'):
	figxys = figure_to_xy(figure,figure_disk_num,figure_distances,figure_angles)
	for i,xy in enumerate(figxys):
		circle = plt.Circle(tuple(xy), figure_radiuses[i], edgecolor=color,facecolor='none')
		ax.add_artist(circle)

def draw(figures,figure_radius,fi_num,voxels,voxel_num,voxel_size,figure_disk_num,figure_radiuses,figure_distances,figure_angles):
	plt.figure(figsize=[24,12])
	ax = plt.gca()
	ax.set_xlim([-10,50])
	ax.set_ylim([-10,50])
	world_s = np.array([cell_num_x*cell_size,cell_num_y*cell_size])
	for i in range(fi_num):
		fig_pos = figures[i]
		#fig_pos = (fig_pos[0],fig_pos[1])
		_draw_figure(fig_pos,figure_disk_num,figure_radiuses,figure_distances,figure_angles,ax)
		#circle1 = plt.Circle(tuple(fig_pos), figure_radius, edgecolor='r',facecolor='none')
		#circle2 = plt.Circle(tuple(fig_pos), figure_radius*2, edgecolor='r',facecolor='none')
		#ax.add_artist(circle1)
		#ax.add_artist(circle2)

		cell_pos = np.floor(np.array(fig_pos)/cell_size)
		fpos2 = np.copy(fig_pos)
		if cell_pos[0]==0 or cell_pos[0]==cell_num_x-1 or cell_pos[1]==0 or cell_pos[1]==cell_num_y-1:
			if cell_pos[0]==0:
				fpos2[0]+=world_s[0]
			if cell_pos[0]==cell_num_x-1:
				fpos2[0]-=world_s[0]
			if cell_pos[1]==0:
				fpos2[1]+=world_s[1]
			if cell_pos[1]==cell_num_y-1:
				fpos2[1]-=world_s[1]
			_draw_figure(fpos2,figure_disk_num,figure_radiuses,figure_distances,figure_angles,ax,'green')
			#circle1 = plt.Circle(tuple(fpos2), figure_radius, edgecolor='green',facecolor='none')
			#circle2 = plt.Circle(tuple(fpos2), figure_radius*2, edgecolor='green',facecolor='none')
			#ax.add_artist(circle1)
			#ax.add_artist(circle2)

	for i in range(voxel_num):
		voxel_pos = voxels[i]
		rect = patches.Rectangle([voxel_pos[0],voxel_pos[1]], voxel_size,voxel_size, edgecolor='black',facecolor='grey',alpha=0.25)
		ax.add_patch(rect)
	plt.show()


def print_time_delta(text,stamp):
	newtime = time.time()
	delta = newtime-stamp
	print(text,delta);
	return newtime

def print_cuda_mem():
	(free,total)=cuda.mem_get_info()
	print("Global memory occupancy:%f%% free"%(free*100/total))

def translate_position(nparray_pos,nparray_world_size,nparray_cell_pos_int,cell_size):
	out = np.copy(nparray_pos)
	neighborhood_p = nparray_cell_pos_int.astype(np.float64)*cell_size
	neighborhood_l = neighborhood_p - cell_size
	neighborhood_u = neighborhood_p + 2*cell_size

	lb = out<neighborhood_l
	ub = out>neighborhood_u
	out[lb]+=nparray_world_size[lb]
	out[ub]-=nparray_world_size[ub]
	return out


def perform_rsas(number_of_trials,fail_ratio_to_subdivide,added_fig_num,cell_size,cell_num_x,cell_num_y,figure_radius):

	print("@ INITIALISATION")
	init_stamp = time.time()

	#constants:
	fail_ratio_to_subdivide = np.float32(fail_ratio_to_subdivide)
	added_fig_num = np.int32(added_fig_num)
	cell_size = np.float32(cell_size)
	cell_num_x = np.int32(cell_num_x)
	cell_num_y = np.int32(cell_num_y)

	figure_radius = np.int32(figure_radius)
	max_figures_per_cell = np.int32(np.ceil(((cell_size+2.0*figure_radius) ** 2.0) / (figure_radius ** 2) * np.pi))
	max_figures_per_neighborhood = np.int32(max_figures_per_cell * 9)

	voxel_removal_tolerance = np.float32(0.0)

	#DEPTH
	#angle in radians
	base_voxel_depth = np.float32(2.0 * np.pi)

	#MULTIPLE CIRCLES currently hardcoded
	figure_disk_num = np.int32(2)
	figure_radiuses = np.float32([0.5,0.5])
	figure_distances = np.float32([0.5,0.5])
	figure_angles = np.float32([0.0,np.pi])



	#====================================================================================
	#source module functions

	module = get_module(added_fig_num,cell_size,cell_num_x,cell_num_y,
	figure_radius,max_figures_per_cell,max_figures_per_neighborhood,
	figure_disk_num,figure_radiuses,figure_distances,figure_angles)

	init_func = module.get_function("init")
	gen_func = module.get_function("gen_figs")
	reject_v_existing_func = module.get_function("reject_figs_vs_existing")
	split_func = module.get_function("split_voxels")
	reject_voxels_func = module.get_function("reject_voxels")

	print_time_delta("--TIME: ",init_stamp)

	for trial in range(number_of_trials):



		#====================================================================================
		#variables
		voxel_size = cell_size
		#DEPTH
		voxel_depth = base_voxel_depth

		voxel_num = np.int32(cell_num_x * cell_num_y)

		#data structures
		cells = (np.ones((cell_num_x,cell_num_y,max_figures_per_cell)) * (-1)).astype(np.int32)
		neighborhoods = (np.ones((cell_num_x,cell_num_y,max_figures_per_neighborhood)) * (-1)).astype(np.int32)

		#COLUMNS: X, Y, ANGLE:DEPTH
		voxels = []
		for x in range(cell_num_x):
			for y in range(cell_num_y):
				voxels.append([float(x)*cell_size,float(y)*cell_size,0.0])
		voxels = np.array(voxels).astype(np.float32)

		#COLUMNS: X,Y,ANGLE
		figures = np.array([[-50.0,-50.0,0.0]]).astype(np.float32)
		#figures = np.array([]).astype(np.float32)
		added_figures = np.zeros((added_fig_num,2)).astype(np.float32)
		fig_num = np.int32(0)




		#==================================================================================



		#initialising pycuda kernels
		seed = np.int32(time.time())
		#seed = np.int32(12341234)
		init_func(seed, block=(512,1,1), grid=(math.ceil(added_fig_num/512),1))


		#performing

		gpu_figures = gpuarray.to_gpu(figures)
		gpu_neighborhoods = gpuarray.to_gpu(neighborhoods)
		gpu_voxels = gpuarray.to_gpu(voxels)
		gpu_added_figures = gpuarray.zeros((added_fig_num,3),np.float32)
		gpu_added_fig_cell_positions = gpuarray.zeros((added_fig_num,2),np.int32)


		total_stamp = time.time()
		iteration_stamp = time.time()
		iteration = 0;
		#iterating
		while(voxel_num > 0):

			#print("iteration",iteration)
			#print("voxel_num",voxel_num)
			#immediate_stamp = time.time()

			#generating figures at free voxels
			gen_func(
				gpu_voxels,voxel_size,voxel_depth,voxel_num,gpu_added_figures,
				block=(512,1,1), grid=(math.ceil(added_fig_num/512),1))

			#if iteration == 0:
			#arrg = gpu_added_figures.get()
			#arrg.fill(5.0)
			#gpu_added_figures.set(arrg)

			#if iteration == 1:
			#	arrg = gpu_added_figures.get()
			#	arrg.fill(2.45)
			#	gpu_added_figures.set(arrg)

			#immediate_stamp = print_time_delta("after_generating",immediate_stamp)
			#if voxel_size <= 0.125:
			#	draw(gpu_added_figures.get(),0.01,added_fig_num,voxels,voxel_num,voxel_size)
			#	draw(figures,figure_radius,fig_num,voxels,voxel_num,voxel_size)
			#rejecting figures colliding with those already existing

			reject_v_existing_func(
				gpu_figures,fig_num,
				gpu_added_figures,gpu_added_fig_cell_positions,
				gpu_neighborhoods,
				block=(512,1,1), grid=(math.ceil(added_fig_num/512),1))


			#immediate_stamp = print_time_delta("after_rejecting 1",immediate_stamp)

			#rejecting figures that were added but collide with each other
			#inserting them to neighborhoods, and add to figures list
			##CPU

			f_list = []
			added_figures = gpu_added_figures.get()

			added_fig_cell_positions = gpu_added_fig_cell_positions.get()
			neighborhoods = gpu_neighborhoods.get()


			counter = 0
			rejected_figures = 0

			added_fig_indexes = (added_figures != -1.0)[:,0]
			added_figures = added_figures[added_fig_indexes,:]
			added_fig_cell_positions = added_fig_cell_positions[added_fig_indexes,:]

			print("figures that passed rejecting by existing",added_figures.size/3)

			pseudo_neighborhoods = [[[] for y in range(cell_num_y+1)] for x in range(cell_num_x+1)]

			fr2 = figure_radius * 2.0
			world_s = np.array([cell_num_x*cell_size,cell_num_y*cell_size])


			for index,figure in enumerate(added_figures):
				figuree = np.array((figure[0],figure[1]))
				cell_pos = added_fig_cell_positions[index]
				checked_figs = pseudo_neighborhoods[cell_pos[0]][cell_pos[1]]
				#print("checked_f",checked_figs)
				addfig=True

				for fig in checked_figs:
					figg = translate_position(np.array([fig[0],fig[1]]),world_s,np.array(cell_pos),cell_size)

					fig_1_xy = figure_to_xy(np.array([figuree[0],figuree[1],figure[2]]),figure_disk_num,figure_distances,figure_angles)
					fig_2_xy = figure_to_xy(np.array([figg[0],figg[1],fig[2]]),figure_disk_num,figure_distances,figure_angles)
					if figure_collide(fig_1_xy,fig_2_xy,figure_radiuses):
						addfig = False

					#if np.linalg.norm(figg-figuree) < fr2:
					#	addfig=False
				if addfig:
					#checked_figs.append(figure)
					f_list.append(figure)
					#pseudo_neighborhoods[cell_pos[0]][cell_pos[1]].append(figure)

					n_array = [(-1,-1),(0,-1),(1,-1),(-1,0),(0,0),(1,0),(-1,1),(0,1),(1,1)]
					t_array = []
					for n in n_array:
						n=np.array(n)
						n=n+cell_pos
						#here lies the crux of the issue
						if n[0] < 0:
							n[0] = cell_num_x-1
						if n[0] >= cell_num_x:
							n[0] = 0
						if n[1] < 0:
							n[1] = cell_num_y-1
						if n[1] >= cell_num_y:
							n[1] = 0
						#if not (n[0]>=cell_num_x or n[1]>=cell_num_y or n[0]<0 or n[1]<0):
						t_array.append(tuple(n))
					t_array = np.array(t_array)

					for n in t_array:
						pseudo_neighborhoods[n[0]][n[1]].append(figure)
						for i in range(max_figures_per_neighborhood):
							if neighborhoods[(n[0],n[1],i)] == -1:
								neighborhoods[(n[0],n[1],i)] = np.int32(counter+fig_num)
								break
					counter +=1

			gpu_neighborhoods.gpudata.free()
			gpu_neighborhoods = gpuarray.to_gpu(neighborhoods)
			fig_num += np.int32(len(f_list))
			if iteration == 0:
				figures = np.array(f_list).astype(np.float32)
			elif len(f_list) != 0:
				figures = np.concatenate((figures,np.array(f_list).astype(np.float32)))

			gpu_figures.gpudata.free()
			gpu_figures = gpuarray.to_gpu(figures)



			#print("figures:")
			#for f in figures:
			#    print(f)

			#immediate_stamp = print_time_delta("after_rejecting 2",immediate_stamp)

			#spliting voxels
			#print(counter,counter/added_fig_num)

			if counter/added_fig_num < fail_ratio_to_subdivide:
				gpu_target_voxels = gpuarray.zeros((voxel_num*8,3),np.float32)

				split_func(
					gpu_voxels,voxel_num,voxel_size,voxel_depth,gpu_target_voxels,
					block=(512,1,1), grid=(math.ceil(voxel_num/512),1))

				#gpu_voxels.gpudata.free()
				gpu_voxels = gpu_target_voxels
				print("before gpu_voxels",gpu_voxels.get().size)

				voxel_depth = np.float32(voxel_depth/2.0)
				voxel_num = np.int32(voxel_num * 8)
				voxel_size = np.float32(voxel_size/2.0)

			#draw(figures,figure_radius,fig_num,gpu_voxels.get(),voxel_num,voxel_size)
			for i in gpu_voxels.get():
				print(i)


			print("voxel size before removing:",voxel_size)
			#immediate_stamp = print_time_delta("after_splitting voxels",immediate_stamp)
			#rejecting voxels

			reject_voxels_func(
				gpu_voxels,voxel_num,voxel_size,voxel_depth,gpu_figures,gpu_neighborhoods,
				block=(512,1,1), grid=(math.ceil(voxel_num/512),1))

			print("voxel size:",voxel_size)
			print("gpu_voxels:",gpu_voxels.size)
			voxels = gpu_voxels.get()
			"""
			voxels = gpu_voxels.get()
			for vi in range(voxel_num):
				print(vi)
				voxel_pos = voxels[vi]
				cell_pos_x = int(np.floor(voxel_pos[0]/cell_size))
				cell_pos_y = int(np.floor(voxel_pos[1]/cell_size))

				va = voxel_pos + np.array([0.0,0.0])
				vb = voxel_pos + np.array([0.0,voxel_size])
				vc = voxel_pos + np.array([voxel_size,0.0])
				vd = voxel_pos + np.array([voxel_size,voxel_size])

				rej_vox = False;

				for ni in range(max_figures_per_neighborhood):
					f_i =gpu_neighborhoods.get()[(cell_pos_x,cell_pos_y,ni)]
					if f_i != -1:

						fig_pos = figures[f_i]
						distances = [np.linalg.norm(fig_pos-vdi) for vdi in [va,vb,vc,vd]]
						distances = [d <= figure_radius*2 for d in distances ]
						if all(distances):
							rej_vox = True
				if rej_vox:
					voxels[(vi,0)] = -1.0
					voxels[(vi,1)] = -1.0
			"""
			voxel_indexes = (voxels != -1.0)[:,0]
			voxels = voxels[voxel_indexes,:]
			voxel_num = np.int32(voxels.size/3.0)

			#if voxel_num <10:
			#	draw(figures,figure_radius,fig_num,voxels,voxel_num,voxel_size)
			#v_list = []
			#for v in voxels:
			#	if v[0]!=-1.0:
			#		v_list.append(v)
			#voxel_num = np.int32(len(v_list))
			#voxels = np.array(v_list).astype(np.float32)
			gpu_voxels.gpudata.free()
			gpu_voxels = gpuarray.to_gpu(voxels)

			#immediate_stamp = print_time_delta("after removing voxels",immediate_stamp)
			#iteration_stamp = print_time_delta("iteration",iteration_stamp)
			draw(figures,figure_radius,fig_num,voxels,voxel_num,voxel_size,figure_disk_num,figure_radiuses,figure_distances,figure_angles)
			iteration+=1

		gpu_figures.gpudata.free()
		gpu_neighborhoods.gpudata.free()
		gpu_added_figures.gpudata.free()
		gpu_added_fig_cell_positions.gpudata.free()
		print_time_delta("TOTAL TIME AT TRIAL:"+str(trial)+"= ",total_stamp)

	end_time = print_time_delta("@ TOTAL TIME: ",init_stamp)
	print("@ AVG TRIAL TIME: ",(end_time - init_stamp)/float(number_of_trials))
	return (end_time - init_stamp)/float(number_of_trials)

fail_ratio_to_subdivide = 0.3
added_fig_num = 512*4
cell_size = 2.0
cell_num_x = 10
cell_num_y = 10
figure_radius = 1.0

number_of_trials = 1
outouts = []

for added_fig_num in [512*1]:
	outs = []
	for fail_ratio_to_subdivide in [0.4]:
		while True:
			try:
				outs.append(perform_rsas(number_of_trials,fail_ratio_to_subdivide,added_fig_num,cell_size,cell_num_x,cell_num_y,figure_radius))
				break
			except ValueError:
				print("trying again")
	outouts.append(outs)

for out in outouts:
	print("###########")
	print(out)

#ISSUE:
# debug figure rejecting vs existing
# 2 possible solutions: wrong neighborhood adding
# :wrong distance play

#more issues:
#1. there is one figure that is referenced by all/many neighborhoods
# -this figure is changed every iteration
#2. some figures are added despite colliding with figures in their neighborhoods.
# the bullshit is piling up
# HELP

#also, the voxels are not removed properly. again.
