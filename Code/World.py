import numpy as np
from Utils.balance_figure import balance_figure
from Utils.fig_area import fig_area
from Utils.coord_conversion import convert_figure
from CUDASourceModuleNew import get_module as get_module

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray
import math
import time
from Utils.draw import draw as draw_func
from Utils.Timer import Timer
#class for storind data for one particular world/figure configuration
#launches RSA algorithms on this world.
#single RSA data is held in RSA class
class World:
	#added_fig_num(int) = num of added figs in iteration
	#voxel_removal_treshold(float) = what percentage of voxels must FAIL at inserting to split voxels
	#figure_configuration = [(float circle_radius,float circle_x,float circle_y),...]
	#cell_size(float) = cell (and initial_voxel) size
	#world_size = (int x, int y) = num of cells in x,y
	def __init__(self,fig_config,cell_size,world_size,added_fig_num,voxel_removal_treshold):
		self.fig_disk_num = np.int32(len(fig_config))
		fig_config = np.array(fig_config)
		self.fig_radiuses = fig_config[:,0]
		self.fig_xys = fig_config[:,[1,2]]
		self.cell_size = np.float32(cell_size)
		self.cell_num_x = np.int32(world_size[0])
		self.cell_num_y = np.int32(world_size[1])
		#absolute world dimensions, not number of cells
		self.world_size_float = np.array([self.cell_num_x*self.cell_size,self.cell_num_y*self.cell_size])

		self.added_fig_num = np.int32(added_fig_num)
		self.voxel_removal_treshold = np.float32(voxel_removal_treshold)

		#fig radius is an approximation. Figure is balanced approximately, but the radius should always cover it
		self.fig_radius,self.fig_xys = balance_figure(self.fig_radiuses,self.fig_xys)
		#fig area is an approximation. (monte carlo)
		self.fig_area = fig_area(self.fig_radiuses,self.fig_xys)

		self.fig_angles,self.fig_distances = convert_figure(self.fig_xys)

		self.max_figs_per_cell = np.int32(np.ceil(((self.cell_size+(2*self.fig_radius))**2.0)/ self.fig_area))
		self.max_figs_per_neighborhood = np.int32( np.ceil( ((self.cell_size*3.0+(2*self.fig_radius))**2.0) /self.fig_area))

		self.base_voxel_depth = np.float32(2.0 * np.pi)
		#conversion for cuda friendly format
		self.fig_angles = self.fig_angles.astype(np.float32)
		self.fig_distances = self.fig_distances.astype(np.float32)
		self.fig_radiuses = self.fig_radiuses.astype(np.float32)
		self.fig_xys = self.fig_xys.astype(np.float32)
		self.fig_radius = np.float32(self.fig_radius)

		print("STARTUP: fig_angles",self.fig_angles)
		print("STARTUP: fig_distances",self.fig_distances)
		print("STARTUP: fig_radiuses",self.fig_radiuses)
		print("STARTUP: fig_radius",self.fig_radius)

		#loading cuda module
		module = get_module(self)

		#loading cuda functions
		self.init_func = module.get_function("init")
		self.gen_func = module.get_function("gen_figs")
		self.reject_v_existing_func = module.get_function("reject_figs_vs_existing")
		self.split_func = module.get_function("split_voxels")
		self.reject_voxels_func = module.get_function("reject_voxels")


	#draw = "NONE", "ITERATION", "END" when to draw
	#print_times = "NONE", "ALL", "TOTAL" if to and when to print execution time
	def perform_rsa(self,draw="NONE",print_times="NONE"):

		if not draw in ["NONE","ITERATION","END"]:
			raise ValueError("draw must be either: NONE, ITERATION or END")

		if not print_times in ["NONE","ALL","TOTAL"]:
			raise ValueError("print_times must be either: NONE, ALL or TOTAL")

		print_times_all = print_times == "ALL"
		iter_timers = []

		summary_dict = {"configuration":{
			"fig_radiuses":list(self.fig_radiuses),
			"fig_positions":list(self.fig_xys),
			"cell_num_world_size":[self.cell_num_x,self.cell_num_y],
			"cell_size":self.cell_size,
			"added_fig_num":self.added_fig_num,
			"voxel_removal_treshold":self.voxel_removal_treshold,
			"fig_area":self.fig_area,
			"fig_radius":self.fig_radius
		}}

		iterations_data = []
		voxel_fraction = 1.0

		self.initialise_rsa()
		while(self.voxel_num > 0):

			timer_iter = Timer()
			timer_iter.start_timer("iteration")

			timer_iter.start_timer("generation")
			self.generate_figs()
			g_t = timer_iter.stop_timer("generation",print_times_all)

			timer_iter.start_timer("reject_vs_existing")
			self.reject_figs_vs_existing()
			re_t = timer_iter.stop_timer("reject_vs_existing",print_times_all)

			timer_iter.start_timer("reject_vs_new")
			self.reject_figs_vs_new()
			rn_t = timer_iter.stop_timer("reject_vs_new",print_times_all)

			timer_iter.start_timer("split_voxels")
			if (1.0-(self.successfully_added_figs_num/self.added_fig_num)) > self.voxel_removal_treshold:
				self.split_voxels()
				voxel_fraction = 0.5 * voxel_fraction
			s_t = timer_iter.stop_timer("split_voxels",print_times_all)

			timer_iter.start_timer("reject_voxels")
			self.reject_voxels()
			rv_t = timer_iter.stop_timer("reject_voxels",print_times_all)

			timer_iter.stop_timer("iteration",print_times_all)
			i_t = iter_timers.append(timer_iter.get_timers())

			iteration_dict = {
				"timers":{
					"generation":g_t,
					"reject_vs_existing":re_t,
					"reject_vs_new":rn_t,
					"split_voxels":s_t,
					"reject_voxels":rv_t,
					"iteration":i_t
				},
				"data":{
					"voxel_num":self.voxel_num,
					"voxel_fraction":voxel_fraction,
					"fig_num":self.fig_num,
					"density":self.calculate_density()
				}
			}
			iterations_data.append(iteration_dict)

			if draw == "ITERATION":
				draw_func(self)

			if print_times_all:
				print("DATA: figures:",self.fig_num)
				print("DATA: voxels:",self.voxel_num)
				print("DATA: voxel_fraction:",voxel_fraction)
				print("DATA: density:",self.calculate_density())
				print("===================")

			self.iteration+=1

		if print_times == "ALL" or print_times == "TOTAL":
			print("DATA: figures:",self.fig_num)
			print("DATA: voxels:",self.voxel_num)
			print("DATA: voxel_fraction:",voxel_fraction)
			print("DATA: density:",self.calculate_density())
			total_time = sum([t["iteration"][2] for t in iter_timers ])
			name = "total"
			print(f'TIMER: {name:20s} {total_time:.20f}')

		if draw == "END":
			draw_func(self)

		final_dict = {
			"voxel_fraction":voxel_fraction,
			"fig_num":self.fig_num,
			"density":self.calculate_density()
			}
		summary_dict["iterations"] = iterations_data
		summary_dict["summary"] = final_dict
		self.finalise()
		return summary_dict
	#multiple RSAS on a single world.
	#trials_num = number of trials
	#result_json = path of the json to which the results are added
	def perform_rsas(self,trials_num,added_fig_num,voxel_removal_treshold,
		result_json="results.json",draw="NONE",print_times="NONE"):
			pass



	#HELPER FUNCTIONS:
	def translate_position(self,nparray_pos,nparray_cell_pos_int):

		out = np.copy(nparray_pos)
		neighborhood_p = nparray_cell_pos_int.astype(np.float64)*self.cell_size
		neighborhood_l = neighborhood_p - self.cell_size
		neighborhood_u = neighborhood_p + 2*self.cell_size

		lb = out<neighborhood_l
		ub = out>neighborhood_u
		out[lb]+=self.world_size_float[lb]
		out[ub]-=self.world_size_float[ub]
		return out

	#figure = [x,y,angle]
	def figure_to_xy(self,figure):
		rotated_angles = self.fig_angles + figure[2]
		xs = self.fig_distances * np.cos(rotated_angles) + figure[0]
		ys = self.fig_distances * np.sin(rotated_angles) + figure[1]
		return np.column_stack((xs,ys))

	#returns 1 if they collide
	#fig_1_xy,fig_2_xy = outputs of figure_to_xy(...)
	def figure_collide(self,fig_1_xy,fig_2_xy):
		for n1,d1 in enumerate(fig_1_xy):
			for n2,d2 in enumerate(fig_2_xy):
				if np.linalg.norm(d1-d2) < self.fig_radiuses[n1]+self.fig_radiuses[n2]:
					return True
		return False


	def calculate_density(self):
		fig_areas = self.fig_num * self.fig_area
		world_area = self.cell_num_x * self.cell_num_y
		return (fig_areas / world_area)

	#ACTUAL-DOING-STUFF FUNCTIONS

	def initialise_rsa(self):
		#variables
		self.voxel_size = self.cell_size
		self.voxel_depth = self.base_voxel_depth
		self.voxel_num = np.int32(self.cell_num_x * self.cell_num_y)

		#data structures
		cells = (np.ones((self.cell_num_x,self.cell_num_y,self.max_figs_per_cell)) * (-1)).astype(np.int32)
		neighborhoods = (np.ones((self.cell_num_x,self.cell_num_y,self.max_figs_per_neighborhood)) * (-1)).astype(np.int32)
		#voxels->COLUMNS: X, Y, ANGLE:DEPTH
		voxels = []
		for x in range(self.cell_num_x):
			for y in range(self.cell_num_y):
				voxels.append([float(x)*self.cell_size,float(y)*self.cell_size,0.0])
		voxels = np.array(voxels).astype(np.float32)
		#figs,added_figs->COLUMNS: X,Y,ANGLE
		figs = np.array([[-50.0,-50.0,0.0]]).astype(np.float32)
		added_figs = np.zeros((self.added_fig_num,2)).astype(np.float32)
		self.fig_num = np.int32(0)

		#initialising pycuda kernels
		seed = np.int32(time.time())
		self.init_func(seed, block=(512,1,1), grid=(math.ceil(self.added_fig_num/512),1))

		#initialising cuda data
		self.gpu_figs = gpuarray.to_gpu(figs)
		self.gpu_neighborhoods = gpuarray.to_gpu(neighborhoods)
		self.gpu_voxels = gpuarray.to_gpu(voxels)
		self.gpu_added_figs = gpuarray.zeros((self.added_fig_num,3),np.float32)
		self.gpu_added_fig_cell_positions = gpuarray.zeros((self.added_fig_num,2),np.int32)
		self.iteration = 0

	def generate_figs(self):
		self.gen_func(
			self.gpu_voxels,
			self.voxel_size,
			self.voxel_depth,
			self.voxel_num,
			self.gpu_added_figs,
			block=(512,1,1),
			grid=(math.ceil(self.added_fig_num/512),1))


	def reject_figs_vs_existing(self):
		self.reject_v_existing_func(
			self.gpu_figs,
			self.fig_num,
			self.gpu_added_figs,
			self.gpu_added_fig_cell_positions,
			self.gpu_neighborhoods,
			block=(512,1,1),
			grid=(math.ceil(self.added_fig_num/512),1))


	def reject_figs_vs_new(self):
		f_list = []
		added_figs = self.gpu_added_figs.get()

		added_fig_cell_positions = self.gpu_added_fig_cell_positions.get()
		neighborhoods = self.gpu_neighborhoods.get()

		counter = 0
		rejected_figs = 0

		added_fig_indexes = (added_figs != -1.0)[:,0]
		added_figs = added_figs[added_fig_indexes,:]
		added_fig_cell_positions = added_fig_cell_positions[added_fig_indexes,:]

		#print("figs that passed rejecting by existing",added_figs.size/3)

		pseudo_neighborhoods = [[[] for y in range(self.cell_num_y+1)] for x in range(self.cell_num_x+1)]

		fr2 = self.fig_radius * 2.0

		for index,figure in enumerate(added_figs):
			figuree = np.array((figure[0],figure[1]))
			cell_pos = added_fig_cell_positions[index]
			checked_figs = pseudo_neighborhoods[cell_pos[0]][cell_pos[1]]
			#print("checked_f",checked_figs)
			addfig=True

			for fig in checked_figs:
				figg = self.translate_position(np.array([fig[0],fig[1]]),np.array(cell_pos))

				fig_1_xy = self.figure_to_xy(np.array([figuree[0],figuree[1],figure[2]]))
				fig_2_xy = self.figure_to_xy(np.array([figg[0],figg[1],fig[2]]))
				if self.figure_collide(fig_1_xy,fig_2_xy):
					addfig = False

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
						n[0] = self.cell_num_x-1
					if n[0] >= self.cell_num_x:
						n[0] = 0
					if n[1] < 0:
						n[1] = self.cell_num_y-1
					if n[1] >= self.cell_num_y:
						n[1] = 0
					#if not (n[0]>=cell_num_x or n[1]>=cell_num_y or n[0]<0 or n[1]<0):
					t_array.append(tuple(n))
				t_array = np.array(t_array)

				for n in t_array:
					pseudo_neighborhoods[n[0]][n[1]].append(figure)
					for i in range(self.max_figs_per_neighborhood):
						if neighborhoods[(n[0],n[1],i)] == -1:
							neighborhoods[(n[0],n[1],i)] = np.int32(counter+self.fig_num)
							break
				counter +=1

		self.gpu_neighborhoods.gpudata.free()
		self.gpu_neighborhoods = gpuarray.to_gpu(neighborhoods)
		self.fig_num += np.int32(len(f_list))
		if self.iteration == 0:
			self.figs = np.array(f_list).astype(np.float32)
		elif len(f_list) != 0:
			self.figs = np.concatenate((self.figs,np.array(f_list).astype(np.float32)))
		#print("figs after rejecting:",self.figs.shape)
		self.gpu_figs.gpudata.free()
		self.gpu_figs = gpuarray.to_gpu(self.figs)
		self.successfully_added_figs_num = counter


	def split_voxels(self):
		gpu_target_voxels = gpuarray.zeros((self.voxel_num*8,3),np.float32)

		self.split_func(
			self.gpu_voxels,
			self.voxel_num,
			self.voxel_size,
			self.voxel_depth,
			gpu_target_voxels,
			block=(512,1,1),
			grid=(math.ceil(self.voxel_num/512),1))

		self.gpu_voxels = gpu_target_voxels
		#print("before gpu_voxels",gpu_voxels.get().size)

		self.voxel_depth = np.float32(self.voxel_depth/2.0)
		self.voxel_num = np.int32(self.voxel_num * 8)
		self.voxel_size = np.float32(self.voxel_size/2.0)


	def reject_voxels(self):
		self.reject_voxels_func(
			self.gpu_voxels,
			self.voxel_num,
			self.voxel_size,
			self.voxel_depth,
			self.gpu_figs,
			self.gpu_neighborhoods,
			block=(512,1,1),
			grid=(math.ceil(self.voxel_num/512),1))

		#print("voxel size:",self.voxel_size)
		#print("gpu_voxels:",self.gpu_voxels.size)
		voxels = self.gpu_voxels.get()

		voxel_indexes = (voxels != -1.0)[:,0]
		voxels = voxels[voxel_indexes,:]
		self.voxel_num = np.int32(voxels.size/3.0)

		self.gpu_voxels.gpudata.free()
		self.gpu_voxels = gpuarray.to_gpu(voxels)


	def finalise(self):
		self.gpu_figs.gpudata.free()
		self.gpu_neighborhoods.gpudata.free()
		self.gpu_added_figs.gpudata.free()
		self.gpu_added_fig_cell_positions.gpudata.free()

#figure configs for future use:
#all fit into 1.0 cell size
# the pseudo-single
config_1 = [(0.5,0.0,0.0)]
# the basic double OO
config_2 = [(0.25,0.0,0.0),(0.25,0.5,0.0)]
# the basic quadruple oooo
config_3 = [(0.125,0.0,0.0),(0.125,0.25,0.0),(0.125,0.5,0.0),(0.125,0.75,0.0)]
# the intersecting double
config_4 = [(0.4,0.0,0.0),(0.4,0.2,0.0)]
# the size-difference double
config_5 = [(0.3,0.0,0.0),(0.1,0.4,0.0)]
# the end-chain
config_6 = [(0.125,0.0,0.0),(0.0625,0.0625*3,0.0),(0.0625,0.0625*5,0.0), (0.0625,0.0625*7,0.0),(0.0625,0.0625*9,0.0),(0.125,0.75,0.0)]
# the middle 3-ball
config_7 = [(0.125,-0.125-0.25,0.0),(0.25,0.0,0.0),(0.125,0.125+0.25,0.0)]

w = World(config_5,1.0,(50,50),512*4,0.5)
w.perform_rsa(draw="NONE",print_times="ALL")
