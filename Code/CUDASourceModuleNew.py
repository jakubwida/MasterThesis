from pycuda.compiler import SourceModule

def get_module(world):
	return SourceModule("""

    #include <cmath>
    #include <curand_kernel.h>

    //constants

    const int added_fig_num = """+str(world.added_fig_num)+""";
    const float cell_size = """+str(world.cell_size)+""";
    const int cell_num_x = """+str(world.cell_num_x)+""";
    const int cell_num_y = """+str(world.cell_num_y)+""";

    const float figure_radius = """+str(world.fig_radius)+""";
    const int max_figures_per_cell = """+str(world.max_figs_per_cell)+""";
    const int max_figures_per_neighborhood = """+str(world.max_figs_per_neighborhood)+""";

    const float vrt = 0.0001;
    __device__ curandState_t states[added_fig_num];

 	const int figure_disk_num = """+str(world.fig_disk_num)+""";
	__device__ float figure_radiuses[figure_disk_num] = {"""+','.join(map(str,world.fig_radiuses.tolist()))+"""};
	__device__ float figure_distances[figure_disk_num] = {"""+','.join(map(str,world.fig_distances.tolist()))+"""};
	__device__ float figure_angles[figure_disk_num] = {"""+','.join(map(str,world.fig_angles.tolist()))+"""};

    extern "C" {

        //helper functions ====================================

        __global__ void init(int seed)
        {
            int tidx = threadIdx.x + blockIdx.x * blockDim.x;

            if (tidx < added_fig_num) {

				curand_init(seed, tidx, 0, &states[tidx]);

                //curandState_t* s = new curandState_t;
                //if (s != 0) {
                //    curand_init(seed, tidx, 0, s);
                //}
                //states[tidx] = s;
            }
        }

        __device__ int array_index_3d(int x, int y, int z, int y_size, int z_size) {
            return x * (z_size * y_size)  + z_size * y +  z;
        }

        __device__ int array_index_2d(int x, int y, int y_size) {
            return x*y_size + y;
        }

        __device__ float get_distance(float x1, float y1, float x2, float y2) {
			float dx = (x1-x2);
			float dy = (y1-y2);
			float dist = sqrt(dx*dx + dy*dy);
			return dist;
            //return( hypot(x1 - x2, y1 - y2) );
        }

		__device__ void min_max_cos(float angle,float angle_delta,float * min_p,float * max_p){

			float PI2 = 2.0 *M_PI;

			float t_angle = fmod(angle + PI2,PI2);
			float v1 = cos(t_angle);
			float v2 = cos(t_angle+angle_delta);

			min_p[0] = min(v1,v2);
			max_p[0] = max(v1,v2);

			bool cond1 = (t_angle < 0.0 && t_angle + angle_delta > 0.0) || (t_angle < PI2 && t_angle+angle_delta > PI2);
			bool cond2 = t_angle < M_PI && t_angle + angle_delta > M_PI;
			max_p[0] = cond1 ? 1.0 : max_p[0];
 			min_p[0] = (!cond1 && cond2) ? -1.0 : min_p[0];

		}

		__device__ void min_max_sin(float angle,float angle_delta,float * min_p,float * max_p){

			float PI0_5 = 0.5 *M_PI;
			float PI1_5 = 1.5 *M_PI;
			float PI2 = 2.0 *M_PI;

			float t_angle = fmod(angle + PI2,PI2);
			float v1 = sin(t_angle);
			float v2 = sin(t_angle+angle_delta);

			min_p[0] = min(v1,v2);
			max_p[0] = max(v1,v2);

			bool cond1 = (t_angle < PI0_5 && t_angle + angle_delta > PI0_5);
			bool cond2 = t_angle < PI1_5 && t_angle + angle_delta > PI1_5;
			max_p[0] = cond1 ? 1.0 : max_p[0];
 			min_p[0] = (!cond1 && cond2) ? -1.0 : min_p[0];

		}

		//takes the figure x,y,angle, returns positions of figure circles in target array, which must be [figure_disk_num,2] sized
		__device__ void get_figure_xy(float x,float y,float angle,float * target_array){
			for(int i=0; i<figure_disk_num;i++) {
				target_array[array_index_2d(i,0,2)]= figure_distances[i] * cos(angle+figure_angles[i]) + x;
				target_array[array_index_2d(i,1,2)]= figure_distances[i] * sin(angle+figure_angles[i]) + y;
			}
		}


		//returns true if voxel is to be rejected
		/*
		__device__ bool reject_voxel(float voxel_size,float voxel_pos_x,float voxel_pos_y, float fig_pos_x, float fig_pos_y){

			float radius_against = 2*figure_radius;
            bool dist0 = (get_distance(fig_pos_x,fig_pos_y,voxel_pos_x,voxel_pos_y) <= (radius_against));
            bool dist1 = (get_distance(fig_pos_x,fig_pos_y,voxel_pos_x+voxel_size,voxel_pos_y) <= (radius_against));
            bool dist2 = (get_distance(fig_pos_x,fig_pos_y,voxel_pos_x,voxel_pos_y+voxel_size) <= (radius_against));
            bool dist3 = (get_distance(fig_pos_x,fig_pos_y,voxel_pos_x+voxel_size,voxel_pos_y+voxel_size) <= (radius_against));

			return (dist0 && dist1 && dist2 && dist3);
		}
		*/
		//voxel depth = how deep is voxel currently
		//voxel angle = at what depth is this voxel
		__device__ bool reject_voxel(float voxel_size,float voxel_pos_x,float voxel_pos_y,
		float voxel_depth, float voxel_angle,
		float fig_pos_x, float fig_pos_y, float figure_angle){


			//voxel_pos_x = 0.0;
			//voxel_pos_y = 0.0;

			bool is_rejected = false;

			for(int i=0;i<figure_disk_num;i++){
				//i = index of disk already existing within voxel


				for(int j=0;j<figure_disk_num;j++){
					//j=index of a disk of a virtual figure 'added' to the voxel

					float min_cos[1];
					float max_cos[1];

					float min_sin[1];
					float max_sin[1];

					min_max_cos(figure_angles[j]+voxel_angle,voxel_depth,min_cos,max_cos);
					min_max_sin(figure_angles[j]+voxel_angle,voxel_depth,min_sin,max_sin);

					float min_max_x_0 = voxel_pos_x + (figure_distances[j] * min_cos[0]);
					float min_max_x_1 = voxel_pos_x + (figure_distances[j] * max_cos[0]) + voxel_size;

					float min_max_y_0 = voxel_pos_y + (figure_distances[j] * min_sin[0]);
					float min_max_y_1 = voxel_pos_y + (figure_distances[j] * max_sin[0]) + voxel_size;


					float max_distance_2 = 0.0;
					float x_0 = fig_pos_x + figure_distances[i] * cos(figure_angles[i]+figure_angle);

						//if cond1
					bool cond1 = x_0 < min_max_x_0;
					max_distance_2 += cond1 ? (x_0-min_max_x_1)*(x_0-min_max_x_1) : 0.0;
						//else if cond2
					bool cond2 = (x_0 >= min_max_x_0 && x_0 <= min_max_x_1);
					max_distance_2 += (!cond1 && cond2) ? max((x_0-min_max_x_0)*(x_0-min_max_x_0), (x_0-min_max_x_1)*(x_0-min_max_x_1)) : 0.0;
						//else
					max_distance_2 += (!cond1 && !cond2) ? (x_0-min_max_x_0)*(x_0-min_max_x_0) : 0.0;

					float y_0 = fig_pos_y + figure_distances[i] * sin(figure_angles[i]+figure_angle);

						//if cond3
					bool cond3 = y_0 < min_max_y_0;
					max_distance_2 += cond3 ? (y_0-min_max_y_1)*(y_0-min_max_y_1) : 0.0;
						//else if cond4
					bool cond4 = (y_0 >= min_max_y_0 && y_0 <= min_max_y_1);
					max_distance_2 += (!cond3 && cond4) ? max((y_0-min_max_y_0)*(y_0-min_max_y_0), (y_0-min_max_y_1)*(y_0-min_max_y_1)) : 0.0;
						//else
					max_distance_2 += (!cond3 && !cond4) ? (y_0-min_max_y_0)*(y_0-min_max_y_0) : 0.0;

					max_distance_2 -= (figure_radiuses[i] + figure_radiuses[j])*(figure_radiuses[i] + figure_radiuses[j]);
					is_rejected = is_rejected ? true : (max_distance_2<0.0);
				}
			}

			return is_rejected;
		}

		//TODO: convert this to be able to use multiple figures

		//returns true if figures collide with each other
		__device__ bool figures_collide(float fig_1_pos_x,float fig_1_pos_y, float fig_1_angle, float fig_2_pos_x, float fig_2_pos_y, float fig_2_angle){

			//float fig_dist = get_distance(fig_1_pos_x,fig_1_pos_y,fig_2_pos_x,fig_2_pos_y);
			//return fig_dist < (2*figure_radius);

			bool is_collision = false;
			float x1;
			float x2;
			float y1;
			float y2;
			float rad1;
			float rad2;

			float fig_1_xy[figure_disk_num*2];
			float fig_2_xy[figure_disk_num*2];
			get_figure_xy(fig_1_pos_x,fig_1_pos_y,fig_1_angle,fig_1_xy);
			get_figure_xy(fig_2_pos_x,fig_2_pos_y,fig_2_angle,fig_2_xy);

			for(int i = 0; i<figure_disk_num;i++) {
				x1 = fig_1_xy[array_index_2d(i,0,2)];
				y1 = fig_1_xy[array_index_2d(i,1,2)];
				rad1 = figure_radiuses[i];

				for(int j = 0; j<figure_disk_num;j++) {
					x2 = fig_2_xy[array_index_2d(j,0,2)];
					y2 = fig_2_xy[array_index_2d(j,1,2)];
					rad2 = figure_radiuses[j];

					is_collision = (get_distance(x1,y1,x2,y2) < rad1+rad2) || is_collision;
				}
			}
			return is_collision;
		}

		//translates the position of figure within the neighborhood, for the purposes of periodical edge conditions
		//does that for a single axis
		__device__ float translate_position(float fig_pos,int cell_pos, int cell_num){
			float lower = (cell_pos*cell_size) - cell_size;
			float upper = (cell_pos*cell_size) + (2.0*cell_size);
			float world_s = cell_num*cell_size;

			float out = fig_pos < lower ? fig_pos + world_s : fig_pos;
			out = fig_pos > upper ? fig_pos - world_s : out;
			return out;
		}

		__device__ float device_fmod(float a, float b){
			return fmod(a,b);
		}


        //target functions ====================================

        __global__ void gen_figs(
            float *voxel_positions, float voxel_size, float voxel_depth, int voxel_num,
            float *added_figures
            ) {

            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int voxel_index;
            float fig_pos_x;
            float fig_pos_y;
			float fig_angle;
            if(idx<added_fig_num){

                curandState_t s = states[idx];

				//curand_uniform(&s);
                voxel_index  = floor((float)voxel_num * device_fmod(curand_uniform(&s),1.0));

                fig_pos_x = voxel_positions[array_index_2d(voxel_index,0,3)] + curand_uniform(&s) * voxel_size;
                fig_pos_y = voxel_positions[array_index_2d(voxel_index,1,3)] + curand_uniform(&s) * voxel_size;
				fig_angle = voxel_positions[array_index_2d(voxel_index,2,3)] + curand_uniform(&s) * voxel_depth;
		        //fig_pos_x = 5.0;
		        //fig_pos_y = 5.0;
				//fig_angle = 5.0;

                states[idx] = s;

                added_figures[array_index_2d(idx,0,3)] = fig_pos_x;
                added_figures[array_index_2d(idx,1,3)] = fig_pos_y;
                added_figures[array_index_2d(idx,2,3)] = fig_angle;
            }
        }

        __global__ void reject_figs_vs_existing(
            float * figures,int fig_num,
            float * added_figures,int * added_fig_cell_positions,
            int *neighborhoods){

            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            float fig_pos_x;
            float fig_pos_y;
			float fig_angle;
            int cell_pos_x;
            int cell_pos_y;
            int checked_fig_index;
            float checked_fig_x;
            float checked_fig_y;
			float checked_fig_angle;
            //float checked_fig_dist;
            int add_fig = 1;
			int reject_fig = 0;

            if(idx<added_fig_num){

                fig_pos_x = added_figures[array_index_2d(idx,0,3)];
                fig_pos_y = added_figures[array_index_2d(idx,1,3)];
				fig_angle = added_figures[array_index_2d(idx,2,3)];

                cell_pos_x = floor(fig_pos_x/cell_size);
                cell_pos_y = floor(fig_pos_y/cell_size);
                added_fig_cell_positions[array_index_2d(idx,0,2)] = cell_pos_x;
                added_fig_cell_positions[array_index_2d(idx,1,2)] = cell_pos_y;

                for(int z=0; z<max_figures_per_neighborhood;z++){
                    checked_fig_index = neighborhoods[array_index_3d(cell_pos_x,cell_pos_y,z,cell_num_y,max_figures_per_neighborhood)];
                    checked_fig_x = checked_fig_index == -1.0 ? 0.0 : figures[array_index_2d(checked_fig_index,0,3)];
                    checked_fig_y = checked_fig_index == -1.0 ? 0.0 : figures[array_index_2d(checked_fig_index,1,3)];
					checked_fig_angle = checked_fig_index == -1.0 ? 0.0 : figures[array_index_2d(checked_fig_index,2,3)];
                    //checked_fig_dist = get_distance(checked_fig_x,checked_fig_y,fig_pos_x,fig_pos_y);

					//periodical edge conditions figure pos translation
					checked_fig_x = translate_position(checked_fig_x,cell_pos_x,cell_num_x);
					checked_fig_y = translate_position(checked_fig_y,cell_pos_y,cell_num_y);


					reject_fig = (checked_fig_index != -1.0) && figures_collide(checked_fig_x,checked_fig_y,checked_fig_angle,fig_pos_x,fig_pos_y,fig_angle);

                    //key expression
                    add_fig = !add_fig ? 0 : !reject_fig;

                    //TEMP
                    //neighborhoods[array_index_3d(cell_pos_x,cell_pos_y,z,cell_num_y,max_figures_per_neighborhood)]=(int)checked_fig_dist;

                }
                added_figures[array_index_2d(idx,0,3)] = (add_fig ? fig_pos_x : -1.0);
                added_figures[array_index_2d(idx,1,3)] = (add_fig ? fig_pos_y : -1.0);
				added_figures[array_index_2d(idx,2,3)] = (add_fig ? fig_angle : -1.0);
            }
        }

		//does not split voxel angle currently
        __global__ void split_voxels(
            float * current_voxels,
            int current_voxel_num,
            float current_voxel_size,
			float voxel_depth,
            float *target_voxels) {

            float voxel_pos_x;
            float voxel_pos_y;
			float voxel_angle;
            float added_size = current_voxel_size/2.0;
			float added_depth = voxel_depth/2.0;
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int v_idx = idx*8;

            if(idx < current_voxel_num) {
                voxel_pos_x = current_voxels[array_index_2d(idx,0,3)];
                voxel_pos_y = current_voxels[array_index_2d(idx,1,3)];
				voxel_angle = current_voxels[array_index_2d(idx,2,3)];

                target_voxels[array_index_2d(v_idx,0,3)] = voxel_pos_x;
                target_voxels[array_index_2d(v_idx,1,3)] = voxel_pos_y;
                target_voxels[array_index_2d(v_idx,2,3)] = voxel_angle;

                target_voxels[array_index_2d(v_idx+1,0,3)] = voxel_pos_x + added_size;
                target_voxels[array_index_2d(v_idx+1,1,3)] = voxel_pos_y;
                target_voxels[array_index_2d(v_idx+1,2,3)] = voxel_angle;

                target_voxels[array_index_2d(v_idx+2,0,3)] = voxel_pos_x;
                target_voxels[array_index_2d(v_idx+2,1,3)] = voxel_pos_y + added_size;
                target_voxels[array_index_2d(v_idx+2,2,3)] = voxel_angle;

                target_voxels[array_index_2d(v_idx+3,0,3)] = voxel_pos_x + added_size;
                target_voxels[array_index_2d(v_idx+3,1,3)] = voxel_pos_y + added_size;
                target_voxels[array_index_2d(v_idx+3,2,3)] = voxel_angle;

                target_voxels[array_index_2d(v_idx+4,0,3)] = voxel_pos_x;
                target_voxels[array_index_2d(v_idx+4,1,3)] = voxel_pos_y;
                target_voxels[array_index_2d(v_idx+4,2,3)] = voxel_angle + added_depth;

                target_voxels[array_index_2d(v_idx+5,0,3)] = voxel_pos_x + added_size;
                target_voxels[array_index_2d(v_idx+5,1,3)] = voxel_pos_y;
                target_voxels[array_index_2d(v_idx+5,2,3)] = voxel_angle + added_depth;

                target_voxels[array_index_2d(v_idx+6,0,3)] = voxel_pos_x;
                target_voxels[array_index_2d(v_idx+6,1,3)] = voxel_pos_y + added_size;
                target_voxels[array_index_2d(v_idx+6,2,3)] = voxel_angle + added_depth;

                target_voxels[array_index_2d(v_idx+7,0,3)] = voxel_pos_x + added_size;
                target_voxels[array_index_2d(v_idx+7,1,3)] = voxel_pos_y + added_size;
                target_voxels[array_index_2d(v_idx+7,2,3)] = voxel_angle + added_depth;
            }

        }
        __global__ void reject_voxels(
            float * voxels, int voxel_num,float voxel_size,float voxel_depth,
            float * figures, int *neighborhoods) {

            float voxel_pos_x;
            float voxel_pos_y;
			float voxel_angle;
            int cell_pos_x;
            int cell_pos_y;
            int checked_fig_index;
            float checked_fig_x;
            float checked_fig_y;
            float checked_fig_angle;

            int reject;

            //float radius_against = 2*(figure_radius+figure_radius*vrt);
            float radius_against = figure_radius*2.0;

            int do_reject_voxel = 0;

            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < voxel_num) {
                voxel_pos_x = voxels[array_index_2d(idx,0,3)];
                voxel_pos_y = voxels[array_index_2d(idx,1,3)];
				voxel_angle = voxels[array_index_2d(idx,2,3)];
                cell_pos_x = floor((voxel_pos_x)/cell_size);
                cell_pos_y = floor((voxel_pos_y)/cell_size);

                for(int z=0;z<max_figures_per_neighborhood;z++){
                    checked_fig_index = neighborhoods[array_index_3d(cell_pos_x,cell_pos_y,z,cell_num_y,max_figures_per_neighborhood)];
                    checked_fig_x = checked_fig_index == -1.0 ? 0.0 : figures[array_index_2d(checked_fig_index,0,3)];
                    checked_fig_y = checked_fig_index == -1.0 ? 0.0 : figures[array_index_2d(checked_fig_index,1,3)];
                    checked_fig_angle = checked_fig_index == -1.0 ? 0.0 : figures[array_index_2d(checked_fig_index,2,3)];

					//periodical edge conditions figure pos translation
					checked_fig_x = checked_fig_index == -1.0 ? 0.0 : translate_position(checked_fig_x,cell_pos_x,cell_num_x);
					checked_fig_y = checked_fig_index == -1.0 ? 0.0 : translate_position(checked_fig_y,cell_pos_y,cell_num_y);

					//reject = reject_voxel(voxel_size,voxel_pos_x,voxel_pos_y,checked_fig_x,checked_fig_y);

					reject = reject_voxel(voxel_size,voxel_pos_x, voxel_pos_y,voxel_depth, voxel_angle,checked_fig_x, checked_fig_y,checked_fig_angle);

					do_reject_voxel = (reject && checked_fig_index != -1.0) || do_reject_voxel;

                }

                //issue: rejects all voxels
                voxels[array_index_2d(idx,0,3)] = (!do_reject_voxel ? voxel_pos_x : -1.0);
                voxels[array_index_2d(idx,1,3)] = (!do_reject_voxel ? voxel_pos_y : -1.0);
				voxels[array_index_2d(idx,2,3)] = (!do_reject_voxel ? voxel_angle : -1.0);
            }
        }
    }



    """, no_extern_c=True)
