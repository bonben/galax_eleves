#ifdef GALAX_MODEL_GPU
#include <iostream>
#include "cuda.h"
#include "kernel.cuh"
#define DIFF_T (0.1f)
#define EPS (1.0f)
#include <stdio.h>
//require computing capability 5.0 or higher

__device__ float3 operator-(const float3 &a, const float3 &b) {

  return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);

}

__device__ float3 operator+(const float3 &a, const float3 &b) {

  return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);

}



__device__ float3 operator*(const float3 &a, const float3 &b) {

  return make_float3(a.x*b.x, a.y*b.y, a.z*b.z);

}

__device__ float3 operator*(const float3 &a, const float b) {

  return make_float3(a.x*b, a.y*b, a.z*b);

}

__device__ float sum(const float3 &a) {

  return a.x + a.y + a.z;
}


__device__ void compute_difference(float3 &posi, float3 &posj, float3 &diff, float &dij)
{
	diff= posj - posi;
	dij=sum(diff*diff);
}




__device__ void compute_forces(const float &mi,const float &mj,float &dij, float &dij_mi,float &dij_mj){
    if (dij > 1)
    {
		dij=rsqrt(dij*dij*dij);
        dij_mj = dij * mj;
        dij_mi = dij * mi;
    }
    else
    {
        dij_mj = mj;
        dij_mi = mi;

    }
}

__global__ void update_acc(float3 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU,float* massesGPU,const int n_particles)
{	
	//const int number_threads=blockDim.x;
	//int thread_id=threadIdx.x;
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (i >= n_particles) return;
	float3 diff;
	float dij;
	float dij_mi;
	float dij_mj;
	printf("ok");
	
	for(int j =0; j<i; j++){ // note that j<i, ineficient on gpu
		
		compute_difference(positionsGPU[i], positionsGPU[j], diff, dij);
		compute_forces(massesGPU[i], massesGPU[j], dij, dij_mi, dij_mj);
		float3 to_add=diff*dij_mi;
		// TODO : aggregate accelerations between threads
		
	}


	printf(" acceleration : %f",accelerationsGPU[0].x);

}


// __global__ void maj_pos(float3 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU)
// {
// 	// unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

// }

static inline int divup(int a, int b) {
	// how manyœblocks of size b should we use to represent a block of size a
	return (a + b - 1)/b;
}



void update_position_cu(float3* positionsGPU,float3* velocitiesGPU, float3* accelerationsGPU, float* massesGPU, int n_particles)
{
	// n blocks on the diagonal
	// n(n-1)/2 blocks on the lower triangle
	// launch diagonal blocks first,
	// then launch full blocks

	int nthreads = 128;
	int nblocks = divup(n_particles, nthreads);
	std::cout << "nblocks: " << nblocks << std::endl;
	std::cout << "nthreads: " << nthreads << std::endl;
	std::cout << "n_particles: " << n_particles << std::endl;

	update_acc <<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU,\
	 massesGPU, n_particles);

	

}


#endif // GALAX_MODEL_GPU