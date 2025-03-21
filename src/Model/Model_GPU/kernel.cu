#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"
#define DIFF_T (0.1f)
#define EPS (1.0f)

__global__ void compute_acc(float4 * positionsGPU, float4 * velocitiesGPU, float4 * accelerationsGPU, int n_particles)
{
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        float3 acc = {0.0f, 0.0f, 0.0f};
        if (i >= n_particles ){
                return;
        }
        for (int j = 0; j < n_particles; j++)
        {

                const float diffx = positionsGPU[j].x - positionsGPU[i].x;
                const float diffy = positionsGPU[j].y - positionsGPU[i].y;
                const float diffz = positionsGPU[j].z - positionsGPU[i].z;

                float dij = diffx * diffx + diffy * diffy + diffz * diffz ;
                

                dij = max(1.0f, dij);

                dij = rsqrtf(dij);
                dij = 10.0 * (dij * dij * dij);
                float temp = positionsGPU[j].w * dij;

                acc.x += diffx * temp;
                acc.y += diffy * temp;
                acc.z += diffz * temp;
                
        }
        accelerationsGPU[i].x = acc.x;
        accelerationsGPU[i].y = acc.y;  
        accelerationsGPU[i].z = acc.z;
}

__global__ void maj_pos(float4 * positionsGPU, float4 * velocitiesGPU, float4 * accelerationsGPU, int n_particles)
{
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_particles ){
                return;
        }

        velocitiesGPU[i].x += accelerationsGPU[i].x * 2.0f;
        velocitiesGPU[i].y += accelerationsGPU[i].y * 2.0f;
        velocitiesGPU[i].z += accelerationsGPU[i].z * 2.0f;
        positionsGPU[i].x += velocitiesGPU[i].x * 0.1f;
        positionsGPU[i].y += velocitiesGPU[i].y * 0.1f;
        positionsGPU[i].z += velocitiesGPU[i].z * 0.1f;

}

void update_position_cu(float4* positionsGPU, float4* velocitiesGPU, float4* accelerationsGPU, int n_particles)
{
        int nthreads = 256;
        int nblocks =  (n_particles + (nthreads -1)) / nthreads;

        compute_acc<<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU,  n_particles);
        maj_pos    <<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, n_particles);
}

#endif //GALAX_MODEL_GPU
