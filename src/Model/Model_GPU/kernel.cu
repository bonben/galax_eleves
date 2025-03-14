#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"
#define DIFF_T (0.1f)
#define EPS (1.0f)

__global__ void compute_acc(float4 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU, int n_particles)
{
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_particles ){
                return;
        }
        for (int j = 0; j < n_particles; j++)
        {

                        const float diffx = positionsGPU[j].x - positionsGPU[i].x;
                        const float diffy = positionsGPU[j].y - positionsGPU[i].y;
                        const float diffz = positionsGPU[j].z - positionsGPU[i].z;

                        float dij = diffx * diffx + diffy * diffy + diffz * diffz + EPS;
                        dij = std::sqrt(dij);
                        dij = 10.0 / (dij * dij * dij);
                        

                        accelerationsGPU[i].x += diffx * dij * positionsGPU[j].w;
                        accelerationsGPU[i].y += diffy * dij * positionsGPU[j].w;
                        accelerationsGPU[i].z += diffz * dij * positionsGPU[j].w;
                
        }




//         int base = i*20+j*14 ;
//         int end = base + 1<<14;
//         if (n >= end ){
//                 return;}
}

__global__ void maj_pos(float4 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU, int n_particles)
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

void update_position_cu(float4* positionsGPU, float3* velocitiesGPU, float3* accelerationsGPU, int n_particles)
{
        int nthreads = 256;
        int nblocks =  (n_particles + (nthreads -1)) / nthreads;

        compute_acc<<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU,  n_particles);
        maj_pos    <<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, n_particles);
}

#endif //GALAX_MODEL_GPU


// #endif // GALAX_MODEL_GPU



// #include <cstdlib>
// #include <iostream>
// #include <cuda_runtime.h>
// #include <vector>
// #include <random>
// #include <algorithm>
// #include <chrono>

// static inline void check(cudaError_t err, const char* context) {
//         if (err != cudaSuccess) {
//                 std::cerr << "CUDA error: " << context << ": "
//                         << cudaGetErrorString(err) << std::endl;
//                 std::exit(EXIT_FAILURE);
//         }
// }

// #define CHECK(x) check(x, #x)

// __global__ void mykernel(float* cGPU,float* aGPU,float* bGPU, int n) {
//         int j = threadIdx.x;
//     int i = blockIdx.y;
//         int base = i*20+j*14 ;
//         int end = base + 1<<14;
//         if (n >= end ){
//                 return;}
        

//         for (int i = base; i < end ; i++) {
//                 cGPU[i] = aGPU[i] + bGPU[i];
//         }

// }


// int main() {
//         int n = 50000000;
//         int seed = 0;

//         constexpr int blocks = 1 << 6;
//         constexpr int threads = 1 << 6;

//         std::mt19937 gen(seed);
//         std::uniform_real_distribution<> dis(1.0, 8.0);

//         std::vector<float> a(n);
//         std::vector<float> b(n);
//         std::vector<float> c(n);

//         for (int i = 0; i < a.size(); i++)
//         {
//                 a[i] = dis(gen);
//                 b[i] = dis(gen);
//         }

//         // allocate memory in gpu
//         float* aGPU = NULL;
//     CHECK(cudaMalloc((void**)&aGPU, n* sizeof(float)));    
        
//     float* bGPU = NULL;
//     CHECK(cudaMalloc((void**)&bGPU, n * sizeof(float)));
        
//         float* cGPU = NULL;
//     CHECK(cudaMalloc((void**)&cGPU, n * sizeof(float)));


//         auto t1 = std::chrono::high_resolution_clock::now();

//         // transfer a & b to GPU memory
//         CHECK(cudaMemcpy(aGPU, a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
//         CHECK(cudaMemcpy(bGPU, b.data(), n * sizeof(float), cudaMemcpyHostToDevice));

//         // do the magic in GPU
//         mykernel<<<blocks, threads>>>(cGPU,aGPU,bGPU,n);

//         // get c into CPU memory
//         cudaMemcpy(c.data(), cGPU, n * sizeof(float) ,cudaMemcpyDeviceToHost);

//         // free allocated GPU memory
//     CHECK(cudaFree(aGPU));
//     CHECK(cudaFree(bGPU));
//         CHECK(cudaFree(cGPU));

//         auto t2 = std::chrono::high_resolution_clock::now();

//         std::chrono::duration<float,std::milli> duration_gpu = t2 - t1;
//         std::cout << "Duration GPU: " << duration_gpu.count() << std::endl;

//         auto t3 = std::chrono::high_resolution_clock::now();

//         for (int i = 0; i < n; i++)
//                 c[i] = a[i] + b[i];

//         auto t4 = std::chrono::high_resolution_clock::now();

//         std::chrono::duration<float,std::milli> duration_cpu = t4 - t3;
//         std::cout << "Duration CPU: " << duration_cpu.count() << std::endl;

//         // what's the fastest ?
// }