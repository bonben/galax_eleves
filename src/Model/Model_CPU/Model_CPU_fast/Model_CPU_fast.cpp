#ifdef GALAX_MODEL_CPU_FAST

#include <cmath>

#include "Model_CPU_fast.hpp"
#include <xsimd/xsimd.hpp>
#include <omp.h>

#define SERIAL 0
#define PARFOR_NAIVE 1
#define SERIAL_IMPROVED 2
#define PARFOR_ATOMIC 3
#define PARFOR_REDUCE 4
#define STD_ATOMIC 5

#define STRATEGY SERIAL_IMPROVED

namespace xs = xsimd;
using b_type = xs::batch<float, xs::avx2>;



Model_CPU_fast
::Model_CPU_fast(const Initstate& initstate, Particles& particles)
: Model_CPU(initstate, particles)
{
}
void Model_CPU_fast
::step()
{
    std::fill(accelerationsx.begin(), accelerationsx.end(), 0);
    std::fill(accelerationsy.begin(), accelerationsy.end(), 0);
    std::fill(accelerationsz.begin(), accelerationsz.end(), 0);

#if STRATEGY == SERIAL 
for (int i = 0; i < n_particles; i++)
	{
		for (int j = 0; j < n_particles; j++)
		{
			if(i != j)
			{
				const float diffx = particles.x[j] - particles.x[i];
				const float diffy = particles.y[j] - particles.y[i];
				const float diffz = particles.z[j] - particles.z[i];

				float dij = diffx * diffx + diffy * diffy + diffz * diffz;

				if (dij < 1.0)
				{
					dij = 10.0;
				}
				else
				{
					dij = std::sqrt(dij);
					dij = 10.0 / (dij * dij * dij);
				}

				accelerationsx[i] += diffx * dij * initstate.masses[j];
				accelerationsy[i] += diffy * dij * initstate.masses[j];
				accelerationsz[i] += diffz * dij * initstate.masses[j];
			}
		}
	}

	for (int i = 0; i < n_particles; i++)
	{
		velocitiesx[i] += accelerationsx[i] * 2.0f;
		velocitiesy[i] += accelerationsy[i] * 2.0f;
		velocitiesz[i] += accelerationsz[i] * 2.0f;
		particles.x[i] += velocitiesx   [i] * 0.1f;
		particles.y[i] += velocitiesy   [i] * 0.1f;
		particles.z[i] += velocitiesz   [i] * 0.1f;
	}

#elif STRATEGY == SERIAL_IMPROVED 

for (int i = 0; i < n_particles; i ++)
    {
        for (int j = 0; j < i; j++)
        {
			if(i != j)
			{
				const float diffx = particles.x[j] - particles.x[i];
				const float diffy = particles.y[j] - particles.y[i];
				const float diffz = particles.z[j] - particles.z[i];

				float dij = diffx * diffx + diffy * diffy + diffz * diffz;

				if (dij < 1.0)
				{
					dij = 10.0;
				}
				else
				{
					dij = std::sqrt(dij);
					dij = 10.0 / (dij * dij * dij);
				}
				accelerationsx[i] += diffx * dij * initstate.masses[j];
				accelerationsy[i] += diffy * dij * initstate.masses[j];
				accelerationsz[i] += diffz * dij * initstate.masses[j];

                accelerationsx[j] -= diffx * dij * initstate.masses[i];
				accelerationsy[j] -= diffy * dij * initstate.masses[i];
				accelerationsz[j] -= diffz * dij * initstate.masses[i];
			}
		}
	}

	for (int i = 0; i < n_particles; i++)
	{
		velocitiesx[i] += accelerationsx[i] * 2.0f;
		velocitiesy[i] += accelerationsy[i] * 2.0f;
		velocitiesz[i] += accelerationsz[i] * 2.0f;
		particles.x[i] += velocitiesx   [i] * 0.1f;
		particles.y[i] += velocitiesy   [i] * 0.1f;
		particles.z[i] += velocitiesz   [i] * 0.1f;
	}

//OMP  version
#elif STRATEGY == PARFOR_NAIVE
#pragma omp parallel for
    for (int i = 0; i < n_particles; i ++)
    {
        for (int j = 0; j < n_particles; j++)
        {
			if(i != j)
			{
				const float diffx = particles.x[j] - particles.x[i];
				const float diffy = particles.y[j] - particles.y[i];
				const float diffz = particles.z[j] - particles.z[i];

				float dij = diffx * diffx + diffy * diffy + diffz * diffz;

				if (dij < 1.0)
				{
					dij = 10.0;
				}
				else
				{
					dij = std::sqrt(dij);
					dij = 10.0 / (dij * dij * dij);
				}
				accelerationsx[i] += diffx * dij * initstate.masses[j];
				accelerationsy[i] += diffy * dij * initstate.masses[j];
				accelerationsz[i] += diffz * dij * initstate.masses[j];
			}
		}
	}
#pragma omp parallel for
	for (int i = 0; i < n_particles; i++)
	{
		velocitiesx[i] += accelerationsx[i] * 2.0f;
		velocitiesy[i] += accelerationsy[i] * 2.0f;
		velocitiesz[i] += accelerationsz[i] * 2.0f;
		particles.x[i] += velocitiesx   [i] * 0.1f;
		particles.y[i] += velocitiesy   [i] * 0.1f;
		particles.z[i] += velocitiesz   [i] * 0.1f;
	}
// #elif STRATEGY == STD_ATOMIC

// std::atomic<float> sumx{0ull};
// std::atomic<float> sumy{0ull};
// std::atomic<float> sumz{0ull};
// for (int i = 0; i < n_particles; i ++)
//     {
//         #pragma omp parallel for
//         for (int j = 0; j <n_particles; j++)
//         {
//             sumx = 0.0;
//             sumy = 0.0;
//             sumz = 0.0;
// 			if(i != j)
// 			{
// 				const float diffx = particles.x[j] - particles.x[i];
// 				const float diffy = particles.y[j] - particles.y[i];
// 				const float diffz = particles.z[j] - particles.z[i];

// 				float dij = diffx * diffx + diffy * diffy + diffz * diffz;

// 				if (dij < 1.0)
// 				{
// 					dij = 10.0;
// 				}
// 				else
// 				{
// 					dij = std::sqrt(dij);
// 					dij = 10.0 / (dij * dij * dij);
// 				}

// 				sumx += diffx * dij * initstate.masses[j];
// 				sumy += diffy * dij * initstate.masses[j];
// 				sumz += diffz * dij * initstate.masses[j];
// 			}
// 		}
//         	accelerationsx[i] = sumx;
// 			accelerationsy[i] = sumy;
// 			accelerationsz[i] = sumz;
// 	}
//     #pragma omp parallel for
// 	for (int i = 0; i < n_particles; i++)
// 	{
// 		velocitiesx[i] += accelerationsx[i] * 2.0f;
// 		velocitiesy[i] += accelerationsy[i] * 2.0f;
// 		velocitiesz[i] += accelerationsz[i] * 2.0f;
// 		particles.x[i] += velocitiesx   [i] * 0.1f;
// 		particles.y[i] += velocitiesy   [i] * 0.1f;
// 		particles.z[i] += velocitiesz   [i] * 0.1f;
// 	}


#endif

// OMP + xsimd version
// #pragma omp parallel for
//     for (int i = 0; i < n_particles; i += b_type::size)
//     {
//         // load registers body i
//         const b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
//         const b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
//         const b_type rposz_i = b_type::load_unaligned(&particles.z[i]);
//               b_type raccx_i = b_type::load_unaligned(&accelerationsx[i]);
//               b_type raccy_i = b_type::load_unaligned(&accelerationsy[i]);
//               b_type raccz_i = b_type::load_unaligned(&accelerationsz[i]);

//         ...
//     }

}

#endif //GALAX_MODEL_CPU_FAST
