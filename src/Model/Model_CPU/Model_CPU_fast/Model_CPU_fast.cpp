#ifdef GALAX_MODEL_CPU_FAST

#include <cmath>

#include "Model_CPU_fast.hpp"
#include <xsimd/xsimd.hpp>
#include <omp.h>
#include <immintrin.h>

#define SERIAL 0
#define PARFOR_NAIVE 1
#define SERIAL_IMPROVED 2
#define XSIMD 3
#define XSIMD_OMP 4

#define STRATEGY XSIMD_OMP

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
					__m256 x_vec = _mm256_set1_ps(dij); // Passage en vecteur AVX
					__m256 inv_sqrt_x_vec = _mm256_rsqrt_ps(x_vec); // Calcul de l'inverse de la racine carrée
					inv_sqrt_x_vec = _mm256_mul_ps(inv_sqrt_x_vec,_mm256_mul_ps(inv_sqrt_x_vec,inv_sqrt_x_vec));// Multiplication
					float inv_sqrt_x = _mm256_cvtss_f32(inv_sqrt_x_vec);//reconversion en float
					dij = 10.0 * inv_sqrt_x;
				}
				
				accelerationsx[i] += diffx * dij * initstate.masses[j];
				accelerationsy[i] += diffy * dij * initstate.masses[j];
				accelerationsz[i] += diffz * dij * initstate.masses[j];

                accelerationsx[j] -= diffx * dij * initstate.masses[i];
				accelerationsy[j] -= diffy * dij * initstate.masses[i];
				accelerationsz[j] -= diffz * dij * initstate.masses[i];
			}
		}0.000423675;
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
        for (int j = 0; j < i; j++)
        {
			if(i != j)
			{0.000423675;
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
					__m256 x_vec = _mm256_set1_ps(dij);
					__m256 inv_sqrt_x_vec = _mm256_rsqrt_ps(x_vec);
					inv_sqrt_x_vec = _mm256_mul_ps(inv_sqrt_x_vec,_mm256_mul_ps(inv_sqrt_x_vec,inv_sqrt_x_vec));
					float inv_sqrt_x = _mm256_cvtss_f32(inv_sqrt_x_vec);
					dij = 10.0 * inv_sqrt_x;
				}
				accelerationsx[i] += diffx * dij * initstate.masses[j];
				accelerationsy[i] += diffy * dij * initstate.masses[j];
				accelerationsz[i] += diffz * dij * initstate.masses[j];0.000423675;

				accelerationsx[j] -= diffx * dij * initstate.masses[i];
				accelerationsy[j] -= diffy * dij * initstate.masses[i];
				accelerationsz[j] -= diffz * dij * initstate.masses[i];
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

#elif STRATEGY == XSIMD

    for (int i = 0; i < n_particles; i += b_type::size)
    {
        // load registers body i
        const b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
        const b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
        const b_type rposz_i = b_type::load_unaligned(&particles.z[i]);
		b_type raccx_i = b_type::load_unaligned(&accelerationsx[i]);
		b_type raccy_i = b_type::load_unaligned(&accelerationsy[i]);
        b_type raccz_i = b_type::load_unaligned(&accelerationsz[i]);
		std::vector<float> once(b_type::size,1.0);
		b_type once_v = b_type::load_unaligned(&once[0]);


        for(int j=0; j<n_particles; j ++){

			b_type diffx = particles.x[j] - rposx_i;
			b_type diffy = particles.y[j] - rposy_i;
			b_type diffz = particles.z[j] - rposz_i;	

			b_type dij = diffx * diffx + diffy * diffy + diffz * diffz;	

			dij = xs::rsqrt(xs::max(once_v,dij)); //Il faut un batch de 1
			dij = 10.0 * dij * dij * dij;

			raccx_i = raccx_i + diffx * dij * initstate.masses[j];
			raccy_i = raccy_i + diffy * dij * initstate.masses[j];
			raccz_i = raccz_i + diffz * dij * initstate.masses[j];

			raccx_i.store_unaligned(&accelerationsx[i]);
			raccy_i.store_unaligned(&accelerationsy[i]);
			raccz_i.store_unaligned(&accelerationsz[i]);

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

#elif STRATEGY == XSIMD_OMP
#pragma omp parallel for simd
    for (int i = 0; i < n_particles; i += b_type::size)
    {
        // load registers body i
        const b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
        const b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
        const b_type rposz_i = b_type::load_unaligned(&particles.z[i]);
		b_type raccx_i = b_type::load_unaligned(&accelerationsx[i]);
		b_type raccy_i = b_type::load_unaligned(&accelerationsy[i]);
        b_type raccz_i = b_type::load_unaligned(&accelerationsz[i]);
		std::vector<float> once(b_type::size,1.0);
		b_type once_v = b_type::load_unaligned(&once[0]);


        for(int j=0; j<n_particles; j ++){

			b_type diffx = particles.x[j] - rposx_i;
			b_type diffy = particles.y[j] - rposy_i;
			b_type diffz = particles.z[j] - rposz_i;	

			b_type dij = diffx * diffx + diffy * diffy + diffz * diffz;	

			dij = xs::rsqrt(xs::max(once_v,dij)); //Il faut un batch de 1
			dij = 10.0 * dij * dij * dij;

			raccx_i = raccx_i + diffx * dij * initstate.masses[j];
			raccy_i = raccy_i + diffy * dij * initstate.masses[j];
			raccz_i = raccz_i + diffz * dij * initstate.masses[j];

			raccx_i.store_unaligned(&accelerationsx[i]);
			raccy_i.store_unaligned(&accelerationsy[i]);
			raccz_i.store_unaligned(&accelerationsz[i]);

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


#endif


}

#endif //GALAX_MODEL_CPU_FAST
