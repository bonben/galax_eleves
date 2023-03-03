#ifdef GALAX_MODEL_CPU_FAST

#include <cmath>
#include <iostream>
#include "Model_CPU_fast.hpp"

#include <xsimd/xsimd.hpp>
#include <omp.h>

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
    using b_type = xsimd::batch<float, xsimd::avx2>;

    std::fill(accelerationsx.begin(), accelerationsx.end(), 0);
    std::fill(accelerationsy.begin(), accelerationsy.end(), 0);
    std::fill(accelerationsz.begin(), accelerationsz.end(), 0);
    
    std::size_t inc = b_type.size();
    std::size_t size;
    
    std::size_t vec_size;
    
    float diffx,diffy,diffz;
    float dij,dij_mj,dij_mi;
    
    for (int i = 0; i < n_particles; i++)
	{
        size=n_particles-1-i;
        vec_size=size-size%inc
		for (int j = 0; j < i; j++)
		{
			
            diffx = particles.x[j] - particles.x[i];
            diffy = particles.y[j] - particles.y[i];
            diffz = particles.z[j] - particles.z[i];

            dij = diffx * diffx + diffy * diffy + diffz * diffz;

            if (dij > 1)
            {
                dij = 10.0/(dij*std::sqrt(dij));// looks like it's the fastest way
                dij_mj = dij * initstate.masses[j];
                dij_mi = dij * initstate.masses[i];

            }
            else
            {
                dij_mj = 10.0 * initstate.masses[j];
                dij_mi = 10.0 * initstate.masses[i];
            }

            accelerationsx[i] += diffx * dij_mj;
            accelerationsy[i] += diffy * dij_mj;
            accelerationsz[i] += diffz * dij_mj;

            accelerationsx[j] -= diffx * dij_mi;
            accelerationsy[j] -= diffy * dij_mi;
            accelerationsz[j] -= diffz * dij_mi;
            
		}
	}

    // quand une boucle sur i est finie, on peut commencer cette boucle
	for (int i = 0; i < n_particles; i++)
	{
		velocitiesx[i] += accelerationsx[i] * 2.0f;
		velocitiesy[i] += accelerationsy[i] * 2.0f;
		velocitiesz[i] += accelerationsz[i] * 2.0f;
		particles.x[i] += velocitiesx   [i] * 0.1f;
		particles.y[i] += velocitiesy   [i] * 0.1f;
		particles.z[i] += velocitiesz   [i] * 0.1f;
	}


// OMP  version
// #pragma omp parallel for
//     for (int i = 0; i < n_particles; i ++)
//     {
//     }


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

#endif // GALAX_MODEL_CPU_FAST
