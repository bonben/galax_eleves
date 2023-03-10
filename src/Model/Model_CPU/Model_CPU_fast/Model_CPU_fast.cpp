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


inline void update_acceleration(const float &xi, const float &yi,const float &zi,\
            const float &xj, const float &yj, const float &zj,\
            const float &massei, const float &massej,\
            float &accelerationsxi, float &accelerationsyi, float &accelerationszi,\
            float &accelerationsxj, float &accelerationsyj, float &accelerationszj
            ){

            
            float dij, dij_mj,dij_mi;
            float diffx = xj - xi;
            float diffy = yj - yi;
            float diffz = zj - zi;
            dij = diffx * diffx + diffy * diffy + diffz * diffz;

            if (dij > 1)
            {
                dij = 1./(dij*std::sqrt(dij));// looks like it's the fastest way
                dij_mj = dij * massej;
                dij_mi = dij * massei;
            }
            else
            {
                dij_mj = massej;
                dij_mi = massei;
            }

            accelerationsxi += diffx * dij_mj;
            accelerationsyi += diffy * dij_mj;
            accelerationszi += diffz * dij_mj;
            accelerationsxj -= diffx * dij_mi;
            accelerationsyj -= diffy * dij_mi;
            accelerationszj -= diffz * dij_mi;
}

inline void update_block_full(int i_min, int i_max,int j_min,int j_max,float* x, float* y, float* z, const float* masses, float* accelerationsx, float* accelerationsy, float* accelerationsz){
    for (int i = i_min; i <  i_max; i++){
            for(int j = j_min; j<j_max; j++){
                update_acceleration(
                        x[i], y[i], z[i],
                        x[j], y[j], z[j],
                        masses[i], masses[j],
                        accelerationsx[i], accelerationsy[i], accelerationsz[i],
                        accelerationsx[j], accelerationsy[j], accelerationsz[j]
                    );
            }
    }
}

inline void update_block_triangular(int i_min, int i_max,float* x, float* y, float* z, const float* masses, float* accelerationsx, float* accelerationsy, float* accelerationsz){
    for (int i = i_min; i <  i_max; i++){
            for(int j =i_min; j<i; j++){
                update_acceleration(
                        x[i], y[i], z[i],
                        x[j], y[j], z[j],
                        masses[i], masses[j],
                        accelerationsx[i], accelerationsy[i], accelerationsz[i],
                        accelerationsx[j], accelerationsy[j], accelerationsz[j]
                    );
            }
    }
}

// }
void Model_CPU_fast
::step()
{   
    using b_type = xsimd::batch<float, xsimd::avx2>;
    // std::size_t inc = b_type.size();
    // std::size_t size;
    // std::size_t vec_size;
            
    std::fill(accelerationsx.begin(), accelerationsx.end(), 0);
    std::fill(accelerationsy.begin(), accelerationsy.end(), 0);
    std::fill(accelerationsz.begin(), accelerationsz.end(), 0);
    
    float diffx,diffy,diffz;
    float dij,dij_mj,dij_mi;

    //operate directly in memory
    float* x = particles.x.data();
    float* y =  particles.y.data();
    float* z =  particles.z.data();
    float* ax= accelerationsx.data();
    float* ay= accelerationsy.data();
    float* az= accelerationsz.data();
    const float* masses = initstate.masses.data();
    const float G = 10.0;

    int memory_block_size = 8;


    int number_blocks = n_particles/memory_block_size ;

    for (int block_id_i = 0; block_id_i < number_blocks; block_id_i++)
	{
        int min_i = block_id_i*memory_block_size;
        int max_i = min_i+memory_block_size;
       
        update_block_triangular(min_i,max_i,x,y,z,masses,ax,ay,az);
        // other blocks on the lower triangle
        for (int block_id_j = 0; block_id_j < block_id_i; block_id_j += 1)
        {   
                int min_j=block_id_j*memory_block_size;
                int max_j = min_j+memory_block_size;
                update_block_full(min_i,max_i,min_j,max_j,x,y,z,masses,ax,ay,az);  
        }
    }

    bool is_enough = (n_particles % memory_block_size) == 0;
    if (!is_enough){
        int min_i = number_blocks*memory_block_size;
        int max_i = n_particles;
        update_block_triangular(min_i,max_i,x,y,z,masses,ax,ay,az);
        for (int block_id_j = 0; block_id_j < number_blocks; block_id_j++)
        	{   
                int min_j=block_id_j*memory_block_size;
                int max_j = min_j+memory_block_size;
                update_block_full(min_i,max_i,min_j,max_j,x,y,z,masses,ax,ay,az);  
        }
    }

    for(int i =0; i<n_particles; i++){
        accelerationsx[i] *= G;
        accelerationsy[i] *= G;
        accelerationsz[i] *= G;
    }

    // quand une boucle sur i est finie, on peut commencer cette boucle
	for (int i = 0; i < n_particles; i++)
	{
		velocitiesx[i] += accelerationsx[i] * 2.0f;
		velocitiesy[i] += accelerationsy[i] * 2.0f;
		velocitiesz[i] += accelerationsz[i] * 2.0f;
		x[i] += velocitiesx   [i] * 0.1f;
		y[i] += velocitiesy   [i] * 0.1f;
		z[i] += velocitiesz   [i] * 0.1f;
	}


// OMP  version
// #pragma omp parallel for
//     for (int i = 0; i < n_particles; i ++)
//     {
//     }


// OMP + xsimd version
// #pragma omp parallel for
//     for (int i = 0; i < compute_accelerationn_particles; i += b_type::size)
//     {
//         // load registers body i
//         const b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
//         const b_type rposy_i = b_type::load_unaligned(&y[i]);
//         const b_type rposz_i = b_type::load_unaligned(&z[i]);
//               b_type raccx_i = b_type::load_unaligned(&accelerationsx[i]);
//               b_type raccy_i = b_type::load_unaligned(&accelerationsy[i]);
//               b_type raccz_i = b_type::load_unaligned(&accelerationsz[i]);

//         ...
//     }

}

#endif // GALAX_MODEL_CPU_FAST
