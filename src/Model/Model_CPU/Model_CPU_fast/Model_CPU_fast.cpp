#ifdef GALAX_MODEL_CPU_FAST

#include <cmath>
#include <iostream>
#include "Model_CPU_fast.hpp"

#include <xsimd/xsimd.hpp>
#include <omp.h>

namespace xs = xsimd;
using b_type = xs::batch<float, xs::avx2>;
const int memory_block_size = 200;
const int num_parral_call =20;


Model_CPU_fast
::Model_CPU_fast(const Initstate& initstate, Particles& particles)
: Model_CPU(initstate, particles)
{
}


inline void compute_difference(const float &xi, const float &yi,const float &zi,\
            const float &xj, const float &yj, const float &zj,\
            float &diffx, float &diffy, float &diffz, float &dij
            ){
            
            diffx = xj - xi;
            diffy = yj - yi;
            diffz = zj - zi;
            dij = diffx * diffx + diffy * diffy + diffz * diffz;
}


inline void accumulate_acceleration(const float &diffx, const float &diffy, const float &diffz,\
            const float &dij_mi, const float &dij_mj, \
            float &accelerationsxi, float &accelerationsyi, float &accelerationszi,\
            float &accelerationsxj, float &accelerationsyj, float &accelerationszj
            ){

            accelerationsxi += diffx * dij_mj;
            accelerationsyi += diffy * dij_mj;
            accelerationszi += diffz * dij_mj;
            accelerationsxj -= diffx * dij_mi;
            accelerationsyj -= diffy * dij_mi;
            accelerationszj -= diffz * dij_mi;
}

inline void compute_forces(const float &mi,const float &mj,float &dij, float &dij_mi,float &dij_mj){
    if (dij > 1)
    {
        dij = 1./(dij*std::sqrt(dij));// looks like it's the fastest way
        dij_mj = dij * mj;
        dij_mi = dij * mi;
    }
    else
    {
        dij_mj = mj;
        dij_mi = mi;

    }

}

inline void update_acceleration(const float &xi, const float &yi,const float &zi,\
            const float &xj, const float &yj, const float &zj,\
            const float &massei, const float &massej,\
            float &accelerationsxi, float &accelerationsyi, float &accelerationszi,\
            float &accelerationsxj, float &accelerationsyj, float &accelerationszj
            ){
            // Total : 20 memory access
            // 8 memory access
            // 6 temporary variable (1 can be avoided)
            // 6 memory write

            float diffx,diffy,diffz;
            float dij, dij_mj,dij_mi;

            
            // 6 memory access -> 6N registers (block)
            // 4 memory write  -> 4N² registers
            // 
            compute_difference(xi,yi,zi,xj,yj,zj,diffx,diffy,diffz,dij);

            
            // 2 memory access -> 2N
            // 1 other variable (dij) -> N²
            // 2 memory write -> N²
            compute_forces(massei,massej,dij,dij_mi,dij_mj);

            // 5 memory access -> 5 N² 
            // 6 memory write ->  6N
            // vectorisable
            accumulate_acceleration(diffx,diffy,diffz,dij_mi,dij_mj,\
            accelerationsxi,accelerationsyi,accelerationszi,accelerationsxj,accelerationsyj,accelerationszj);
}

inline void update_block_register(int i_min, int i_max,int j_min,int j_max,int block_size,Particles &particles, const float* masses,\
 float* accelerationsx, float* accelerationsy, float* accelerationsz){
    // parralel number register : 6 N (N+1) <a  -> n< a/6 -1 -> n=6
    // looking for x such as (x*x)*number_write
    // +4 to count index ?
    
    assert (block_size % num_parral_call ==0);


    
    float dij_mj[num_parral_call][num_parral_call];
    float dij_mi[num_parral_call][num_parral_call];
    
    
    for (int i = i_min; i <  i_max; i +=num_parral_call ){
            for(int j = j_min; j<j_max; j+=num_parral_call){
                
                //allocate space for data acess
                float xi[num_parral_call];
                float xj[num_parral_call];
                float yi[num_parral_call];
                float yj[num_parral_call];
                float zi[num_parral_call];
                float zj[num_parral_call];

                // allocate space for tempory variables write
                float dij[num_parral_call][num_parral_call];
                float diffx[num_parral_call][num_parral_call];
                float diffy[num_parral_call][num_parral_call];
                float diffz[num_parral_call][num_parral_call];
                
                #pragma unroll
                for (int i_reg = 0; i_reg< num_parral_call; i_reg++ ){
                    xi[i_reg] = particles.x[i+i_reg];
                    yi[i_reg] = particles.y[i+i_reg];
                    zi[i_reg] = particles.z[i+i_reg];
                    xj[i_reg] = particles.x[j+i_reg];
                    yj[i_reg] = particles.y[j+i_reg];
                    zj[i_reg] = particles.z[j+i_reg];
                }

                // compute distances and differences
                #pragma unroll
                for (int i_reg = 0; i_reg< num_parral_call; i_reg++ ){
                    for(int j_reg= 0; j_reg< num_parral_call; j_reg++){
                        compute_difference(xi[i_reg],yi[i_reg],zi[i_reg],xj[j_reg],yj[j_reg],zj[j_reg],\
                        diffx[i_reg][j_reg],diffy[i_reg][j_reg],diffz[i_reg][j_reg],dij[i_reg][j_reg]);
                    }
                }

                // now, allocate memory for the masses and the forces
                float dij_mj[num_parral_call][num_parral_call];
                float dij_mi[num_parral_call][num_parral_call];

                float massei[num_parral_call];
                float massej[num_parral_call];
                // acctualy fill the masses
                #pragma unroll
                for (int i_reg = 0; i_reg< num_parral_call; i_reg++ ){
                    massei[i_reg] = masses[i+i_reg];
                    massej[i_reg] = masses[j+i_reg];
                }

                // update dij using masses 
                #pragma unroll
                for (int i_reg = 0; i_reg< num_parral_call; i_reg++ ){
                    for(int j_reg= 0; j_reg< num_parral_call; j_reg++){
                        compute_forces(
                            massei[i_reg],massej[j_reg],\
                            dij[i_reg][j_reg],dij_mi[i_reg][j_reg],dij_mj[i_reg][j_reg]
                        );
                    }
                }

                // update accelerations
                #pragma unroll
                for (int i_reg = 0; i_reg< num_parral_call; i_reg++ ){
                    #pragma unroll
                    for(int j_reg= 0; j_reg< num_parral_call; j_reg++){
                        accumulate_acceleration(
                            diffx[i_reg][j_reg],diffy[i_reg][j_reg],diffz[i_reg][j_reg],\
                            dij_mi[i_reg][j_reg],dij_mj[i_reg][j_reg],\
                            accelerationsx[i+i_reg],accelerationsy[i+i_reg],accelerationsz[i+i_reg],\
                            accelerationsx[j+j_reg],accelerationsy[j+j_reg],accelerationsz[j+j_reg]
                        );
                    }
                }
            }
    }


}




inline void update_block_full(int i_min, int i_max,int j_min,int j_max,int block_size,Particles &particles, const float* masses, float* accelerationsx, float* accelerationsy, float* accelerationsz){
    
    for (int ij = 0; ij < block_size * block_size; ij++) {
        int i = i_min + ij / block_size;
        int j = j_min + ij % block_size;
        update_acceleration(
                particles.x[i], particles.y[i], particles.z[i],
                particles.x[j], particles.y[j], particles.z[j],
                masses[i], masses[j],
                accelerationsx[i], accelerationsy[i], accelerationsz[i],
                accelerationsx[j], accelerationsy[j], accelerationsz[j]
            );
            
    }

}

inline void update_block_triangular(int i_min, int i_max,int block_size,Particles &particles, const float* masses, float* accelerationsx, float* accelerationsy, float* accelerationsz){
    
    for (int i = i_min; i <  i_max; i++){
            for(int j =i_min; j<i; j++){
                update_acceleration(
                        particles.x[i], particles.y[i], particles.z[i],
                        particles.x[j], particles.y[j], particles.z[j],
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

    


    int number_blocks = n_particles/memory_block_size ;

    for (int block_id_i = 0; block_id_i < number_blocks; block_id_i++)
	{
        int min_i = block_id_i*memory_block_size;
        int max_i = min_i+memory_block_size;
       
        update_block_triangular(min_i,max_i,memory_block_size,particles,masses,ax,ay,az);
        // other blocks on the lower triangle
        for (int block_id_j = 0; block_id_j < block_id_i; block_id_j += 1)
        {   
                int min_j=block_id_j*memory_block_size;
                int max_j = min_j+memory_block_size;
                update_block_register(min_i,max_i,min_j,max_j,memory_block_size,particles,masses,ax,ay,az);
                //update_block_full(min_i,max_i,min_j,max_j,memory_block_size,particles,masses,ax,ay,az);  
        }
    }

    bool is_enough = (n_particles % memory_block_size) == 0;
    if (!is_enough){
        int min_i = number_blocks*memory_block_size;
        int max_i = n_particles;
        update_block_triangular(min_i,max_i,memory_block_size,particles,masses,ax,ay,az);
        for (int block_id_j = 0; block_id_j < number_blocks; block_id_j++)
        {   
                int min_j=block_id_j*memory_block_size;
                int max_j = min_j+memory_block_size;
                update_block_full(min_i,max_i,min_j,max_j,memory_block_size,particles,masses,ax,ay,az);  
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
