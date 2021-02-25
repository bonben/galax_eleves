#ifdef GALAX_MODEL_CPU_FAST

#include <cmath>

#include "Model_CPU_fast.hpp"

#include <mipp.h>
#include <omp.h>

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

// OMP  version
// #pragma omp parallel for
//     for (int i = 0; i < n_particles; i += mipp::N<float>())
//     {
//     }


// OMP + MIPP version
// #pragma omp parallel for
//     for (int i = 0; i < n_particles; i += mipp::N<float>())
//     {
//         // load registers body i
//         const mipp::Reg<float> rposx_i = &particles.x[i];
//         const mipp::Reg<float> rposy_i = &particles.y[i];
//         const mipp::Reg<float> rposz_i = &particles.z[i];
//               mipp::Reg<float> raccx_i = &accelerationsx[i];
//               mipp::Reg<float> raccy_i = &accelerationsy[i];
//               mipp::Reg<float> raccz_i = &accelerationsz[i];
//     }
}

#endif // GALAX_MODEL_CPU_FAST
