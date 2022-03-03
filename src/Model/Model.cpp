#include "Model.hpp"
#include <cassert>
#include <numeric>
#include <cmath>
#include <iostream>

Model
::Model(const Initstate& initstate, Particles& particles)
: initstate(initstate),
  n_particles(particles.x.size()),
  particles(particles)
{
}

float Model
::compareParticlesState(const Model& referenceModel)
{
    // Compute the average distance between the particles in the two datasets.
    // We could also do a relative error, but given that we expect extremely
    // close results regardless of the Model used, this should be good enough.
    assert(particles.x.size() == referenceModel.particles.x.size() &&
           "The two compared models should be of the same simulation.");
    std::vector<float> distances(particles.x.size(), 0.0); // Maybe make this static

    // Should we parallelize?
    for(size_t p = 0; p < distances.size(); ++p)
        distances[p] = std::sqrt((particles.x[p] - referenceModel.particles.x[p]) * (particles.x[p] - referenceModel.particles.x[p]) +
                                 (particles.y[p] - referenceModel.particles.y[p]) * (particles.y[p] - referenceModel.particles.y[p]) +
                                 (particles.z[p] - referenceModel.particles.z[p]) * (particles.z[p] - referenceModel.particles.z[p]));

    return std::accumulate(distances.cbegin(), distances.cend(), 0.0) / distances.size();
}

