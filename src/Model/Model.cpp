#include "Model.hpp"
#include <cassert>
#include <numeric>
#include <cmath>
#include <iostream>
#include <algorithm>

Model
::Model(const Initstate& initstate, Particles& particles)
: initstate(initstate),
  n_particles(particles.x.size()),
  particles(particles)
{
}

std::tuple<float, float, float> Model
::compareParticlesState(const Model& referenceModel, bool returnRelativeDistances)
{
    // Compute the average distance between the particles in the two datasets.
    // We could also do a relative error, but given that we expect extremely
    // close results regardless of the Model used, this should be good enough.
    assert(particles.x.size() == referenceModel.particles.x.size() &&
           "The two compared models should be of the same simulation.");
    std::vector<float> distances(particles.x.size(), 0.0); // Maybe make this static
    std::vector<float> relative_distances(particles.x.size(), 0.0); // Maybe make this static

    // Should we parallelize?
    for(size_t p = 0; p < distances.size(); ++p)
    {
        distances[p] = std::sqrt((particles.x[p] - referenceModel.particles.x[p]) * (particles.x[p] - referenceModel.particles.x[p]) +
                                 (particles.y[p] - referenceModel.particles.y[p]) * (particles.y[p] - referenceModel.particles.y[p]) +
                                 (particles.z[p] - referenceModel.particles.z[p]) * (particles.z[p] - referenceModel.particles.z[p]));

        relative_distances[p] = distances[p] * std::sqrt((referenceModel.particles.x[p]) * (referenceModel.particles.x[p]) +
                                                         (referenceModel.particles.y[p]) * (referenceModel.particles.y[p]) +
                                                         (referenceModel.particles.z[p]) * (referenceModel.particles.z[p]));
    }

    if(returnRelativeDistances)
    {
        auto minmax = minmax_element(distances.cbegin(), distances.cend());
        return {*minmax.first, *minmax.second, std::accumulate(distances.cbegin(), distances.cend(), 0.0) / distances.size()};
    }
    else
    {
        auto minmax = minmax_element(relative_distances.cbegin(), relative_distances.cend());
        return {*minmax.first, *minmax.second, std::accumulate(relative_distances.cbegin(), relative_distances.cend(), 0.0) / relative_distances.size()};
    }
}

