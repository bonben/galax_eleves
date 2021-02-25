#include "Model.hpp"

Model
::Model(const Initstate& initstate, Particles& particles)
: initstate(initstate),
  n_particles(particles.x.size()),
  particles(particles)
{
}
