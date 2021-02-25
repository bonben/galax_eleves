#include <cmath>

#include "Model_CPU.hpp"

Model_CPU
::Model_CPU(const Initstate& initstate, Particles& particles)
: Model(initstate, particles),
  velocitiesx   (n_particles),
  velocitiesy   (n_particles),
  velocitiesz   (n_particles),
  accelerationsx(n_particles),
  accelerationsy(n_particles),
  accelerationsz(n_particles)
{
	for (int i = 0; i < n_particles; i++)
	{
		particles.x[i] = initstate.positionsx[i];
		particles.y[i] = initstate.positionsy[i];
		particles.z[i] = initstate.positionsz[i];
	}
    std::copy(initstate.velocitiesx.begin(), initstate.velocitiesx.end(), velocitiesx.begin());
    std::copy(initstate.velocitiesy.begin(), initstate.velocitiesy.end(), velocitiesy.begin());
    std::copy(initstate.velocitiesz.begin(), initstate.velocitiesz.end(), velocitiesz.begin());
}
