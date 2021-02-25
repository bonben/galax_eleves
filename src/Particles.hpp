#ifndef PARTICLES_HPP_
#define PARTICLES_HPP_

#include <vector>

struct Particles
{
	std::vector<float> x;
	std::vector<float> y;
	std::vector<float> z;

	Particles(const int n_particles);
};

#endif // PARTICLES_HPP_
