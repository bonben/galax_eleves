#include <fstream>

#include "Initstate.hpp"

Initstate
::Initstate(const int n_particles)
: positionsx (n_particles),
  positionsy (n_particles),
  positionsz (n_particles),
  velocitiesx(n_particles),
  velocitiesy(n_particles),
  velocitiesz(n_particles),
  masses     (n_particles)
{
    std::ifstream ifs;
	ifs.open(filename, std::ifstream::in);
	std::vector<std::vector<float> > all_particles (max_n_particles, std::vector<float>(7));

	size_t n_lines_read = 0;
	while (n_lines_read < all_particles.size())
	{
		for (size_t i = 0; i < 7; i++)
			ifs >> all_particles[n_lines_read][i];
		if (ifs.eof())
			break;
		n_lines_read++;
	}

	size_t stride = max_n_particles / n_particles;
	for (size_t i = 0; i < n_particles; i++)
    {
		positionsx [i] = all_particles[i * stride][1];
        positionsy [i] = all_particles[i * stride][2];
        positionsz [i] = all_particles[i * stride][3];
        velocitiesx[i] = all_particles[i * stride][4];
        velocitiesy[i] = all_particles[i * stride][5];
        velocitiesz[i] = all_particles[i * stride][6];
        masses     [i] = all_particles[i * stride][0];
    }
}
