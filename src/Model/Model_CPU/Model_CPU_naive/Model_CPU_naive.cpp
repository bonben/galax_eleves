#include <cmath>

#include "Model_CPU_naive.hpp"

Model_CPU_naive
::Model_CPU_naive(const Initstate& initstate, Particles& particles)
: Model_CPU(initstate, particles)
{
}

void Model_CPU_naive
::step()
{
	std::fill(accelerationsx.begin(), accelerationsx.end(), 0);
	std::fill(accelerationsy.begin(), accelerationsy.end(), 0);
	std::fill(accelerationsz.begin(), accelerationsz.end(), 0);

	// we do the same operation twice
	for (int i = 0; i < n_particles; i++)
	{
		for (int j = 0; j < n_particles; j++)
		{
			if(i != j)
			{
				// la, pas trés utile de prendre l'array pour i
				const float diffx = particles.x[j] - particles.x[i];
				const float diffy = particles.y[j] - particles.y[i];
				const float diffz = particles.z[j] - particles.z[i];

				// ici, on doit attendre que les instructions précédentes soit fait
				// seul endroit ou les instructions dependent des autres
				float dij = diffx * diffx + diffy * diffy + diffz * diffz;

				
				if (dij < 1.0)
				{
					dij = 10.0;
				}
				else
				{
					// is this the best way to compute the distance ? 
					dij = std::sqrt(dij);
					dij = 10.0 / (dij * dij * dij);//-mrecip
				}

				// On accede à j (variable), on write sur i (constante)
				// calcul de dij avant '3 fois le meme calcul)
				accelerationsx[i] += diffx * dij * initstate.masses[j];
				accelerationsy[i] += diffy * dij * initstate.masses[j];
				accelerationsz[i] += diffz * dij * initstate.masses[j];
			}
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
}