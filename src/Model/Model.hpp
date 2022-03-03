#ifndef MODEL_HPP_
#define MODEL_HPP_

#include <vector>

#include "../Initstate.hpp"
#include "../Particles.hpp"

class Model
{
protected:
    const Initstate& initstate;
    const int n_particles;

    Particles& particles;

public:
    Model(const Initstate& initstate, Particles& particles);

    float compareParticlesState(const Model& reference);

    virtual ~Model() = default;

    virtual void step() = 0;
};

#endif // MODEL_HPP
