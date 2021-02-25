#ifndef MODEL_CPU_HPP_
#define MODEL_CPU_HPP_

#include "../Model.hpp"

class Model_CPU : public Model
{
protected:
    std::vector<float> velocitiesx;
    std::vector<float> velocitiesy;
    std::vector<float> velocitiesz;
    std::vector<float> accelerationsx;
    std::vector<float> accelerationsy;
    std::vector<float> accelerationsz;

public:
    Model_CPU(const Initstate& initstate, Particles& particles);

    virtual ~Model_CPU() = default;

    virtual void step() = 0;
};
#endif // MODEL_CPU_NAIVE_HPP_