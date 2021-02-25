#ifndef MODEL_CPU_NAIVE_HPP_
#define MODEL_CPU_NAIVE_HPP_

#include "../Model_CPU.hpp"

class Model_CPU_naive : public Model_CPU
{
public:
    Model_CPU_naive(const Initstate& initstate, Particles& particles);

    virtual ~Model_CPU_naive() = default;

    virtual void step();
};
#endif // MODEL_CPU_NAIVE_HPP_