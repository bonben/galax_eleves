#ifndef INITSTATE_HPP_
#define INITSTATE_HPP_

#include <string>
#include <vector>

#include "config.hpp"

class Initstate
{
private:
    const std::string filename = std::string(GALAX_ROOT) + "/data/dubinski.tab";
    const int max_n_particles  = 81920;

public:
    std::vector<float> positionsx;
    std::vector<float> positionsy;
    std::vector<float> positionsz;
    std::vector<float> velocitiesx;
    std::vector<float> velocitiesy;
    std::vector<float> velocitiesz;
    std::vector<float> masses;

public:
    Initstate(const int n_particles);
};

#endif // INITSTATE_HPP_
