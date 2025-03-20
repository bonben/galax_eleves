#ifndef MODEL_CPU_BH_H
#define MODEL_CPU_BH_H

#include "../Model_CPU.hpp"
#include "../../Body.hpp"
#include <condition_variable>
#include <latch>
#include <mutex>
#include <thread>
#include <vector>
#include <array>
#include "../BHTree.hpp"

class Model_CPU_BH : public Model_CPU
{
public:
    Model_CPU_BH(const Initstate& initstate, Particles& particles);

    virtual ~Model_CPU_BH() = default;

    virtual void step();
private:
    std::vector<Body> bodies;
    std::array<std::vector<Body>,8> bodies_per_thread;
    std::array<std::unique_ptr<BHTree>,8> trees;
    double radius;
    Vector3 last_mass_center = {0.0, 0.0, 0.0};
    std::array<std::thread, 8> threads;
    void thread_insert(int index);
    void thread_proc(int index);
    std::array<std::mutex, 8> mutexes;
    std::condition_variable cv;
    std::array<char, 8> orders {{0,0,0,0,0,0,0,0}};
    std::unique_ptr<std::latch> insert_done;
};

#endif // MODEL_CPU_BH_H
