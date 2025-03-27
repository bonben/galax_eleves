#include "model_cpu_bh.hpp"
#include "../BHTree.hpp"
#include <iostream>
#include <omp.h>
#include <syncstream>

Model_CPU_BH::Model_CPU_BH(const Initstate &initstate, Particles &particles) : Model_CPU(initstate, particles)
{
    radius = 0;
    for(size_t i = 0; i < initstate.positionsx.size(); i++) {
        Body b;
        b.pos = {initstate.positionsx[i], initstate.positionsy[i], initstate.positionsz[i]};
        b.spd = {0.1f * initstate.velocitiesx[i], 0.1f * initstate.velocitiesy[i], 0.1f * initstate.velocitiesz[i]};
        b.mass = initstate.masses[i];
        bodies.push_back(b);
        if(std::abs(b.pos.x) > radius) {
            radius = std::abs(b.pos.x);
        }
        if(std::abs(b.pos.y) > radius) {
            radius = std::abs(b.pos.y);
        }
        if(std::abs(b.pos.z) > radius) {
            radius = std::abs(b.pos.z);
        }
    }
    radius *= 2;
    for(auto& v : bodies_per_thread)
        v.reserve(bodies.size());
}

void Model_CPU_BH::step()
{

    radius = 0.0;
 
    Region r;
    r.center = {0.0,0.0,0.0};
    for(auto & v : bodies_per_thread)
        v.clear();
    for(auto& b : bodies) {
        size_t dir = (b.pos.x > r.center.x) | ((b.pos.y > r.center.y) << 1) | ((b.pos.z > r.center.z) << 2);
        if(std::abs(b.pos.x) > radius) {
            radius = std::abs(b.pos.x);
        }
        if(std::abs(b.pos.y) > radius) {
            radius = std::abs(b.pos.y);
        }
        if(std::abs(b.pos.z) > radius) {
            radius = std::abs(b.pos.z);
        }
        bodies_per_thread[dir].push_back(b);
    }
    r.width = 2*radius;
    r.width_sqr = r.width*r.width;

    for(int i = 0; i < 8 ;i++){
        trees[i] = (std::make_unique<BHTree>(r.get_sub(Direction{i})));
    }

    #pragma omp parallel for
    for(int i = 0; i < 8; i++) {
        for(auto& b : bodies_per_thread[i]) {
            trees[i]->insert(b);
        }
    }

    BHTree tree{r, trees};

    for(auto& b : bodies) {
        b.acceleration = {0.0,0.0,0.0};
    }

    #pragma omp parallel for
    for(auto& b : bodies) {
        tree.update_force(b);
        b.update_pos();
    }

    last_mass_center = tree.mass_center.pos;

    for(size_t i = 0; i < bodies.size(); i++) {
        particles.x[i] = bodies[i].pos.x;
        particles.y[i] = bodies[i].pos.y;
        particles.z[i] = bodies[i].pos.z;
    }
}

void Model_CPU_BH::thread_insert(int index)
{
    orders[index] = 1;
    cv.notify_all();
}

void Model_CPU_BH::thread_proc(int index)
{
    while(true) {
        std::unique_lock<std::mutex> lock(mutexes[index]);
        cv.wait(lock, [index, this]{std::osyncstream(std::cout) << "Test CV thread " << index << std::endl; return this->orders[index]==1;});
        std::osyncstream(std::cout) << "Start of thread" << index << std::endl;
        for(auto& b : bodies_per_thread[index]) {
            trees[index]->insert(b);
        }
        std::osyncstream(std::cout) << "End of thread" << index << std::endl;
        orders[index] = 0;
        insert_done->count_down();
    }
}
