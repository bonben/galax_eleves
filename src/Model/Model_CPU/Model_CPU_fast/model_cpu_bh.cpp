#include "model_cpu_bh.hpp"
#include "../BHTree.hpp"
#include <iostream>

Model_CPU_BH::Model_CPU_BH(const Initstate &initstate, Particles &particles) : Model_CPU(initstate, particles)
{
    radius = 0;
    for(size_t i = 0; i < initstate.positionsx.size(); i++) {
        Body b;
        b.pos = {initstate.positionsx[i], initstate.positionsy[i], initstate.positionsz[i]};
        b.spd = {initstate.velocitiesx[i], initstate.velocitiesy[i], initstate.velocitiesz[i]};
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
    std::cout << "Radius = " << radius << std::endl;
    for(auto& v : bodies_per_thread)
        v.reserve(bodies.size());

    for(int i = 0; i < 8; i++){
        threads[i] = std::thread([this,i] () {thread_proc(i);});
    }
}

void Model_CPU_BH::step()
{
    double dt = 0.1;

    /*for(auto& v : bodies_per_thread)
        v.reserve(bodies.size());*/

    radius = 0.0;
 
    Region r;
    r.center = {0.0,0.0,0.0};//last_mass_center;
    r.width = 2*radius;
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
    //std::cout << radius << std::endl;
    r.width = 2*radius;
    /*std::cout << r.width << std::endl;

    r.to_string();
    r.get_sub(UNE).to_string();
    r.get_sub(UNW).to_string();
    r.get_sub(USE).to_string();
    r.get_sub(USW).to_string();
    r.get_sub(LNE).to_string();
    r.get_sub(LNW).to_string();
    r.get_sub(LSE).to_string();
    r.get_sub(LSW).to_string();*/

    /*for(int i = 0; i < 8; i++)
        r.get_sub(Direction{i}).to_string();*/

    /*std::vector<std::unique_ptr<BHTree>> trees;*/
    for(int i = 0; i < 8 ;i++){
        trees[i] = (std::make_unique<BHTree>(r.get_sub(Direction{i})));
        trees[i]->to_string();
    }

    /*#pragma omp parallel for
    for(int i = 0; i < 8; i++) {
        for(auto& b : bodies_per_thread[i]) {
            trees[i]->insert(b);
        }
    }*/

    insert_done = std::make_unique<std::latch>(8);

    for(int i = 0; i < 8; i++) {
        thread_insert(i);
    }

    insert_done->wait();

    for(int i = 0; i < 8 ;i++){
        trees[i]->to_string();
    }

    /*BHTree tree {r};
    for(auto& b : bodies) {
        tree.insert(b);
    }*/

    BHTree tree{r, trees};

    tree.print_tree();

    exit(5);

    for(auto& b : bodies) {
        b.force = {0.0,0.0,0.0};
    }

    for(auto& b : bodies) {
        tree.update_force(b);
        b.update_pos(dt);
    }

    last_mass_center = tree.mass_center.pos;

    //std::cout << tree.height() << std::endl;

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
        cv.wait(lock, [index, this]{return orders[index]==1;});
        for(auto& b : bodies_per_thread[index]) {
            trees[index]->insert(b);
        }
        orders[index] = 0;
        insert_done->count_down();
    }
}
