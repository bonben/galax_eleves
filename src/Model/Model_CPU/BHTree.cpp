#include "BHTree.hpp"
#include <cmath>
#include <iostream>
#include <fstream>

inline int stid(bool b) {
    return b ? 1 : -1;
}

Region Region::get_sub(Direction dir) {
    Region res;
    res.center.z = center.z + (width/4)*stid(dir >> 2);
    res.center.y = center.y + (width/4)*stid((dir >> 1) & 1);
    res.center.x = center.x + (width/4)*stid(dir & 1);
    res.width = width / 2.0;
    if(res.width == 0) {
        exit(125);
    }
    return res;
}

void Region::to_string() {
    std::cout << "center : x " << center.x << ", y " << center.y << ", z " << center.z << ", r " << width << std::endl;
}

BHTree::BHTree(Region region) : region(region) {
    mass_center.mass = 0;
}

BHTree::BHTree(Region region, std::array<std::unique_ptr<BHTree>, 8>& c) : region(region) {
    mass_center.mass = 0;
    for(int i = 0; i < c.size(); i++) {
        mass_center += c[i]->mass_center;
        children[i] = std::move(c[i]);
        //children[i]->to_string();
    }
}

void BHTree::insert(Body b) {
    size_t dir = (b.pos.x > region.center.x) | ((b.pos.y > region.center.y) << 1) | ((b.pos.z > region.center.z) << 2);
    if(mass_center.mass == 0) {
        mass_center = b;
        return;
    }
    if(!is_leaf()) {
        mass_center += b;
        add(b);
    }
    else {
        add(mass_center);
        add(b);
        mass_center += b;
    }
}

void BHTree::add(Body b) {
    size_t dir = (b.pos.x > region.center.x) | ((b.pos.y > region.center.y) << 1) | ((b.pos.z > region.center.z) << 2);
    if(!children[dir]) {
        children[dir] = std::make_unique<BHTree>(region.get_sub(Direction(dir)));
        children[dir]->insert(b);
    }
    else {
        children[dir]->insert(b);
    }
}

bool BHTree::is_leaf() {
    for(auto& c : children) {
        if(c)
            return false;
    }
    return true;
}

int BHTree::height() {
    int res = 0;
    for(auto&c : children) {
        if(c) {
            int h = 1 + c->height();
            if(h > res) {
                res = h;
            }
        }
    }
    return res;
}

void BHTree::update_force(Body& b) {
    if(mass_center.mass == 0 || mass_center == b){
        return;
    }
    if(is_leaf()) {
        b.update_force(mass_center);
    }
    else {
        if(region.width / std::sqrt(mass_center.dist_sq(b)) < theta) {
            b.update_force(mass_center);
        }
        else {
            for(auto& c : children) {
                if(c)
                    c->update_force(b);
            }
        }
    }
}

void BHTree::to_string() {
    region.to_string();
}

void BHTree::print_tree() {
    std::ofstream out("out_tree.txt");
    explore(out, 0);
    /*std::cout << "Print tree" << std::endl;
    for(auto& c : children) {
        if(c) c->to_string();
    }*/
}

void BHTree::explore(std::ofstream& out, int depth) {
    for(int i=0; i< depth; i++)
        out << '\t';
    out << '[' << region.center.x << ',' << region.center.y << ',' << region.center.y << " " << region.width << " " << mass_center.pos.x << " " << mass_center.pos.y << " " << mass_center.pos.z << " " << mass_center.mass << ']' << std::endl;
    for(auto& c : children) {
        if(c)
            c->explore(out, depth+1);
    }
}
