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
    res.width_sqr = res.width*res.width;
    return res;
}

void Region::to_string() {
    std::cout << "center : x " << center.x << ", y " << center.y << ", z " << center.z << ", r " << width << std::endl;
}

BHTree::BHTree(Region region) : region(region) {
    mass_center.mass = 0;
    leaf = true;
}

BHTree::BHTree(Region region, std::array<std::unique_ptr<BHTree>, 8>& c) : region(region) {
    mass_center.mass = 0;
    for(int i = 0; i < c.size(); i++) {
        mass_center += c[i]->mass_center;
        children[i] = std::move(c[i]);
    }
    leaf = false;
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
        leaf = false;
        float quarter_width = region.width/4;
        float half_width = region.width/2;
        float width_sqr = half_width*half_width;
        for(int i = 0;i != 8;++i)
        {
            Region res;
            res.center.z = region.center.z + quarter_width*stid(i >> 2);
            res.center.y = region.center.y + quarter_width*stid((i >> 1) & 1);
            res.center.x = region.center.x + quarter_width*stid(i & 1);
            res.width = half_width;
            res.width_sqr = width_sqr;
            children[i] = std::make_unique<BHTree>(res);
        }
        add(mass_center);
        add(b);
        mass_center += b;
    }
}

void BHTree::add(Body b) {
    size_t dir = (b.pos.x > region.center.x) | ((b.pos.y > region.center.y) << 1) | ((b.pos.z > region.center.z) << 2);
    children[dir]->insert(b);
}

bool BHTree::is_leaf() {
    return leaf;
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
    if(is_leaf()) {
        if(mass_center != b && mass_center.mass != 0)
            b.update_force(mass_center);
    }
    else {
        if(region.width_sqr < theta_sqr*mass_center.dist_sq(b)) {
            b.update_force(mass_center);
        }
        else {
            children[0]->update_force(b);
            children[1]->update_force(b);
            children[2]->update_force(b);
            children[3]->update_force(b);
            children[4]->update_force(b);
            children[5]->update_force(b);
            children[6]->update_force(b);
            children[7]->update_force(b);
        }
    }
}

void BHTree::to_string() {
    region.to_string();
}

void BHTree::print_tree() {
    std::ofstream out("out_tree.txt");
    explore(out, 0);
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
