#ifndef BHTREE_H
#define BHTREE_H

#include <array>
#include <memory>
#include "../Body.hpp"
#include <vector>

constexpr double theta = 0.5;

constexpr double theta_sqr = theta*theta;

enum Direction : int {
    UNW,
    UNE,
    USW,
    USE,
    LNW,
    LNE,
    LSW,
    LSE
};

struct Region
{
    Vector3 center;
    double width;
    double width_sqr;
    Region get_sub(Direction dir);
    void to_string();
};

class BHTree
{
public:
    BHTree(Region region);
    BHTree(Region region, std::array<std::unique_ptr<BHTree>,8>& children);
    void insert(Body b);
    bool is_leaf();
    void update_force(Body& b);
    int height();
    void print_tree();
    void explore(std::ofstream& fout, int depth);
    Body mass_center;
    void to_string();
private:
    Region region;
    std::array<std::unique_ptr<BHTree>,8> children;
    bool leaf;
    void add(Body b);
};

#endif // BHTREE_H
