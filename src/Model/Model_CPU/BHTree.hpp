#ifndef BHTREE_H
#define BHTREE_H

#include <array>
#include <memory>
#include "../Body.hpp"
#include <vector>

constexpr double theta = 1.0;

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
    void add(Body b);
};

#endif // BHTREE_H
