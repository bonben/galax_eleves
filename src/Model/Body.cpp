#include "Body.hpp"
#include <cmath>
#include <iostream>

double Vector3::dist_sq(Vector3 const & v) const {
    return (v.x-x)*(v.x-x) + (v.y-y)*(v.y-y) + (v.z-z)*(v.z-z);
}

Vector3 Vector3::med(const Vector3 &v)
{
    Vector3 res;
    res.x = (x+v.x) / 2;
    res.y = (y+v.y) / 2;
    res.z = (z+v.z) / 2;
    return res;
}

Vector3 Vector3::operator+(const Vector3& v) const
{
    Vector3 res;
    res.x = x + v.x;
    res.y = y + v.y;
    res.z = z + v.z;
    return res;
}

Vector3 Vector3::operator*(double f) const
{
    Vector3 res;
    res.x = f*x;
    res.y = f*y;
    res.z = f*z;
    return res;
}

void Vector3::operator/=(double f)
{
    x /= f;
    y /= f;
    z /= f;
}

bool Vector3::operator==(const Vector3 &v) const
{
    return v.x == x && v.y == y && v.z == z;
}

double Body::dist_sq(Body const & b) const {
    return pos.dist_sq(b.pos);
}

void Body::update_force(const Body &b)
{
    /*double d = dist_sq(b);
    double d2 = std::sqrt(d);
    if(d2 == 0) {
        exit(520);
    }
    //d2 = 10.0 / (d2*d2*d2);
    double F = mass * b.mass / d;
    force.x += F * (b.pos.x - pos.x) / d2;
    force.y += F * (b.pos.y - pos.y) / d2;
    force.z += F * (b.pos.z - pos.z) / d2;*/
    const float diffx = -(pos.x - b.pos.x);
    const float diffy = -(pos.y - b.pos.y);
    const float diffz = -(pos.z - b.pos.z);

    float dij = diffx * diffx + diffy * diffy + diffz * diffz;

    if (dij < 1.0)
    {
        dij = 2.0;
    }
    else
    {
        dij = std::sqrt(dij);
        dij = 2.0 / (dij * dij * dij);
    }

    force.x += diffx * dij * b.mass * mass;
    force.y += diffy * dij * b.mass * mass;
    force.z += diffz * dij * b.mass * mass;
}

void Body::update_pos(double dt)
{
    spd.x += force.x / mass;
    spd.y += force.y / mass;
    spd.z += force.z / mass;
    pos.x += spd.x;
    pos.y += spd.y;
    pos.z += spd.z;
}

void Body::operator+=(const Body &b)
{
    double mt = mass + b.mass;
    pos = pos*mass + b.pos*b.mass;
    spd = {0,0,0};//spd*mass + b.spd*b.mass;
    force = {0,0,0};//acc*mass + b.acc*b.mass;
    pos /= mt;
    mass += b.mass;
}

bool Body::operator==(const Body &b) const
{
    return b.pos == pos;
}
