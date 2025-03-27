#include "Body.hpp"
#include <cmath>
#include <iostream>




float Vector3::dist_sq(Vector3 const & v) const {
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

Vector3 Vector3::operator-(const Vector3& v) const
{
    Vector3 res;
    res.x = x - v.x;
    res.y = y - v.y;
    res.z = z - v.z;
    return res;
}

Vector3 Vector3::operator+(const Vector3& v) const
{
    Vector3 res;
    res.simd = _mm_add_ps(simd, v.simd);
    return res;
}

void Vector3::operator+=(const Vector3 &v)
{
    x += v.x;
    y += v.y;
    z += v.z;

}

Vector3 Vector3::operator*(float f) const
{
    Vector3 res;
    res.x = f*x;
    res.y = f*y;
    res.z = f*z;
    return res;
}

void Vector3::operator/=(float f)
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

float Vector3::normSqr() const
{
    return x*x+y*y+z*z;
}

void Body::update_force(const Body &b)
{
    const Vector3 diff = b.pos-pos;

    float dij = diff.normSqr();

    if (dij < 1.0)
    {
        dij = 2.0;
    }
    else
    {
        dij = std::sqrt(dij);
        dij = 2.0 / (dij * dij * dij);
    }

    acceleration += diff * (dij * b.mass);
}

void Body::update_pos(double dt)
{
    spd += acceleration;
    pos += spd;
}

void Body::operator+=(const Body &b)
{
    double mt = mass + b.mass;
    pos = pos*mass + b.pos*b.mass;
    spd = {0,0,0};//spd*mass + b.spd*b.mass;
    acceleration = {0,0,0};//acc*mass + b.acc*b.mass;
    pos /= mt;
    mass += b.mass;
}

bool Body::operator==(const Body &b) const
{
    return b.pos == pos;
}
