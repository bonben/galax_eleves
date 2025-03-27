#include "Body.hpp"
#include <cmath>
#include <iostream>




float Vector3::dist_sq(Vector3 const & v) const {
    /*__m128 diff = _mm_sub_ps(simd, v.simd);
    __m128 squared = _mm_mul_ps(diff, diff);
    return _mm_cvtss_f32(_mm_dp_ps(squared, squared, 0x71));*/

    return (v.x-x)*(v.x-x)+(v.y-y)*(v.y-y)+(v.z-z)*(v.z-z);
}

Vector3 Vector3::med(const Vector3 &v)
{
    Vector3 res;
    res.simd = _mm_mul_ps(_mm_add_ps(simd, v.simd), _mm_set1_ps(0.5f));
    return res;
}

Vector3 Vector3::operator-(const Vector3& v) const
{
    Vector3 res;
    res.simd = _mm_sub_ps(simd, v.simd);
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
    simd = _mm_add_ps(simd, v.simd);
}

Vector3 Vector3::operator*(float f) const
{
    Vector3 res;
    res.simd = _mm_mul_ps(simd, _mm_set1_ps(f));
    /*res.x = f*x;
    res.y = f*y;
    res.z = f*z;*/
    return res;
}

void Vector3::operator/=(float f)
{
    /*x /=f;
    y /=f;
    z /=f;*/
    simd = _mm_div_ps(simd, _mm_set1_ps(f));
}

bool Vector3::operator==(const Vector3 &v) const
{
    __m128 cmp = _mm_cmpeq_ps(simd, v.simd);
    return (_mm_movemask_ps(cmp) & 0x7) == 0x7; // Only check x/y/z
}

double Body::dist_sq(Body const & b) const {
    return pos.dist_sq(b.pos);
}

float Vector3::normSqr() const
{
    return x*x+y*y+z*z;
    //return _mm_cvtss_f32(_mm_dp_ps(simd, simd, 0x71));
}

float invSqrt( float number ){
    union {
        float f;
        int i;
    } conv;

    float x2;
    const float threehalfs = 1.5F;

    x2 = number * 0.5F;
    conv.f  = number;
    conv.i  = 0x5f3759df - ( conv.i >> 1 );
    conv.f  = conv.f * ( threehalfs - ( x2 * conv.f * conv.f ) );
    return conv.f;
}

void Body::update_force(const Body& b) {
    const Vector3 diff = b.pos - pos;

    const float dij_sq = diff.normSqr();
    //const float dij = invSqrt(dij_sq);

    const float inv_factor = (dij_sq >= 1.0f)
                                 ? (2.0f/ (dij_sq * _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(dij_sq)))))
                                 : 2.0f;

    // or ? (2.0f *dij*dij*dij)

    const float factor = b.mass * inv_factor;

    acceleration += diff * factor;
}

void Body::update_pos()
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
