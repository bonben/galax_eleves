#ifndef BODY_HPP
#define BODY_HPP

#include <immintrin.h>
constexpr double G = 6.67e-11;

class Vector3
{
public:
    union {
        struct { float x, y, z, w; }; // w for padding/alignment
        __m128 simd; // SIMD register type
    };

    float dist_sq(Vector3 const& v) const;

    Vector3 med(Vector3 const& v);

    Vector3 operator-(Vector3 const& v) const;

    Vector3 operator+(Vector3 const& v) const;
    void operator+=(Vector3 const& v);

    Vector3 operator*(float f) const;

    void operator/=(float f);

    bool operator==(Vector3 const& v) const;
    float normSqr() const;
};

struct Body
{
    Vector3 pos;
    Vector3 spd;
    Vector3 acceleration;

    double mass;

    double dist_sq(Body const& b) const;

    void update_force(Body const& b);

    void update_pos();

    void operator+=(Body const& b);

    bool operator==(Body const& b) const;
};

#endif // BODY_HPP
