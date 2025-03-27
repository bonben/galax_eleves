#ifndef BODY_HPP
#define BODY_HPP

constexpr double G = 6.67e-11;

struct Vector3
{
    float x;
    float y;
    float z;

    double dist_sq(Vector3 const& v) const;

    Vector3 med(Vector3 const& v);

    Vector3 operator-(Vector3 const& v) const;

    Vector3 operator+(Vector3 const& v) const;

    Vector3 operator*(float f) const;

    void operator/=(float f);

    bool operator==(Vector3 const& v) const;
    float normSqr() const;
};

struct Body
{
    Vector3 pos;
    Vector3 spd;
    Vector3 force;

    double mass;

    double dist_sq(Body const& b) const;

    void update_force(Body const& b);

    void update_pos(double dt);

    void operator+=(Body const& b);

    bool operator==(Body const& b) const;
};

#endif // BODY_HPP
