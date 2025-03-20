#ifndef BODY_HPP
#define BODY_HPP

constexpr double G = 6.67e-11;

struct Vector3
{
    double x;
    double y;
    double z;

    double dist_sq(Vector3 const& v) const;

    Vector3 med(Vector3 const& v);

    Vector3 operator+(Vector3 const& v) const;

    Vector3 operator*(double f) const;

    void operator/=(double f);

    bool operator==(Vector3 const& v) const;
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
