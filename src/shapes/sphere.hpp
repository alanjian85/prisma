#ifndef PRISM_SHAPES_SPHERE_HPP
#define PRISM_SHAPES_SPHERE_HPP

#include <config/types.h>
#include <core/point.hpp>

#include "shape.hpp"

namespace prism {
    class sphere : public shape {
    public:
        PRISM_CPU_GPU sphere(point3f origin, real_t radius)
                          : origin(origin), radius(radius) {}

        PRISM_CPU_GPU bool intersect(const ray &r) const override {
            real_t a = dot(r.d, r.d);
            real_t b = 2 * (dot(vector3f(r.o), r.d) - dot(r.d, vector3f(origin)));
            real_t c = dot(vector3f(r.o), vector3f(r.o)) +
                       dot(vector3f(origin), vector3f(origin)) -
                       radius * radius;
            return solve_quadratic_equation(a, b, c);
        }

        point3f origin;
        real_t radius;
    };
}

#endif // PRISM_SHAPES_SPHERE_HPP
