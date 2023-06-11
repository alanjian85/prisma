// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

#ifndef PRISM_SHAPES_SPHERE_HPP
#define PRISM_SHAPES_SPHERE_HPP

#include <config/types.h>
#include <core/vector.hpp>

#include "shape.hpp"

namespace prism {
    class sphere : public shape {
    public:
        PRISM_CPU_GPU sphere(point3f origin, real_t radius)
                          : origin(origin), radius(radius) {}

        PRISM_CPU_GPU bool intersect(const ray &r, interaction &i) const override {
            real_t a = dot(r.d, r.d);
            real_t b = 2 * (dot(r.o, r.d) - dot(r.d, origin));
            real_t c = dot(r.o, r.o) + dot(origin, origin) - radius * radius;
            real_t t0, t1;
            if (!solve_quadratic_equation(a, b, c, t0, t1))
                return false;
            i.n = normalize(r(t0) - origin);
            return true;
        }

    private:
        point3f origin;
        real_t radius;
    };
}

#endif // PRISM_SHAPES_SPHERE_HPP
