// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

#ifndef PRISM_SHAPES_TRIANGLE_HPP
#define PRISM_SHAPES_TRIANGLE_HPP

#include <core/point.hpp>

#include "shape.hpp"

namespace prism {
    class triangle : public shape {
    public:
        PRISM_CPU_GPU triangle(point3f a, point3f b, point3f c)
                          : a(a), b(b), c(c) {
            n = normalize(cross(b - a, c - a));
        }

        PRISM_CPU_GPU bool intersect(const ray &r) const override {
            vector3 ao = a - r.o, bo = b - r.o, co = c - r.o;
            real_t a0 = co.x * ao.y - co.y * ao.x;
            real_t a1 = ao.x * bo.y - ao.y * bo.x;
            real_t a2 = bo.x * co.y - bo.y * co.x;
            return a0 > 0 && a1 > 0 && a2 > 0;
        }

    private:
        point3f a, b, c;
        vector3f n;
    };
}

#endif // PRISM_SHAPES_TRIANGLE_HPP
