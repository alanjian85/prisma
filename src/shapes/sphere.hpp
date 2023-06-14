// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

#ifndef PRISM_SHAPES_SPHERE_HPP
#define PRISM_SHAPES_SPHERE_HPP

#include <config/types.h>
#include <core/vector.hpp>

#include "shape.hpp"

class Sphere : public Shape {
public:
    PRISM_CPU_GPU Sphere(Point3f o, Real r)
                      : o(o), r(r) {}

    PRISM_CPU_GPU bool intersect(const Ray &ray, Interaction &interaction) const override {
        Real a = dot(ray.d, ray.d);
        Real b = 2 * (dot(ray.o, ray.d) - dot(ray.d, o));
        Real c = dot(ray.o, ray.o) + dot(o, o) - r * r;
        Real t0, t1, t;
        if (!solveQuadraticEquation(a, b, c, t0, t1))
            return false;
        t = t0;
        if (t <= 0 || t >= ray.tMax) {
            t = t1;
            if (t <= 0 || t >= ray.tMax)
                return false;
        }
        ray.tMax = t;
        interaction.n = normalize(ray(t) - o);
        return true;
    }

private:
    Point3f o;
    Real r;
};

#endif // PRISM_SHAPES_SPHERE_HPP
