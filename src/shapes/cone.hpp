// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

#ifndef PRISM_SHAPES_CONE_HPP
#define PRISM_SHAPES_CONE_HPP

#include <config/types.h>
#include <core/vector.hpp>

#include "shape.hpp"

class Cone : public Shape {
public:
    PRISM_CPU_GPU Cone(Point3f o, Real yMin, Real yMax)
                      : o(o), yMin(yMin), yMax(yMax) {}

    PRISM_CPU_GPU bool intersect(const Ray &ray, Interaction &interaction) const override {
        Vector2f op(o.x, o.z);
        Vector2f ap(ray.o.x, ray.o.z);
        Vector2f bp(ray.d.x, ray.d.z);
        Real a = dot(bp, bp) - ray.d.y * ray.d.y;
        Real b = 2 * (dot(ap, bp) - dot(bp, op) - ray.o.y * ray.d.y + ray.d.y * o.y);
        Real c = dot(ap, ap) + dot(op, op) - 2 * dot(ap, op) -
                 ray.o.y * ray.o.y - o.y * o.y + 2 * ray.o.y * o.y;
        Real t0, t1, t;
        if (!solveQuadraticEquation(a, b, c, t0, t1))
            return false;
        t = t0;
        if (t <= 0 || t > ray.tMax) {
            t = t1;
            if (t <= 0 || t > ray.tMax)
                return false;
        }
        Vector3f p = ray(t);
        if (p.y < yMin || p.y > yMax)
            return false;
        Real r = Vector2f(p.x - o.x, p.z - o.z).length();
        if (p.y > o.y)
            interaction.n = normalize(Vector3f(p.x - o.x, -r, p.z - o.z));
        else
            interaction.n = normalize(Vector3f(p.x - o.x, r, p.z - o.z));
        return true; 
    }

private:
    Point3f o;
    Real yMin, yMax;
};

#endif // PRISM_SHAPES_CONE_HPP
