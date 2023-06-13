// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

#ifndef PRISM_SHAPES_TRIANGLE_HPP
#define PRISM_SHAPES_TRIANGLE_HPP

#include <config/types.h>
#include <core/vector.hpp>

#include "shape.hpp"

class Triangle : public Shape {
public:
    PRISM_CPU_GPU Triangle(Point3f a, Point3f b, Point3f c)
                      : a(a), b(b), c(c)
    {
        n = normalize(cross(b - a, c - a));
    }

    PRISM_CPU_GPU bool intersect(const Ray &ray, Interaction &interaction) const override {
        Point3f ap = a - ray.o;
        Point3f bp = b - ray.o;
        Point3f cp = c - ray.o;
        int z = ray.d.maxDim();
        int x = (z + 1) % 3, y = (z + 2) % 3;
        ap = permute(ap, x, y, z);
        bp = permute(bp, x, y, z);
        cp = permute(cp, x, y, z);
        Real iz = 1 / ray.d.z;
        ap.z *= iz;
        ap.x -= ray.d.x * ap.z;
        ap.y -= ray.d.y * ap.z;
        bp.z *= iz;
        bp.x -= ray.d.x * bp.z;
        bp.y -= ray.d.y * bp.z;
        cp.z *= iz;
        cp.x -= ray.d.x * cp.z;
        cp.y -= ray.d.y * cp.z;
        Real a0 = ap.x * bp.y - ap.y * bp.x;
        Real a1 = bp.x * cp.y - bp.y * cp.x;
        Real a2 = cp.x * ap.y - cp.y * ap.x;
        if (a0 > 0 && a1 > 0 && a2 > 0) {
            interaction.n = n;
            return true;
        }
        return false;
    }

private:
    Point3f a, b, c;
    Vector3f n;
};

#endif // PRISM_SHAPES_TRIANGLE_HPP
