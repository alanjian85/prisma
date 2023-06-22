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
        int z = abs(ray.d).maxDim();
        if (ray.d[z] == 0)
            return false;
        int x = (z + 1) % 3, y = (z + 2) % 3;
        ap = permute(ap, x, y, z);
        bp = permute(bp, x, y, z);
        cp = permute(cp, x, y, z);
        Real ix = ray.d[x] / ray.d[z];
        Real iy = ray.d[y] / ray.d[z];
        ap.x -= ap.z * ix;
        ap.y -= ap.z * iy;
        bp.x -= bp.z * ix;
        bp.y -= bp.z * iy;
        cp.x -= cp.z * ix;
        cp.y -= cp.z * iy;
        Real a0 = ap.x * bp.y - ap.y * bp.x;
        Real a1 = bp.x * cp.y - bp.y * cp.x;
        Real a2 = cp.x * ap.y - cp.y * ap.x;
        if (a0 > 0 && (a1 < 0 || a2 < 0) ||
            a0 < 0 && (a1 > 0 || a2 > 0))
            return false;
        Real a = a0 + a1 + a2;
        Real t = (cp.z * a0 + ap.z * a1 + bp.z * a2) / ray.d[z];
        if (a == 0 || a > 0 && t <= 0 || a < 0 && t >= 0)
            return false;
        t /= a;
        if (t > ray.tMax)
            return false;
        ray.tMax = t;
        interaction.n = n;
        return true;
    }

private:
    Point3f a, b, c;
    Vector3f n;
};

#endif // PRISM_SHAPES_TRIANGLE_HPP
