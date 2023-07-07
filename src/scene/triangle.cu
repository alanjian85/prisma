// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

#include "triangle.hpp"

PRISM_CPU_GPU bool Triangle::intersect(const Ray &ray, Interaction &interaction) const {
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
    Real a0 = bp.x * cp.y - bp.y * cp.x;
    Real a1 = cp.x * ap.y - cp.y * ap.x;
    Real a2 = ap.x * bp.y - ap.y * bp.x;
    if ((a0 < 0 || a1 < 0 || a2 < 0) && (a0 > 0 || a1 > 0 || a2 > 0))
        return false;
    Real a = a0 + a1 + a2;
    Real t = (ap.z * a0 + bp.z * a1 + cp.z * a2) / (ray.d[z] * a);
    if (t < ray.tMin || t > ray.tMax)
        return false;
    ray.tMax = t;
    interaction.n = n;
    interaction.p = (this->a * a0 + this->b * a1 + this->c * a2) / a;
    return true;
}
