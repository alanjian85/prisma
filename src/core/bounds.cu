// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

#include "bounds.hpp"

PRISM_CPU_GPU bool Bounds3f::intersect(Ray ray) const {
    if (min.x > max.x || min.y > max.y || min.z > max.z)
        return false;
    Real t0 = ray.tMin, t1 = ray.tMax;
    for (int i = 0; i < 3; ++i) {
        Real tmin = (min[i] - ray.o[i]) / ray.d[i];
        Real tmax = (max[i] - ray.o[i]) / ray.d[i];
        if (tmin > tmax)
            cuda::std::swap(tmin, tmax);
        t0 = t0 < tmin ? tmin : t0;
        t1 = tmax < t1 ? tmax : t1;
        if (t0 > t1) return false;
    }
    return true;
}
