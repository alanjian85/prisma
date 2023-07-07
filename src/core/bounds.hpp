// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

#ifndef PRISM_CORE_BOUNDS_HPP
#define PRISM_CORE_BOUNDS_HPP

#include <cuda/std/utility>

#include "ray.hpp"
#include "utils.h"
#include "vector.hpp"

struct Bounds3f {
    PRISM_CPU_GPU Bounds3f() {
        min = Vector3f(inf, inf, inf);
        max = Vector3f(-inf, -inf, -inf);
    }

    PRISM_CPU_GPU Bounds3f(Vector3f a, Vector3f b) {
        min = ::min(a, b);
        max = ::max(a, b);
    }

    PRISM_CPU_GPU Vector3f diagonal() const {
        return max - min;
    }

    PRISM_CPU_GPU bool intersect(Ray ray) const;

    Vector3f min, max;
};

PRISM_CPU_GPU inline Bounds3f boundUnion(Bounds3f bound, Point3f point) {
    bound.min = ::min(bound.min, point);
    bound.max = ::max(bound.max, point);
    return bound;
}

PRISM_CPU_GPU inline Bounds3f boundUnion(Bounds3f bound, Bounds3f other) {
    bound.min = ::min(bound.min, other.min);
    bound.max = ::max(bound.max, other.max);
    return bound;
}

#endif // PRISM_CORE_BOUNDS_HPP
