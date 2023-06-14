// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

#ifndef PRISM_CORE_UTILS_H
#define PRISM_CORE_UTILS_H

#include <cmath>
#include <utility>

#include "config/types.h"

#define PRISM_KERNEL __global__
#define PRISM_CPU __host__
#define PRISM_GPU __device__
#define PRISM_CPU_GPU __device__ __host__

const Real pi = Real(3.141592653589793238463);

PRISM_CPU_GPU bool solveQuadraticEquation(Real a, Real b, Real c,
                                          Real &r1, Real &r2)
{
    Real discr = b * b - 4 * a * c;
    if (discr < 0)
        return false;
    discr = sqrt(discr);
    Real i2a = 1 / (2 * a);
    r1 = (-b - discr) * i2a;
    r2 = (-b + discr) * i2a;
    if (r1 > r2) {
        Real temp = r1;
        r1 = r2;
        r2 = temp;
    }
    return true;
}

PRISM_CPU_GPU Real radians(Real degrees) {
    return degrees / 180 * pi;
}

#endif // PRISM_CORE_UTILS_H
