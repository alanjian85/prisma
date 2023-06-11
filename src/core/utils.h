// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

#ifndef PRISM_CORE_UTILS_H
#define PRISM_CORE_UTILS_H

#include "config/types.h"

#define PRISM_KERNEL __global__
#define PRISM_CPU __host__
#define PRISM_GPU __device__
#define PRISM_CPU_GPU __device__ __host__

namespace prism {
    const real_t pi = 3.141592653589793238463;

    PRISM_CPU_GPU bool solve_quadratic_equation(real_t a, real_t b, real_t c) {
        return b * b - 4 * a * c >= 0;
    }

    PRISM_CPU_GPU real_t radians(real_t degrees) {
        return degrees / 180 * pi;
    }
}

#endif // PRISM_CORE_UTILS_H
