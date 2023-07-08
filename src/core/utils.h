// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

#ifndef PRISM_CORE_UTILS_H
#define PRISM_CORE_UTILS_H

#include <cmath>
#include <limits>
#include <utility>

#include "config/constants.h"
#include "config/types.h"

#define PRISM_KERNEL __global__
#define PRISM_CPU __host__
#define PRISM_GPU __device__
#define PRISM_CPU_GPU __device__ __host__

const Real pi = Real(3.141592653589793238463);

const Real inf = std::numeric_limits<Real>::infinity();

PRISM_CPU_GPU inline Real radians(Real degrees) {
    return degrees / 180 * pi;
}

#endif // PRISM_CORE_UTILS_H
