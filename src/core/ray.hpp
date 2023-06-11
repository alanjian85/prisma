// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

#ifndef PRISM_CORE_RAY_HPP
#define PRISM_CORE_RAY_HPP

#include "utils.h"
#include "vector.hpp"

namespace prism {
    struct ray {
        ray() = default;

        PRISM_CPU_GPU ray(point3f o, vector3f d) : o(o), d(d) {}

        point3f o;
        vector3f d;
    };
}

#endif // PRISM_CORE_RAY_HPP
