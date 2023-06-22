// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

#ifndef PRISM_CORE_INTERACTION_HPP
#define PRISM_CORE_INTERACTION_HPP

#include "utils.h"
#include "vector.hpp"

struct Interaction {
    PRISM_CPU_GPU void setNormal(Ray ray, Vector3f n) {
        this->n = normalize(n);
        if (dot(ray.d, n) > 0)
            this->n = -this->n;
    }

    Vector3f n;
};

#endif // PRISM_CORE_INTERACTION_HPP
