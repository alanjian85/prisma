// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

#ifndef PRISM_CORE_COLOR_HPP
#define PRISM_CORE_COLOR_HPP

#include <cassert>
#include <cmath>

#include <config/types.h>

#include "utils.h"

namespace prism {
    struct color {
        PRISM_CPU_GPU color(real_t r, real_t g, real_t b)
                                : r(r), g(g), b(b) {
            assert(!has_nans());
        }

        PRISM_CPU_GPU bool has_nans() {
            return isnan(r) || isnan(g) || isnan(b);
        }

        real_t r, g, b;
    };
}

#endif // PRISM_CORE_COLOR_HPP
