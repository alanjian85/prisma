// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

#ifndef PRISM_CORE_COLOR_HPP
#define PRISM_CORE_COLOR_HPP

#include <cassert>
#include <cmath>

#include "utils.h"

struct Color {
    PRISM_CPU_GPU Color(Real r, Real g, Real b)
                      : r(r), g(g), b(b) {
        assert(!hasNaNs());
    }

    PRISM_CPU_GPU bool hasNaNs() {
        return isnan(r) || isnan(g) || isnan(b);
    }

    PRISM_CPU_GPU Color &operator*=(Color color) {
        r *= color.r;
        g *= color.g;
        b *= color.b;
        return *this;
    }

    Real r, g, b;
};

#endif // PRISM_CORE_COLOR_HPP
