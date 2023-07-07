// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

#include "utils.h"

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
