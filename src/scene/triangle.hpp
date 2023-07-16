// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

#pragma once

#include "core/bounds.hpp"
#include "core/ray.hpp"
#include "core/utils.h"
#include "core/vector.hpp"
#include "interaction.hpp"

class Triangle {
public:
    Triangle() = default;

    PRISM_CPU_GPU Triangle(Point3f a, Point3f b, Point3f c)
                      : a(a), b(b), c(c)
    {
        n = normalize(cross(b - a, c - a));
    }

    PRISM_CPU_GPU bool intersect(const Ray &ray, Interaction &interaction) const;

    PRISM_CPU_GPU Bounds3f worldBound() const {
        return boundUnion(Bounds3f(a, b), c);
    }

private:
    Point3f a, b, c;
    Vector3f n;
};
