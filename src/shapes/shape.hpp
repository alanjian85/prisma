// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

#ifndef PRISM_SHAPES_SHAPE_HPP
#define PRISM_SHAPES_SHAPE_HPP

#include <core/interaction.hpp>
#include <core/ray.hpp>
#include <core/utils.h>

class Shape {
public:
    PRISM_CPU_GPU virtual bool intersect(const Ray &ray, Interaction &interaction) const = 0;
};

#endif // PRISM_SHAPES_SHAPE_HPP
