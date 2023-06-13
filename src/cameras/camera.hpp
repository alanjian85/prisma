// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

#ifndef PRISM_CAMERAS_CAMERA_HPP
#define PRISM_CAMERAS_CAMERA_HPP

#include <core/film.hpp>
#include <core/ray.hpp>
#include <core/utils.h>

class Camera {
public:
    PRISM_CPU_GPU Camera(void *pixels, int width, int height)
                      : film(pixels, width, height) {}

    PRISM_CPU_GPU virtual Ray generateRay(Point2f p) const = 0;

    Film film;
};

#endif // PRISM_CAMERAS_CAMERA_HPP
