// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

#ifndef PRISM_CAMERAS_CAMERA_HPP
#define PRISM_CAMERAS_CAMERA_HPP

#include <core/film.hpp>

namespace prism {
    class camera {
    public:
        camera(int width, int height) : film(width, height) {}

        prism::film film;
    };
}

#endif // PRISM_CAMERAS_CAMERA_HPP
