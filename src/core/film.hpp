// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

#ifndef PRISM_CORE_FILM_HPP
#define PRISM_CORE_FILM_HPP

#include <string>

#include <stb_image_write.h>

#include "color.hpp"
#include "utils.h"
#include "vector.hpp"

class Film {
public:
    PRISM_CPU_GPU Film(void *pixels, int width, int height)
                      : width_(width), height_(height) {
        assert(pixels != nullptr);
        this->pixels = static_cast<unsigned char*>(pixels);
        assert(width >= 0);
        assert(height >= 0);
    }

    PRISM_CPU_GPU int width() const {
        return width_;
    }

    PRISM_CPU_GPU int height() const {
        return height_;
    }

    PRISM_CPU_GPU void addSample(Point2i p, Color color) {
        int idx = p.y * width_ + p.x;
        pixels[idx * 3 + 0] = color.r * 255;
        pixels[idx * 3 + 1] = color.g * 255;
        pixels[idx * 3 + 2] = color.b * 255;
    }

    PRISM_CPU void writeImage(const std::string &path) const {
        stbi_write_png(path.c_str(), width_, height_, 3, pixels, 0);
    }

private:
    unsigned char *pixels;
    int width_, height_;
};

#endif // PRISM_CORE_FILM_HPP
