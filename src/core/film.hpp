// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

#ifndef PRISM_CORE_FILM_HPP
#define PRISM_CORE_FILM_HPP

#include <string>

#include <stb_image_write.h>

#include "color.hpp"
#include "point.hpp"
#include "utils.h"

namespace prism {
    class film {
    public:
        film(int width, int height) : width(width), height(height) {
            assert(width >= 0);
            assert(height >= 0);
            cudaMallocManaged(&pixels, width * height * 3);
        }

        ~film() {
            cudaFree(pixels);
        }

        film(const film&) = delete;

        film &operator=(const film&) = delete;

        PRISM_GPU void add_sample(point2i p, color c) {
            int idx = p.y * width + p.x;
            pixels[idx * 3 + 0] = c.r * 255;
            pixels[idx * 3 + 1] = c.g * 255;
            pixels[idx * 3 + 2] = c.b * 255;
        }

        void write_image(const std::string &path) const {
            stbi_write_png(path.c_str(), width, height, 3, pixels, 0);
        }

        const int width, height;

    private:
        unsigned char *pixels;
    };
}

#endif // PRISM_CORE_FILM_HPP
