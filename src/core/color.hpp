#ifndef PRISM_CORE_COLOR_HPP
#define PRISM_CORE_COLOR_HPP

#include <config/types.h>

namespace prism {
    struct color {
        __device__ __host__ color(real_t r, real_t g, real_t b)
                                : r(r), g(g), b(b) {}

        float r, g, b;
    };
}

#endif // PRISM_CORE_COLOR_HPP
