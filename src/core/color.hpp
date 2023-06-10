#ifndef PRISM_CORE_COLOR_HPP
#define PRISM_CORE_COLOR_HPP

namespace prism {
    struct color {
        __device__ __host__ color(float r, float g, float b) : r(r), g(g), b(b) {}

        float r, g, b;
    };
}

#endif // PRISM_CORE_COLOR_HPP
