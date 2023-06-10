#ifndef PRISM_CORE_RAY_HPP
#define PRISM_CORE_RAY_HPP

#include "point.hpp"
#include "utils.h"
#include "vector.hpp"

namespace prism {
    struct ray {
        PRISM_CPU_GPU ray() = default;

        PRISM_CPU_GPU ray(point3f o, vector3f d) : o(o), d(d) {}

        point3f o;
        vector3f d;
    };
}

#endif // PRISM_CORE_RAY_HPP
