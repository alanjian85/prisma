#ifndef PRISM_CORE_POINT_HPP
#define PRISM_CORE_POINT_HPP

namespace prism {
    template <typename T>
    struct point2 {
        __device__ __host__ point2(T x, T y) : x(x), y(y) {}

        T x, y;
    };

    using point2i = point2<int>;
    using point2f = point2<float>;
}

#endif // PRISM_CORE_POINT_HPP
