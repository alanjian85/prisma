#ifndef PRISM_CORE_POINT_HPP
#define PRISM_CORE_POINT_HPP

#include <cassert>
#include <cmath>

#include <config/types.h>

#include "vector.hpp"

namespace prism {
    template <typename T>
    struct point2 {
        __device__ __host__ explicit point2(T t = 0) {
            x = y = t;
            assert(!has_nans());
        }

        __device__ __host__ point2(T x, T y)
                                : x(x), y(y) {
            assert(!has_nans());
        }

        __device__ __host__ bool has_nans() const {
            return false;
        }

        __device__ __host__ T &operator[](int i) {
            assert(i >= 0 && i <= 1);
            if (i == 0) return x;
            return y;
        }

        __device__ __host__ T operator[](int i) const {
            assert(i >= 0 && i <= 1);
            if (i == 0) return x;
            return y;
        }

        __device__ __host__ point2 operator-() const {
            return point2(-x, -y);
        }

        __device__ __host__ point2 &operator+=(vector2<T> v) {
            x += v.x;
            y += v.y;
            return *this;
        }

        __device__ __host__ point2 &operator-=(vector2<T> v) {
            x -= v.x;
            y -= v.y;
            return *this;
        }

        T x, y;
    };

    template <>
    __device__ __host__ bool point2<float>::has_nans() const {
        return isnan(x) || isnan(y);
    }

    template <>
    __device__ __host__ bool point2<double>::has_nans() const {
        return isnan(x) || isnan(y);
    }

    template <typename T>
    __device__ __host__ point2<T> operator+(point2<T> lhs, vector2<T> rhs) {
        return point2(lhs.x + rhs.x, lhs.y + rhs.y);
    }

    template <typename T>
    __device__ __host__ vector2<T> operator-(point2<T> lhs, point2<T> rhs) {
        return vector2<T>(lhs.x - rhs.x, lhs.y - rhs.y);
    }

    template <typename T>
    __device__ __host__ point2<T> operator-(point2<T> lhs, vector2<T> rhs) {
        return point2<T>(lhs.x - rhs.x, lhs.y - rhs.y);
    }

    using point2i = point2<int>;
    using point2f = point2<real_t>;

    template <typename T>
    struct point3 {
        __device__ __host__ explicit point3(T t = 0) {
            x = y = z = t;
            assert(!has_nans());
        }

        __device__ __host__ point3(T x, T y, T z)
                                : x(x), y(y), z(z) {
            assert(!has_nans());
        }

        __device__ __host__ bool has_nans() const {
            return false;
        }

        __device__ __host__ T &operator[](int i) {
            assert(i >= 0 && i <= 2);
            if (i == 0) return x;
            if (i == 1) return y;
            return z;
        }

        __device__ __host__ T operator[](int i) const {
            assert(i >= 0 && i <= 2);
            if (i == 0) return x;
            if (i == 1) return y;
            return z;
        }

        __device__ __host__ point3 operator-() const {
            return point3(-x, -y, -z);
        }

        __device__ __host__ point3 &operator+=(vector3<T> v) {
            x += v.x;
            y += v.y;
            z += v.z;
            return *this;
        }

        __device__ __host__ point3 &operator-=(vector3<T> v) {
            x -= v.x;
            y -= v.y;
            z -= v.z;
            return *this;
        }

        T x, y, z;
    };

    template <>
    __device__ __host__ bool point3<float>::has_nans() const {
        return isnan(x) || isnan(y) || isnan(z);
    }

    template <>
    __device__ __host__ bool point3<double>::has_nans() const {
        return isnan(x) || isnan(y) || isnan(z);
    }

    template <typename T>
    __device__ __host__ point3<T> operator+(point3<T> lhs, vector3<T> rhs) {
        return point3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
    }

    template <typename T>
    __device__ __host__ vector3<T> operator-(point3<T> lhs, point3<T> rhs) {
        return vector3<T>(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
    }

    template <typename T>
    __device__ __host__ point3<T> operator-(point3<T> lhs, vector3<T> rhs) {
        return point3<T>(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
    }

    using point3i = point3<int>;
    using point3f = point3<real_t>;
}

#endif // PRISM_CORE_POINT_HPP
