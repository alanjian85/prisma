#ifndef PRISM_CORE_POINT_HPP
#define PRISM_CORE_POINT_HPP

#include <cassert>
#include <cmath>

#include <config/types.h>

#include "utils.h"
#include "vector.hpp"

namespace prism {
    template <typename T>
    struct point2 {
        PRISM_CPU_GPU explicit point2(T t = 0) {
            x = y = t;
            assert(!has_nans());
        }

        PRISM_CPU_GPU point2(T x, T y)
                                : x(x), y(y) {
            assert(!has_nans());
        }

        PRISM_CPU_GPU bool has_nans() const {
            return false;
        }

        template <typename U>
        PRISM_CPU_GPU explicit operator point2<U>() const {
            return point2<U>(x, y);
        }

        template <typename U>
        PRISM_CPU_GPU explicit operator vector2<U>() const {
            return vector2<U>(x, y);
        }

        PRISM_CPU_GPU T &operator[](int i) {
            assert(i >= 0 && i <= 1);
            if (i == 0) return x;
            return y;
        }

        PRISM_CPU_GPU T operator[](int i) const {
            assert(i >= 0 && i <= 1);
            if (i == 0) return x;
            return y;
        }

        PRISM_CPU_GPU point2 &operator+=(vector2<T> v) {
            x += v.x;
            y += v.y;
            return *this;
        }

        PRISM_CPU_GPU point2 &operator-=(vector2<T> v) {
            x -= v.x;
            y -= v.y;
            return *this;
        }

        T x, y;
    };

    template <>
    PRISM_CPU_GPU bool point2<real_t>::has_nans() const {
        return isnan(x) || isnan(y);
    }

    template <typename T>
    PRISM_CPU_GPU point2<T> operator+(point2<T> lhs, vector2<T> rhs) {
        return point2(lhs.x + rhs.x, lhs.y + rhs.y);
    }

    template <typename T>
    PRISM_CPU_GPU vector2<T> operator-(point2<T> lhs, point2<T> rhs) {
        return vector2<T>(lhs.x - rhs.x, lhs.y - rhs.y);
    }

    template <typename T>
    PRISM_CPU_GPU point2<T> operator-(point2<T> lhs, vector2<T> rhs) {
        return point2<T>(lhs.x - rhs.x, lhs.y - rhs.y);
    }

    template <typename T>
    PRISM_CPU_GPU real_t distance_squared(point2<T> a, point2<T> b) {
        return (a - b).length_squared();
    }

    template <typename T>
    PRISM_CPU_GPU real_t distance(point2<T> a, point2<T> b) {
        return (a - b).length();
    }

    template <typename T>
    PRISM_CPU_GPU point2<T> lerp(point2<T> a, point2<T> b, real_t t) {
        return (1 - t) * a + vector2<T>(t * b);
    }

    using point2i = point2<int>;
    using point2f = point2<real_t>;

    template <typename T>
    struct point3 {
        PRISM_CPU_GPU explicit point3(T t = 0) {
            x = y = z = t;
            assert(!has_nans());
        }

        PRISM_CPU_GPU point3(T x, T y, T z)
                                : x(x), y(y), z(z) {
            assert(!has_nans());
        }

        PRISM_CPU_GPU bool has_nans() const {
            return false;
        }

        template <typename U>
        PRISM_CPU_GPU explicit operator point2<U>() const {
            return point2<U>(x, y);
        }

        template <typename U>
        PRISM_CPU_GPU explicit operator point3<U>() const {
            return point3<U>(x, y, z);
        }

        template <typename U>
        PRISM_CPU_GPU explicit operator vector3<U>() const {
            return vector3<U>(x, y, z);
        }

        PRISM_CPU_GPU T &operator[](int i) {
            assert(i >= 0 && i <= 2);
            if (i == 0) return x;
            if (i == 1) return y;
            return z;
        }

        PRISM_CPU_GPU T operator[](int i) const {
            assert(i >= 0 && i <= 2);
            if (i == 0) return x;
            if (i == 1) return y;
            return z;
        }

        PRISM_CPU_GPU point3 &operator+=(vector3<T> v) {
            x += v.x;
            y += v.y;
            z += v.z;
            return *this;
        }

        PRISM_CPU_GPU point3 &operator-=(vector3<T> v) {
            x -= v.x;
            y -= v.y;
            z -= v.z;
            return *this;
        }

        T x, y, z;
    };

    template <>
    PRISM_CPU_GPU bool point3<real_t>::has_nans() const {
        return isnan(x) || isnan(y) || isnan(z);
    }

    template <typename T>
    PRISM_CPU_GPU point3<T> operator+(point3<T> lhs, vector3<T> rhs) {
        return point3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
    }

    template <typename T>
    PRISM_CPU_GPU vector3<T> operator-(point3<T> lhs, point3<T> rhs) {
        return vector3<T>(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
    }

    template <typename T>
    PRISM_CPU_GPU point3<T> operator-(point3<T> lhs, vector3<T> rhs) {
        return point3<T>(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
    }

    template <typename T>
    PRISM_CPU_GPU real_t distance_squared(point3<T> a, point3<T> b) {
        return (a - b).length_squared();
    }

    template <typename T>
    PRISM_CPU_GPU real_t distance(point3<T> a, point3<T> b) {
        return (a - b).length();
    }

    template <typename T>
    PRISM_CPU_GPU point3<T> lerp(point3<T> a, point3<T> b, real_t t) {
        return (1 - t) * a + vector3<T>(t * b);
    }

    using point3i = point3<int>;
    using point3f = point3<real_t>;
}

#endif // PRISM_CORE_POINT_HPP
