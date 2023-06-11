// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

#ifndef PRISM_CORE_VECTOR_HPP
#define PRISM_CORE_VECTOR_HPP

#include <cassert>
#include <cmath>

#include <config/types.h>

#include "utils.h"

namespace prism {
    template <typename T>
    struct vector2 {
        PRISM_CPU_GPU explicit vector2 (T t = 0) {
            x = y = t;
            assert(!has_nans());
        }

        PRISM_CPU_GPU vector2(T x, T y)
                                : x(x), y(y) {
            assert(!has_nans());
        }

        PRISM_CPU_GPU bool has_nans() const {
            return false;
        }

        PRISM_CPU_GPU real_t length_squared() const {
            return x * x + y * y;
        }

        PRISM_CPU_GPU real_t length() const {
            return sqrt(length_squared());
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

        PRISM_CPU_GPU vector2 operator-() const {
            return vector2(-x, -y);
        }

        PRISM_CPU_GPU vector2 &operator+=(vector2 v) const {
            x += v.x;
            y += v.y;
            return *this;
        }

        PRISM_CPU_GPU vector2 &operator-=(vector2 v) const {
            x -= v.x;
            y -= v.y;
            return *this;
        }

        PRISM_CPU_GPU vector2 &operator*=(T t) const {
            x *= t;
            y *= t;
            return *this;
        }

        PRISM_CPU_GPU vector2 &operator/=(T t) const {
            assert(t != 0);
            real_t inv = real_t(1) / t;
            x *= inv;
            y *= inv;
            return *this;
        }

        T x, y;
    };

    template <>
    PRISM_CPU_GPU bool vector2<real_t>::has_nans() const {
        return isnan(x) || isnan(y);
    }

    template <typename T>
    PRISM_CPU_GPU vector2<T> operator+(vector2<T> lhs, vector2<T> rhs) {
        return vector2<T>(lhs.x + rhs.x, lhs.y + rhs.y);
    }

    template <typename T>
    PRISM_CPU_GPU vector2<T> operator-(vector2<T> lhs, vector2<T> rhs) {
        return vector2<T>(lhs.x - rhs.x, lhs.y - rhs.y);
    }

    template <typename T>
    PRISM_CPU_GPU vector2<T> operator*(vector2<T> lhs, T rhs) {
        return vector2<T>(lhs.x * rhs, lhs.y * rhs);
    }

    template <typename T>
    PRISM_CPU_GPU vector2<T> operator*(T lhs, vector2<T> rhs) {
        return rhs * lhs;
    }

    template <typename T>
    PRISM_CPU_GPU vector2<T> operator/(vector2<T> lhs, T rhs) {
        assert(rhs != 0);
        real_t inv = real_t(1) / rhs;
        return vector2<T>(lhs.x * inv, lhs.y * inv);
    }

    template <typename T>
    PRISM_CPU_GPU vector2<T> normalize(vector2<T> v) {
        return v / v.length();
    }

    template <typename T>
    PRISM_CPU_GPU real_t dot(vector2<T> lhs, vector2<T> rhs) {
        return lhs.x * rhs.x + lhs.y * rhs.y;
    }

    using vector2i = vector2<int>;
    using vector2f = vector2<real_t>;

    template <typename T>
    struct vector3 {
        PRISM_CPU_GPU explicit vector3(T t = 0) {
            x = y = z = t;
            assert(!has_nans());
        }

        PRISM_CPU_GPU vector3(T x, T y, T z)
                                : x(x), y(y), z(z) {
            assert(!has_nans());
        }

        PRISM_CPU_GPU bool has_nans() const {
            return false;
        }

        PRISM_CPU_GPU real_t length_squared() const {
            return x * x + y * y + z * z;
        }

        PRISM_CPU_GPU real_t length() const {
            return sqrt(length_squared());
        }

        PRISM_CPU_GPU int max_dim() const {
            if (x > y) {
                if (x > z)
                    return 0;
                else
                    return 2;
            } else {
                if (y > z)
                    return 1;
                else
                    return 2;
            }
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

        PRISM_CPU_GPU vector3 operator-() const {
            return vector3(-x, -y, -z);
        }

        PRISM_CPU_GPU vector3 &operator+=(vector3 v) const {
            x += v.x;
            y += v.y;
            z += v.z;
            return *this;
        }

        PRISM_CPU_GPU vector3 &operator-=(vector3 v) const {
            x -= v.x;
            y -= v.y;
            z -= v.z;
            return *this;
        }

        PRISM_CPU_GPU vector3 &operator*=(T t) const {
            x *= t;
            y *= t;
            z *= t;
            return *this;
        }

        PRISM_CPU_GPU vector3 &operator/=(T t) const {
            assert(t != 0);
            real_t inv = real_t(1) / t;
            x *= inv;
            y *= inv;
            z *= inv;
            return *this;
        }

        T x, y, z;
    };

    template <>
    PRISM_CPU_GPU bool vector3<real_t>::has_nans() const {
        return isnan(x) || isnan(y) || isnan(z);
    }

    template <typename T>
    PRISM_CPU_GPU vector3<T> operator+(vector3<T> lhs, vector3<T> rhs) {
        return vector3<T>(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
    }

    template <typename T>
    PRISM_CPU_GPU vector3<T> operator-(vector3<T> lhs, vector3<T> rhs) {
        return vector3<T>(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
    }

    template <typename T>
    PRISM_CPU_GPU vector3<T> operator*(vector3<T> lhs, T rhs) {
        return vector3<T>(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs);
    }

    template <typename T>
    PRISM_CPU_GPU vector3<T> operator*(T lhs, vector3<T> rhs) {
        return rhs * lhs;
    }

    template <typename T>
    PRISM_CPU_GPU vector3<T> operator/(vector3<T> lhs, T rhs) {
        assert(rhs != 0);
        real_t inv = real_t(1) / rhs;
        return vector3<T>(lhs.x * inv, lhs.y * inv, lhs.z * inv);
    }

    template <typename T>
    PRISM_CPU_GPU vector3<T> normalize(vector3<T> v) {
        return v / v.length();
    }

    template <typename T>
    PRISM_CPU_GPU T dot(vector3<T> lhs, vector3<T> rhs) {
        return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
    }

    template <typename T>
    PRISM_CPU_GPU vector3<T> cross(vector3<T> lhs, vector3<T> rhs) {
        return vector3<T>(
            lhs.y * rhs.z - lhs.z * rhs.y,
            lhs.z * rhs.x - lhs.x * rhs.z,
            lhs.x * rhs.y - lhs.y * rhs.x
        );
    }

    template <typename T>
    PRISM_CPU_GPU vector3<T> permute(vector3<T> v, int x, int y, int z) {
        return vector3<T>(v[x], v[y], v[z]);
    }

    using vector3i = vector3<int>;
    using vector3f = vector3<real_t>;
}

#endif // PRISM_CORE_POINT_HPP
