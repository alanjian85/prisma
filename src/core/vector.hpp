#ifndef PRISM_CORE_VECTOR_HPP
#define PRISM_CORE_VECTOR_HPP

#include <cassert>
#include <cmath>

#include <config/types.h>

namespace prism {
    template <typename T>
    struct vector2 {
        __device__ __host__ explicit vector2 (T t = 0) {
            x = y = t;
            assert(!has_nans());
        }

        __device__ __host__ vector2(T x, T y)
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

        __device__ __host__ vector2 operator-() const {
            return vector2(-x, -y);
        }

        __device__ __host__ vector2 &operator+=(vector2 v) const {
            x += v.x;
            y += v.y;
            return *this;
        }

        __device__ __host__ vector2 &operator-=(vector2 v) const {
            x -= v.x;
            y -= v.y;
            return *this;
        }

        __device__ __host__ vector2 &operator*=(T t) const {
            x *= t;
            y *= t;
            return *this;
        }

        __device__ __host__ vector2 &operator/=(T t) const {
            assert(t != 0);
            double inv = 1.0 / t;
            x *= inv;
            y *= inv;
            return *this;
        }

        T x, y;
    };

    template <>
    __device__ __host__ bool vector2<float>::has_nans() const {
        return isnan(x) || isnan(y);
    }

    template <>
    __device__ __host__ bool vector2<double>::has_nans() const {
        return isnan(x) || isnan(y);
    }

    template <typename T>
    __device__ __host__ vector2<T> operator+(vector2<T> lhs, vector2<T> rhs) {
        return vector2<T>(lhs.x + rhs.x, lhs.y + rhs.y);
    }

    template <typename T>
    __device__ __host__ vector2<T> operator-(vector2<T> lhs, vector2<T> rhs) {
        return vector2<T>(lhs.x - rhs.x, lhs.y - rhs.y);
    }

    template <typename T>
    __device__ __host__ vector2<T> operator*(vector2<T> lhs, T rhs) {
        return vector2<T>(lhs.x * rhs, lhs.y * rhs);
    }

    template <typename T>
    __device__ __host__ vector2<T> operator*(T lhs, vector2<T> rhs) {
        return rhs * lhs;
    }

    template <typename T>
    __device__ __host__ vector2<T> operator/(vector2<T> lhs, T rhs) {
        assert(rhs != 0);
        double inv = 1.0 / rhs;
        return vector2<T>(lhs.x * inv, lhs.y * inv);
    }

    using vector2i = vector2<int>;
    using vector2f = vector2<real_t>;

    template <typename T>
    struct vector3 {
        __device__ __host__ explicit vector3(T t = 0) {
            x = y = z = t;
            assert(!has_nans());
        }

        __device__ __host__ vector3(T x, T y, T z)
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

        __device__ __host__ vector3 operator-() const {
            return vector3(-x, -y, -z);
        }

        __device__ __host__ vector3 &operator+=(vector3 v) const {
            x += v.x;
            y += v.y;
            z += v.z;
            return *this;
        }

        __device__ __host__ vector3 &operator-=(vector3 v) const {
            x -= v.x;
            y -= v.y;
            z -= v.z;
            return *this;
        }

        __device__ __host__ vector3 &operator*=(T t) const {
            x *= t;
            y *= t;
            z *= t;
            return *this;
        }

        __device__ __host__ vector3 &operator/=(T t) const {
            assert(t != 0);
            double inv = 1.0 / t;
            x *= inv;
            y *= inv;
            z *= inv;
            return *this;
        }

        T x, y, z;
    };

    template <>
    __device__ __host__ bool vector3<float>::has_nans() const {
        return isnan(x) || isnan(y) || isnan(z);
    }

    template <>
    __device__ __host__ bool vector3<double>::has_nans() const {
        return isnan(x) || isnan(y) || isnan(z);
    }

    template <typename T>
    __device__ __host__ vector3<T> operator+(vector3<T> lhs, vector3<T> rhs) {
        return vector3<T>(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
    }

    template <typename T>
    __device__ __host__ vector3<T> operator-(vector3<T> lhs, vector3<T> rhs) {
        return vector3<T>(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
    }

    template <typename T>
    __device__ __host__ vector3<T> operator*(vector3<T> lhs, T rhs) {
        return vector3<T>(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs);
    }

    template <typename T>
    __device__ __host__ vector3<T> operator*(T lhs, vector3<T> rhs) {
        return rhs * lhs;
    }

    template <typename T>
    __device__ __host__ vector3<T> operator/(vector3<T> lhs, T rhs) {
        assert(rhs != 0);
        double inv = 1.0 / rhs;
        return vector3<T>(lhs.x * inv, lhs.y * inv, lhs.z * inv);
    }

    using vector3i = vector3<int>;
    using vector3f = vector3<real_t>;
}

#endif // PRISM_CORE_POINT_HPP
