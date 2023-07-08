// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

#ifndef PRISM_CORE_VECTOR_HPP
#define PRISM_CORE_VECTOR_HPP

#include <cassert>
#include <cmath>

#include "utils.h"

template <typename T>
struct Vector2 {
    PRISM_CPU_GPU explicit Vector2 (T t = 0) {
        x = y = t;
        assert(!hasNaNs());
    }

    PRISM_CPU_GPU Vector2(T x, T y)
                      : x(x), y(y) {
       assert(!hasNaNs());
    }

    PRISM_CPU_GPU bool hasNaNs() const {
        return isnan(Real(x)) || isnan(Real(y));
    }

    PRISM_CPU_GPU Real lengthSquared() const {
        return x * x + y * y;
    }

    PRISM_CPU_GPU Real length() const {
       return sqrt(lengthSquared());
    }

    PRISM_CPU_GPU int maxDim() const {
        if (x > y) return 0;
        return 1;
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

    PRISM_CPU_GPU Vector2 operator-() const {
        return Vector2(-x, -y);
    }

    PRISM_CPU_GPU Vector2 &operator+=(Vector2 v) {
        x += v.x;
        y += v.y;
        return *this;
    }

    PRISM_CPU_GPU Vector2 &operator-=(Vector2 v) {
        x -= v.x;
        y -= v.y;
        return *this;
    }

    PRISM_CPU_GPU Vector2 &operator*=(T t) {
        x *= t;
        y *= t;
        return *this;
    }

    PRISM_CPU_GPU Vector2 &operator/=(T t) {
        assert(t != 0);
        Real inv = Real(1) / t;
        x *= inv;
        y *= inv;
        return *this;
    }

    PRISM_CPU_GPU bool normalized() const {
        return lengthSquared() == 1;
    }

    T x, y;
};

template <typename T>
PRISM_CPU_GPU Vector2<T> operator+(Vector2<T> lhs, Vector2<T> rhs) {
    return Vector2<T>(lhs.x + rhs.x, lhs.y + rhs.y);
}

template <typename T>
PRISM_CPU_GPU Vector2<T> operator-(Vector2<T> lhs, Vector2<T> rhs) {
    return Vector2<T>(lhs.x - rhs.x, lhs.y - rhs.y);
}

template <typename T>
PRISM_CPU_GPU Vector2<T> operator*(Vector2<T> lhs, T rhs) {
    return Vector2<T>(lhs.x * rhs, lhs.y * rhs);
}

template <typename T>
PRISM_CPU_GPU Vector2<T> operator*(T lhs, Vector2<T> rhs) {
    return rhs * lhs;
}

template <typename T>
PRISM_CPU_GPU Vector2<T> operator/(Vector2<T> lhs, T rhs) {
    assert(rhs != 0);
    Real inv = Real(1) / rhs;
    return Vector2<T>(lhs.x * inv, lhs.y * inv);
}

template <typename T>
PRISM_CPU_GPU Vector2<T> normalize(Vector2<T> v) {
    return v / v.length();
}

template <typename T>
PRISM_CPU_GPU Real dot(Vector2<T> lhs, Vector2<T> rhs) {
    return lhs.x * rhs.x + lhs.y * rhs.y;
}

template <typename T>
PRISM_CPU_GPU Vector2<T> permute(Vector2<T> v, int x, int y) {
    return Vector2<T>(v[x], v[y]);
}

template <typename T>
PRISM_CPU_GPU Vector2<T> abs(Vector2<T> v) {
    return Vector2<T>(abs(v.x), abs(v.y));
}

using Point2i = Vector2<int>;
using Point2f = Vector2<Real>;
using Vector2i = Vector2<int>;
using Vector2f = Vector2<Real>;

template <typename T>
struct Vector3 {
    PRISM_CPU_GPU explicit Vector3(T t = 0) {
        x = y = z = t;
        assert(!hasNaNs());
    }

    PRISM_CPU_GPU Vector3(T x, T y, T z)
                      : x(x), y(y), z(z) {
        assert(!hasNaNs());
    }

    PRISM_CPU_GPU bool hasNaNs() const {
        return isnan(Real(x)) || isnan(Real(y)) || isnan(Real(z));
    }

    PRISM_CPU_GPU Real lengthSquared() const {
        return x * x + y * y + z * z;
    }

    PRISM_CPU_GPU Real length() const {
        return sqrt(lengthSquared());
    }

    PRISM_CPU_GPU int maxDim() const {
        if (x > y) {
            if (x > z) return 0;
            else return 2;
        } else {
            if (y > z) return 1;
            else return 2;
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

    PRISM_CPU_GPU Vector3 operator-() const {
        return Vector3(-x, -y, -z);
    }

    PRISM_CPU_GPU Vector3 &operator=(Vector3 v) {
        x = v.x;
        y = v.y;
        z = v.z;
        return *this;
    }

    PRISM_CPU_GPU Vector3 &operator+=(Vector3 v) {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    PRISM_CPU_GPU Vector3 &operator-=(Vector3 v) {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }

    PRISM_CPU_GPU Vector3 &operator*=(T t) {
        x *= t;
        y *= t;
        z *= t;
        return *this;
    }

    PRISM_CPU_GPU Vector3 &operator/=(T t) {
        assert(t != 0);
        Real inv = Real(1) / t;
        x *= inv;
        y *= inv;
        z *= inv;
        return *this;
    }

    PRISM_CPU_GPU bool normalized() const {
        return lengthSquared() == 1;
    }

    T x, y, z;
    T &r = x, &g = y, &b = z;
};

template <typename T>
PRISM_CPU_GPU Vector3<T> operator+(Vector3<T> lhs, Vector3<T> rhs) {
    return Vector3<T>(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
}

template <typename T>
PRISM_CPU_GPU Vector3<T> operator-(Vector3<T> lhs, Vector3<T> rhs) {
    return Vector3<T>(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}

template <typename T>
PRISM_CPU_GPU Vector3<T> operator*(Vector3<T> lhs, T rhs) {
    return Vector3<T>(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs);
}

template <typename T>
PRISM_CPU_GPU Vector3<T> operator*(T lhs, Vector3<T> rhs) {
    return rhs * lhs;
}

template <typename T>
PRISM_CPU_GPU Vector3<T> operator/(Vector3<T> lhs, T rhs) {
    assert(rhs != 0);
    Real inv = Real(1) / rhs;
    return Vector3<T>(lhs.x * inv, lhs.y * inv, lhs.z * inv);
}

template <typename T>
PRISM_CPU_GPU Vector3<T> normalize(Vector3<T> v) {
    return v / v.length();
}

template <typename T>
PRISM_CPU_GPU T dot(Vector3<T> lhs, Vector3<T> rhs) {
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

template <typename T>
PRISM_CPU_GPU Vector3<T> cross(Vector3<T> lhs, Vector3<T> rhs) {
    return Vector3<T>(
        lhs.y * rhs.z - lhs.z * rhs.y,
        lhs.z * rhs.x - lhs.x * rhs.z,
        lhs.x * rhs.y - lhs.y * rhs.x
    );
}

template <typename T>
PRISM_CPU_GPU Vector3<T> permute(Vector3<T> v, int x, int y, int z) {
    return Vector3<T>(v[x], v[y], v[z]);
}

template <typename T>
PRISM_CPU_GPU Vector3<T> abs(Vector3<T> v) {
    return Vector3<T>(abs(v.x), abs(v.y), abs(v.z));
}

template <typename T>
PRISM_CPU_GPU Vector3<T> min(Vector3<T> lhs, Vector3<T> rhs) {
    return Vector3<T>(fmin(lhs.x, rhs.x),
                      fmin(lhs.y, rhs.y),
                      fmin(lhs.z, rhs.z));
}

template <typename T>
PRISM_CPU_GPU Vector3<T> max(Vector3<T> lhs, Vector3<T> rhs) {
    return Vector3<T>(fmax(lhs.x, rhs.x),
                      fmax(lhs.y, rhs.y),
                      fmax(lhs.z, rhs.z));
}

template <typename T>
PRISM_CPU_GPU Vector3<T> clamp(Vector3<T> v, Vector3<T> min, Vector3<T> max) {
    v = ::max(v, min);
    v = ::min(v, max);
    return v;
}

template <typename T>
using Point3 = Vector3<T>;
using Point3i = Vector3<int>;
using Point3f = Vector3<Real>;
using Vector3i = Vector3<int>;
using Vector3f = Vector3<Real>;
using Color = Vector3<Real>;

PRISM_CPU_GPU inline Vector3f reflect(Vector3f i, Vector3f n) {
    return i - Real(2.0) * n * dot(i, n);
}

template <typename T>
PRISM_CPU_GPU Color normalToColor(Vector3<T> n) {
    return Color(
        n.x * 0.5 + 0.5,
        n.y * 0.5 + 0.5,
        n.z * 0.5 + 0.5
    );
}

#endif // PRISM_CORE_POINT_HPP
