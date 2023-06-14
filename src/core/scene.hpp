// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

#ifndef PRISM_CORE_SCENE_HPP
#define PRISM_CORE_SCENE_HPP

#include <cstddef>

#include <thrust/device_vector.h>

#include <shapes/shape.hpp>

struct Scene {
public:
    Scene() = default;

    PRISM_CPU static void *operator new(std::size_t count) {
        Scene *ptr;
        cudaMallocManaged(&ptr, count);
        return ptr;
    }

    PRISM_CPU static void operator delete(void *ptr) {
        cudaFree(ptr);
    }

    PRISM_CPU void addShape(Shape *shape) {
        shapes.push_back(shape);
        begin = shapes.begin();
        end = shapes.end();
    }

    PRISM_CPU_GPU bool intersect(const Ray &ray, Interaction &interaction) const {
        bool intersected = false;
        for (auto iter = begin; iter != end; ++iter) {
            if (static_cast<Shape*>(*iter)->intersect(ray, interaction))
                intersected = true;
        }
        return intersected;
    }

    thrust::device_vector<Shape*> shapes;

private:
    thrust::device_vector<Shape*>::iterator begin, end;
};

#endif // PRISM_CORE_SCENE_HPP
