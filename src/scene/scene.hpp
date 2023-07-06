// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

#ifndef PRISM_SCENE_SCENE_HPP
#define PRISM_SCENE_SCENE_HPP

#include <cassert>
#include <cstddef>

#include "core/utils.h"

#include "bvh.hpp"

struct Scene {
public:
    PRISM_CPU Scene(std::vector<Triangle> &primitives) : bvh(primitives) {}

    PRISM_CPU static void *operator new(size_t count) {
        Scene *ptr;
        cudaError_t status = cudaMallocManaged(&ptr, count);
        assert(status == cudaSuccess);
        return ptr;
    }

    PRISM_CPU static void operator delete(void *ptr) {
        cudaError_t status = cudaFree(ptr);
        assert(status == cudaSuccess);
    }

    PRISM_GPU bool intersect(const Ray &ray, Interaction &interaction) const {
        return bvh.intersect(ray, interaction);
    }

private:
    BVH bvh;
};

#endif // PRISM_SCENE_SCENE_HPP
