// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include <thrust/device_vector.h>

#include "core/bounds.hpp"
#include "core/utils.h"

#include "triangle.hpp"

class BVH {
public:
    struct BVHNode {
        size_t rightOffset;
        Bounds3f bound;
        size_t primitive;
    };

    PRISM_CPU BVH(std::vector<Triangle> &primitives);

    PRISM_CPU_GPU bool intersect(const Ray &ray, Interaction &interaction) const;

private:
    struct BVHBuildNode {
        std::unique_ptr<BVHBuildNode> left, right;
        Bounds3f bound;
        size_t primitive;
    };

    PRISM_CPU std::unique_ptr<BVHBuildNode> recursiveBuild(std::vector<Triangle> &primitives, size_t begin, size_t end);

    PRISM_CPU size_t flattenBVHBuildTree(BVHBuildNode *buildNode, std::vector<BVHNode> &nodes, size_t &idx);

    thrust::device_vector<Triangle> primitives;
    thrust::device_vector<BVHNode> nodes;
    Triangle *primitivesPtr;
    BVHNode *nodesPtr;
    size_t nodeCount;
};
