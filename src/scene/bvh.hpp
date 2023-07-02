// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

#ifndef PRISM_SCENE_BVH_HPP
#define PRISM_SCENE_BVH_HPP

#include <cstddef>
#include <vector>

#include <thrust/device_vector.h>

#include <core/interaction.hpp>

#include "triangle.hpp"

class BVH {
public:
    PRISM_CPU BVH(std::vector<Triangle> &primitives) : primitives(primitives) {
        root = recursiveBuild(primitives, 0, primitives.size());
    }

    PRISM_CPU ~BVH() {
        freeBVHTree(root);
    }

    PRISM_CPU_GPU bool intersect(const Ray &ray, Interaction &interaction) const {
        return traverseBVHTree(root, ray, interaction);
    }

private:
    struct BVHNode {
        BVHNode *left, *right;
        Bound3f bound;
        thrust::device_vector<Triangle>::iterator primitive;
    };

    PRISM_CPU BVHNode *recursiveBuild(std::vector<Triangle> &primitives, size_t begin, size_t end)
    {
        BVHNode *node;
        cudaMallocManaged(&node, sizeof(BVHNode));
        Bound3f centroidBound, primitiveBound;
        for (size_t i = begin; i < end; ++i) {
            Bound3f bound = primitives[i].worldBound();
            centroidBound = boundUnion(centroidBound, 0.5 * bound.min + 0.5 * bound.max);
            primitiveBound = boundUnion(primitiveBound, bound);
        }
        node->bound = primitiveBound;
        if (end == begin + 1) {
            node->primitive = this->primitives.begin() + begin;
            return node;
        }
        int splitAxis = centroidBound.diagonal().maxDim();
        Vector3f center = 0.5 * centroidBound.min + 0.5 * centroidBound.max;
        size_t mid = std::partition(primitives.begin() + begin, primitives.begin() + end,
            [splitAxis, center](Triangle primitive) {
                Bound3f bound = primitive.worldBound();
                Vector3f centroid = 0.5 * bound.min + 0.5 * bound.max;
                return centroid[splitAxis] < center[splitAxis];
            }) - primitives.begin();
        node->left = recursiveBuild(primitives, begin, mid);
        node->right = recursiveBuild(primitives, mid, end);
        return node;
    }

    PRISM_CPU_GPU bool traverseBVHTree(BVHNode *node, const Ray &ray, Interaction &interaction) const {
        if (node->left == nullptr)
            return Triangle(*node->primitive).intersect(ray, interaction);
        if (!node->bound.intersect(ray))
            return false;
        bool intersected = false;
        if (traverseBVHTree(node->left, ray, interaction))
            intersected = true;
        if (traverseBVHTree(node->right, ray, interaction))
            intersected = true;
        return intersected;
    }

    PRISM_CPU void freeBVHTree(BVHNode *node) {
        if (node->left == nullptr) {
            cudaFree(node);
            return;
        }
        freeBVHTree(node->left);
        freeBVHTree(node->right);
    }

    BVHNode *root;
    thrust::device_vector<Triangle> primitives;
};

#endif // PRISM_SCENE_BVH_HPP
