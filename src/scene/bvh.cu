// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

#include "bvh.hpp"

PRISM_CPU BVH::BVH(std::vector<Triangle> &primitives) : primitives(primitives) {
    if (primitives.empty()) {
        primitivesPtr = nullptr;
        nodesPtr = nullptr;
        return;
    }
    nodeCount = 0;
    auto root = recursiveBuild(primitives, 0, primitives.size());
    std::vector<BVHNode> nodes(nodeCount);
    size_t idx = 0;
    flattenBVHBuildTree(root.get(), nodes, idx);
    root.reset();
    this->nodes = nodes;
    primitivesPtr = thrust::raw_pointer_cast(this->primitives.data());
    nodesPtr = thrust::raw_pointer_cast(this->nodes.data());
}

PRISM_CPU_GPU bool BVH::intersect(const Ray &ray, Interaction &interaction) const {
    if (nodesPtr == nullptr)
        return false;
    size_t nodesToVisit[64];
    size_t toVisitIdx = 0, currIdx = 0;
    bool intersected = false;
    while (true) {
        BVHNode node = nodesPtr[currIdx];
        if (node.rightOffset == 0) {
            if (primitivesPtr[node.primitive].intersect(ray, interaction))
                intersected = true;
            if (toVisitIdx == 0) break;
            currIdx = nodesToVisit[--toVisitIdx];
        }
        else {
            if (node.bound.intersect(ray)) {
                nodesToVisit[toVisitIdx++] = node.rightOffset;
                ++currIdx;
            }
            else {
                if (toVisitIdx == 0) break;
                currIdx = nodesToVisit[--toVisitIdx];
            }
        }
    }
    return intersected;
}

PRISM_CPU auto BVH::recursiveBuild(std::vector<Triangle> &primitives, size_t begin, size_t end) -> std::unique_ptr<BVHBuildNode>{
    ++nodeCount;
    auto node = std::make_unique<BVHBuildNode>();
    Bound3f bound;
    for (size_t i = begin; i < end; ++i) {
        Bound3f primitiveBound = primitives[i].worldBound();
        bound = boundUnion(bound, primitiveBound);
    }
    node->bound = bound;
    if (end == begin + 1) {
        node->left = nullptr;
        node->right = nullptr;
        node->primitive = begin;
        return node;
    }
    size_t mid = (begin + end) / 2;
    node->left = recursiveBuild(primitives, begin, mid);
    node->right = recursiveBuild(primitives, mid, end);
    return node;
}

PRISM_CPU size_t BVH::flattenBVHBuildTree(BVHBuildNode *buildNode, std::vector<BVHNode> &nodes, size_t &idx) {
    size_t currIdx = idx++;
    BVHNode &node = nodes[currIdx];
    node.bound = buildNode->bound;
    if (buildNode->left == nullptr) {
        node.rightOffset = 0;
        node.primitive = buildNode->primitive;
        return currIdx;
    }
    flattenBVHBuildTree(buildNode->left.get(), nodes, idx);
    node.rightOffset = flattenBVHBuildTree(buildNode->right.get(), nodes, idx);
    return currIdx;
}
