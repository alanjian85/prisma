use encase::ShaderType;

use crate::{
    core::{Aabb3, Triangle},
    primitives::Primitives,
};

struct BvhNode {
    left: Option<Box<BvhNode>>,
    right: Option<Box<BvhNode>>,
    aabb: Aabb3,
    triangle_start: u32,
    triangle_end: u32,
}

impl BvhNode {
    pub fn new(
        primitives: &Primitives,
        triangles: &mut [Triangle],
        start: usize,
        end: usize,
    ) -> Self {
        if end - start == 1 {
            return Self {
                left: None,
                right: None,
                aabb: triangles[start].aabb(primitives),
                triangle_start: start as u32,
                triangle_end: start as u32 + 1,
            };
        }

        let mut centroid_aabb = Aabb3::new();
        for triangle in &triangles[start..end] {
            centroid_aabb = centroid_aabb.union_point(triangle.aabb(primitives).centroid());
        }

        let dim = centroid_aabb.max_dim();
        let mid = (centroid_aabb.min[dim] + centroid_aabb.max[dim]) / 2.0;
        let split_idx = start
            + itertools::partition(&mut triangles[start..end], |elem| {
                elem.aabb(primitives).centroid()[dim] < mid
            });

        if split_idx == start {
            return Self {
                left: None,
                right: None,
                aabb: triangles[start].aabb(primitives),
                triangle_start: start as u32,
                triangle_end: end as u32,
            };
        }

        let left = Box::new(Self::new(primitives, triangles, start, split_idx));
        let right = Box::new(Self::new(primitives, triangles, split_idx, end));
        let aabb = left.aabb.union(&right.aabb);
        Self {
            left: Some(left),
            right: Some(right),
            aabb,
            triangle_start: 0,
            triangle_end: 0,
        }
    }
}

#[derive(ShaderType)]
pub struct FlatBvhNode {
    aabb: Aabb3,
    right_idx: u32,
    triangle_start: u32,
    triangle_end: u32,
}

pub struct Bvh {
    root: Box<BvhNode>,
}

impl Bvh {
    pub fn new(primitives: &Primitives, triangles: &mut [Triangle]) -> Self {
        let len = triangles.len();
        let root = Box::new(BvhNode::new(primitives, triangles, 0, len));
        Self { root }
    }

    pub fn flatten(&self) -> Vec<FlatBvhNode> {
        let mut nodes = Vec::new();
        Self::flatten_node(&self.root, &mut nodes);
        nodes
    }

    fn flatten_node(node: &BvhNode, nodes: &mut Vec<FlatBvhNode>) {
        let curr_idx = nodes.len();
        nodes.push(FlatBvhNode {
            aabb: node.aabb,
            right_idx: 0,
            triangle_start: node.triangle_start,
            triangle_end: node.triangle_end,
        });

        if node.left.is_none() {
            return;
        }

        Self::flatten_node(node.left.as_ref().unwrap(), nodes);
        nodes[curr_idx].right_idx = nodes.len() as u32;

        Self::flatten_node(node.right.as_ref().unwrap(), nodes);
    }
}
