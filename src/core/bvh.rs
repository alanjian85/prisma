use encase::ShaderType;

use crate::primitives::Triangle;

use super::Aabb3;

struct BvhNode {
    left: Option<Box<BvhNode>>,
    right: Option<Box<BvhNode>>,
    aabb: Aabb3,
    primitive: u32,
}

impl BvhNode {
    pub fn new(primitives: &mut [Triangle], start: usize, end: usize) -> Self {
        if end - start == 1 {
            return Self {
                left: None,
                right: None,
                aabb: primitives[start].aabb(),
                primitive: start as u32,
            };
        }

        let mut centroid_aabb = Aabb3::new();
        for primitive in &primitives[start..end] {
            centroid_aabb = centroid_aabb.union_point(primitive.aabb().centroid());
        }

        let dim = centroid_aabb.max_dim();
        if centroid_aabb.min[dim] == centroid_aabb.max[dim] {
            return Self {
                left: None,
                right: None,
                aabb: primitives[start].aabb(),
                primitive: start as u32,
            };
        }

        let mid = (centroid_aabb.min[dim] + centroid_aabb.max[dim]) / 2.0;
        let split_idx = start
            + itertools::partition(&mut primitives[start..end], |elem| {
                elem.aabb().centroid()[dim] < mid
            });

        let left = Box::new(Self::new(primitives, start, split_idx));
        let right = Box::new(Self::new(primitives, split_idx, end));
        let aabb = left.aabb.union(&right.aabb);
        Self {
            left: Some(left),
            right: Some(right),
            aabb,
            primitive: 0,
        }
    }
}

#[derive(ShaderType)]
pub struct FlatBvhNode {
    aabb: Aabb3,
    right_idx: u32,
    primitive: u32,
}

pub struct Bvh {
    root: Box<BvhNode>,
}

impl Bvh {
    pub fn new(primitives: &mut [Triangle]) -> Self {
        let len = primitives.len();
        let root = Box::new(BvhNode::new(primitives, 0, len));
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
            primitive: node.primitive,
        });

        if node.left.is_none() {
            return;
        }

        Self::flatten_node(node.left.as_ref().unwrap(), nodes);
        nodes[curr_idx].right_idx = nodes.len() as u32;

        Self::flatten_node(node.right.as_ref().unwrap(), nodes);
    }
}
