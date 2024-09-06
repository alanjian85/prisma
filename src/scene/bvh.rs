use encase::ShaderType;

use super::{
    aabb::Aabb3,
    primitive::{Triangle, Vertex},
};

struct BvhNode {
    left: Option<Box<BvhNode>>,
    right: Option<Box<BvhNode>>,
    aabb: Aabb3,
    primitive_start: u32,
    primitive_end: u32,
}

impl BvhNode {
    pub fn new(vertices: &[Vertex], primitives: &mut [Triangle], start: usize, end: usize) -> Self {
        if end - start == 1 {
            return Self {
                left: None,
                right: None,
                aabb: primitives[start].aabb(vertices),
                primitive_start: start as u32,
                primitive_end: start as u32 + 1,
            };
        }

        let mut centroid_aabb = Aabb3::new();
        for primitive in &primitives[start..end] {
            centroid_aabb = centroid_aabb.union_point(primitive.aabb(vertices).centroid());
        }

        let dim = centroid_aabb.max_dim();
        if centroid_aabb.min[dim] == centroid_aabb.max[dim] {
            return Self {
                left: None,
                right: None,
                aabb: primitives[start].aabb(vertices),
                primitive_start: start as u32,
                primitive_end: end as u32,
            };
        }

        let mid = (centroid_aabb.min[dim] + centroid_aabb.max[dim]) / 2.0;
        let split_idx = start
            + itertools::partition(&mut primitives[start..end], |elem| {
                elem.aabb(vertices).centroid()[dim] < mid
            });

        let left = Box::new(Self::new(vertices, primitives, start, split_idx));
        let right = Box::new(Self::new(vertices, primitives, split_idx, end));
        let aabb = left.aabb.union(&right.aabb);
        Self {
            left: Some(left),
            right: Some(right),
            aabb,
            primitive_start: 0,
            primitive_end: 0,
        }
    }
}

#[derive(ShaderType)]
pub struct FlatBvhNode {
    aabb: Aabb3,
    right_idx: u32,
    primitive_start: u32,
    primitive_end: u32,
}

pub struct Bvh {
    root: Box<BvhNode>,
}

impl Bvh {
    pub fn new(vertices: &[Vertex], primitives: &mut [Triangle]) -> Self {
        let len = primitives.len();
        let root = Box::new(BvhNode::new(vertices, primitives, 0, len));
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
            primitive_start: node.primitive_start,
            primitive_end: node.primitive_end,
        });

        if node.left.is_none() {
            return;
        }

        Self::flatten_node(node.left.as_ref().unwrap(), nodes);
        nodes[curr_idx].right_idx = nodes.len() as u32;

        Self::flatten_node(node.right.as_ref().unwrap(), nodes);
    }
}
