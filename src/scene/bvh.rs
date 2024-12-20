use crate::{core::Aabb3, primitives::Primitives, scene::TriangleInfo};
use encase::ShaderType;

struct BvhNode {
    left: Option<Box<BvhNode>>,
    right: Option<Box<BvhNode>>,
    aabb: Aabb3,
    triangle_start: u32,
    triangle_end: u32,
}

#[derive(Clone)]
struct Bucket {
    aabb: Aabb3,
    count: u32,
}

impl BvhNode {
    pub fn new(
        _primitives: &Primitives,
        triangle_infos: &mut [TriangleInfo],
        start: usize,
        end: usize,
    ) -> Self {
        if end - start == 1 {
            return Self {
                left: None,
                right: None,
                aabb: triangle_infos[start].aabb,
                triangle_start: start as u32,
                triangle_end: start as u32 + 1,
            };
        }

        if end - start <= 4 {
            let split_idx = (end + start) / 2;
            let left = Box::new(Self::new(_primitives, triangle_infos, start, split_idx));
            let right = Box::new(Self::new(_primitives, triangle_infos, split_idx, end));
            let aabb = left.aabb.union(&right.aabb);
            return Self {
                left: Some(left),
                right: Some(right),
                aabb,
                triangle_start: 0,
                triangle_end: 0,
            };
        }

        let mut centroid_aabb = Aabb3::new();
        for triangle_info in &triangle_infos[start..end] {
            centroid_aabb = centroid_aabb.union_point(triangle_info.aabb.centroid());
        }
        let dim = centroid_aabb.max_dim();
        let dim_extent = centroid_aabb.max[dim] - centroid_aabb.min[dim];

        let mut buckets = vec![
            Bucket {
                aabb: Aabb3::new(),
                count: 0,
            };
            12
        ];
        for triangle_info in &triangle_infos[start..end] {
            let mut bucket_idx = (buckets.len() as f32
                * ((triangle_info.centroid[dim] - centroid_aabb.min[dim]) / dim_extent))
                as usize;
            if bucket_idx == buckets.len() {
                bucket_idx = buckets.len() - 1;
            }

            buckets[bucket_idx].aabb = buckets[bucket_idx].aabb.union(&triangle_info.aabb);
            buckets[bucket_idx].count += 1;
        }

        let mut costs = Vec::new();
        for i in 0..(buckets.len() - 1) {
            let mut aabb0 = Aabb3::new();
            let mut count0 = 0;
            for bucket in buckets.iter().take(i + 1) {
                aabb0 = aabb0.union(&bucket.aabb);
                count0 += bucket.count;
            }

            let mut aabb1 = Aabb3::new();
            let mut count1 = 0;
            for bucket in buckets.iter().skip(i + 1) {
                aabb1 = aabb1.union(&bucket.aabb);
                count1 += bucket.count;
            }

            costs.push(0.125 + aabb0.area() * count0 as f32 + aabb1.area() * count1 as f32);
        }

        let split_bucket = costs
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let split_idx = start
            + itertools::partition(&mut triangle_infos[start..end], |triangle_info| {
                let mut bucket_idx = (buckets.len() as f32
                    * ((triangle_info.centroid[dim] - centroid_aabb.min[dim]) / dim_extent))
                    as usize;
                if bucket_idx == buckets.len() {
                    bucket_idx = buckets.len() - 1;
                }
                bucket_idx <= split_bucket
            });

        let left = Box::new(Self::new(_primitives, triangle_infos, start, split_idx));
        let right = Box::new(Self::new(_primitives, triangle_infos, split_idx, end));
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
    pub fn new(primitives: &Primitives, triangle_infos: &mut [TriangleInfo]) -> Self {
        let len = triangle_infos.len();
        let root = Box::new(BvhNode::new(primitives, triangle_infos, 0, len));
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
