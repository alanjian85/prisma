use encase::ShaderType;
use glam::Vec3;

#[derive(Default, ShaderType, Copy, Clone)]
pub struct Aabb3 {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb3 {
    pub fn new() -> Self {
        Self {
            min: Vec3::INFINITY,
            max: Vec3::NEG_INFINITY,
        }
    }

    pub fn from_corners(a: Vec3, b: Vec3) -> Self {
        Self {
            min: a.min(b),
            max: a.max(b),
        }
    }

    pub fn union(&self, rhs: &Self) -> Self {
        let min = self.min.min(rhs.min);
        let max = self.max.max(rhs.max);
        Self { min, max }
    }

    pub fn union_point(&self, rhs: Vec3) -> Self {
        let min = self.min.min(rhs);
        let max = self.max.max(rhs);
        Self { min, max }
    }

    pub fn max_dim(&self) -> usize {
        let diff = self.max - self.min;
        if diff.x > diff.y && diff.x > diff.z {
            0
        } else if diff.y > diff.z {
            1
        } else {
            2
        }
    }

    pub fn centroid(&self) -> Vec3 {
        (self.min + self.max) / 2.0
    }
}
