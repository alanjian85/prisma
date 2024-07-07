use nalgebra::{Point3, Vector3};
use std::ops::{Add, Mul};

pub struct Ray {
    pub orig: Point3<f64>,
    pub dir: Vector3<f64>,
}

impl Ray {
    pub fn new(orig: Point3<f64>, dir: Vector3<f64>) -> Self {
        Self { orig, dir }
    }

    pub fn at(&self, t: f64) -> Point3<f64> {
        self.orig + t * self.dir
    }
}

pub struct Color {
    pub r: f64,
    pub g: f64,
    pub b: f64,
}

impl Color {
    pub fn new(r: f64, g: f64, b: f64) -> Self {
        Self { r, g, b }
    }

    pub fn from(vec: Vector3<f64>) -> Self {
        Self {
            r: vec.x,
            g: vec.y,
            b: vec.z,
        }
    }
}

impl Add for Color {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            r: self.r + rhs.r,
            g: self.g + rhs.g,
            b: self.b + rhs.b,
        }
    }
}

impl Mul<Color> for f64 {
    type Output = Color;

    fn mul(self, rhs: Color) -> Color {
        Color {
            r: self * rhs.r,
            g: self * rhs.g,
            b: self * rhs.b,
        }
    }
}
