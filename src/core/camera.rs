use crate::core::Ray;
use crate::utils;
use nalgebra::{Point2, Point3, Vector3};
use rand::prelude::*;

pub struct CameraBuilder {
    width: u32,
    height: u32,
    pos: Point3<f64>,
    center: Point3<f64>,
    up: Vector3<f64>,
    fov: f64,
    focus_dist: f64,
    lens_angle: f64,
}

impl CameraBuilder {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            pos: Point3::new(0.0, 0.0, 0.0),
            center: Point3::new(0.0, 0.0, -1.0),
            up: Vector3::new(0.0, 1.0, 0.0),
            fov: 90.0_f64.to_radians(),
            focus_dist: 1.0,
            lens_angle: 0.0,
        }
    }

    pub fn pos(&mut self, pos: Point3<f64>) -> &mut CameraBuilder {
        self.pos = pos;
        self
    }

    pub fn center(&mut self, center: Point3<f64>) -> &mut CameraBuilder {
        self.center = center;
        self
    }

    pub fn up(&mut self, up: Vector3<f64>) -> &mut CameraBuilder {
        self.up = up;
        self
    }

    pub fn fov(&mut self, fov: f64) -> &mut CameraBuilder {
        self.fov = fov;
        self
    }

    pub fn focus_dist(&mut self, focus_dist: f64) -> &mut CameraBuilder {
        self.focus_dist = focus_dist;
        self
    }

    pub fn lens_angle(&mut self, lens_angle: f64) -> &mut CameraBuilder {
        self.lens_angle = lens_angle;
        self
    }

    pub fn build(&self) -> Camera {
        let aspect_ratio = self.width as f64 / self.height as f64;
        let viewport_height = 2.0 * (self.fov / 2.0).tan() * self.focus_dist;
        let viewport_width = viewport_height * aspect_ratio;

        let front = (self.center - self.pos).normalize();
        let right = front.cross(&self.up).normalize();
        let up = right.cross(&front);

        let pix_delta_x = viewport_width / self.width as f64 * right;
        let pix_delta_y = viewport_height / self.height as f64 * -up;
        let pix_orig = self.pos + self.focus_dist * front
            - self.width as f64 / 2.0 * pix_delta_x
            - self.height as f64 / 2.0 * pix_delta_y;

        let lens_radius = self.focus_dist * (self.lens_angle / 2.0).tan();
        let lens_delta_x = lens_radius * right;
        let lens_delta_y = lens_radius * -up;

        Camera {
            pos: self.pos,
            pix_orig,
            pix_delta_x,
            pix_delta_y,
            lens_delta_x,
            lens_delta_y,
        }
    }
}

pub struct Camera {
    pos: Point3<f64>,
    pix_orig: Point3<f64>,
    pix_delta_x: Vector3<f64>,
    pix_delta_y: Vector3<f64>,
    lens_delta_x: Vector3<f64>,
    lens_delta_y: Vector3<f64>,
}

impl Camera {
    pub fn generate_ray(&self, rng: &mut ThreadRng, p: Point2<u32>) -> Ray {
        let offset = utils::rand_disk_vec2(rng);
        let ray_pos = self.pos + offset.x * self.lens_delta_x + offset.y * self.lens_delta_y;

        let offset = utils::rand_square_vec2(rng);
        let pix_pos = self.pix_orig
            + (p.x as f64 + offset.x) * self.pix_delta_x
            + (p.y as f64 + offset.y) * self.pix_delta_y;

        Ray::new(ray_pos, pix_pos - ray_pos)
    }
}
