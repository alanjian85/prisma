use crate::core::Ray;
use crate::utils;
use nalgebra::{Point2, Point3, Vector3};
use rand::rngs::ThreadRng;

pub struct Camera {
    pos: Point3<f64>,
    pix_orig: Point3<f64>,
    pix_delta_x: Vector3<f64>,
    pix_delta_y: Vector3<f64>,
}

impl Camera {
    pub fn new(width: u32, height: u32, pos: Point3<f64>, focal_len: f64) -> Self {
        let aspect_ratio = width as f64 / height as f64;
        let viewport_height = 2.0;
        let viewport_width = viewport_height * aspect_ratio;

        let pix_delta_x = Vector3::new(viewport_width, 0.0, 0.0) / width as f64;
        let pix_delta_y = Vector3::new(0.0, -viewport_height, 0.0) / height as f64;
        let pix_orig = pos + Vector3::new(0.0, 0.0, -focal_len)
            - width as f64 / 2.0 * pix_delta_x
            - height as f64 / 2.0 * pix_delta_y;

        Camera {
            pos,
            pix_orig,
            pix_delta_x,
            pix_delta_y,
        }
    }

    pub fn generate_ray(&self, rng: &mut ThreadRng, p: Point2<u32>) -> Ray {
        let offset = utils::rand_square_vec2(rng);
        let pix_pos = self.pix_orig
            + (p.x as f64 + offset.x) * self.pix_delta_x
            + (p.y as f64 + offset.y) * self.pix_delta_y;
        Ray::new(self.pos, pix_pos - self.pos)
    }
}
