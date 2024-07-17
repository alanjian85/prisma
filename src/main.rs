mod cli;
mod core;

use clap::Parser;
use cli::{Cli, Size};
use core::Ray;
use image::{Rgb, RgbImage};
use indicatif::ProgressBar;
use nalgebra::{Point3, Vector3};
use palette::{LinSrgb, Srgb};

fn hit_sphere(center: Point3<f64>, radius: f64, ray: &Ray) -> Option<f64> {
    let a = ray.dir.magnitude_squared();
    let b = (-2.0 * ray.dir).dot(&(center - ray.orig));
    let c = (center - ray.orig).magnitude_squared() - radius * radius;
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return None;
    }
    Some((-b - discriminant.sqrt()) / (2.0 * a))
}

fn ray_color(ray: &Ray) -> LinSrgb<f64> {
    let sphere_center = Point3::new(0.0, 0.0, -1.0);
    if let Some(t) = hit_sphere(sphere_center, 0.5, ray) {
        let normal = (ray.at(t) - sphere_center).normalize();
        let normal = 0.5 * (normal + Vector3::new(1.0, 1.0, 1.0));
        return LinSrgb::new(normal.x, normal.y, normal.z);
    }

    let dir = ray.dir.normalize();
    let alpha = 0.5 * (dir.y + 1.0);
    LinSrgb::new(1.0, 1.0, 1.0) * (1.0 - alpha) + LinSrgb::new(0.5, 0.7, 1.0) * alpha
}

fn main() {
    let cli = Cli::parse();
    let Size { width, height } = cli.size;

    let mut image = RgbImage::new(width, height);
    let progress_bar = ProgressBar::new(height as u64);

    let camera_pos = Point3::new(0.0, 0.0, 0.0);
    let camera_focal_len = 1.0;

    let viewport_height = 2.0;
    let viewport_width = viewport_height * (width as f64 / height as f64);

    let pixel_delta_x = Vector3::new(viewport_width, 0.0, 0.0) / width as f64;
    let pixel_delta_y = Vector3::new(0.0, -viewport_height, 0.0) / height as f64;
    let pixel_pos_orig = camera_pos + Vector3::new(0.0, 0.0, -camera_focal_len)
        - width as f64 / 2.0 * pixel_delta_x
        - height as f64 / 2.0 * pixel_delta_y;

    for y in 0..height {
        for x in 0..width {
            let pixel_pos = pixel_pos_orig + x as f64 * pixel_delta_x + y as f64 * pixel_delta_y;
            let ray = Ray::new(camera_pos, pixel_pos - camera_pos);
            let color: Srgb<f64> = Srgb::from_linear(ray_color(&ray));

            let r = (255.999 * color.red) as u8;
            let g = (255.999 * color.green) as u8;
            let b = (255.999 * color.blue) as u8;

            image.put_pixel(x, y, Rgb([r, g, b]));
        }
        progress_bar.inc(1);
    }

    progress_bar.finish();
    image.save(cli.output).unwrap();
}
