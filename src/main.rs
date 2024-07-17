mod camera;
mod cli;
mod core;
mod shapes;

use camera::Camera;
use clap::Parser;
use cli::{Cli, Size};
use core::{Intersect, Ray};
use image::{Rgb, RgbImage};
use indicatif::ProgressBar;
use nalgebra::{Point2, Point3, Vector3};
use palette::{LinSrgb, Srgb};
use shapes::Sphere;

fn ray_color(ray: &Ray) -> LinSrgb<f64> {
    let sphere = Sphere::new(Point3::new(0.0, 0.0, -1.0), 0.5);
    if let Some(t) = sphere.intersect(ray) {
        let normal = (ray.at(t) - Point3::new(0.0, 0.0, -1.0)).normalize();
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

    let camera = Camera::new(Point3::new(0.0, 0.0, 0.0), width, height, 1.0);

    for y in 0..height {
        for x in 0..width {
            let ray = camera.generate_ray(Point2::new(x, y));
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
