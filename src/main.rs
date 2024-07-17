use clap::Parser;
use image::{Rgb, RgbImage};
use indicatif::ProgressBar;
use nalgebra::{Point2, Point3};
use palette::{LinSrgb, Srgb};
use prisma::config::{Config, Size};
use prisma::core::{Camera, Ray, Scene};
use prisma::materials::Lambertian;
use prisma::primitives::Sphere;
use rand::rngs::ThreadRng;
use std::rc::Rc;

fn compute_ray_color(
    config: &Config,
    ray: &Ray,
    rng: &mut ThreadRng,
    scene: &Scene,
    depth: u32,
) -> LinSrgb<f64> {
    if depth > config.depth {
        return LinSrgb::new(0.0, 0.0, 0.0);
    }

    if let Some(intersection) = scene.intersect(ray, &(0.001..f64::INFINITY)) {
        let (ray, color) = intersection.material.scatter(rng, ray, &intersection);
        return color * compute_ray_color(&config, &ray, rng, scene, depth + 1) * 0.5;
    }

    let dir = ray.dir.normalize();
    let alpha = 0.5 * (dir.y + 1.0);
    LinSrgb::new(1.0, 1.0, 1.0) * (1.0 - alpha) + LinSrgb::new(0.5, 0.7, 1.0) * alpha
}

fn main() {
    let config = Config::parse();
    let Size { width, height } = config.size;

    let mut image = RgbImage::new(width, height);
    let progress_bar = ProgressBar::new(height as u64);
    let mut rng = rand::thread_rng();

    let camera = Camera::new(width, height, Point3::new(0.0, 0.0, 0.0), 1.0);
    let mut scene = Scene::new();

    let material_ground = Lambertian::new(LinSrgb::new(0.8, 0.8, 0.0));
    let material_center = Lambertian::new(LinSrgb::new(0.1, 0.2, 0.5));

    scene.add(Box::new(Sphere::new(
        Point3::new(0.0, -100.5, -1.0),
        100.0,
        Rc::new(material_ground),
    )));
    scene.add(Box::new(Sphere::new(
        Point3::new(0.0, 0.0, -1.0),
        0.5,
        Rc::new(material_center),
    )));

    for y in 0..height {
        for x in 0..width {
            let mut color = LinSrgb::new(0.0, 0.0, 0.0);
            for _ in 0..config.samples {
                let ray = camera.generate_ray(&mut rng, Point2::new(x, y));
                color += compute_ray_color(&config, &ray, &mut rng, &scene, 0);
            }
            color /= config.samples as f64;
            let color: Srgb<f64> = Srgb::from_linear(color);

            let r = (255.999 * color.red) as u8;
            let g = (255.999 * color.green) as u8;
            let b = (255.999 * color.blue) as u8;

            image.put_pixel(x, y, Rgb([r, g, b]));
        }
        progress_bar.inc(1);
    }

    progress_bar.finish();
    image.save(config.output).unwrap();
}
