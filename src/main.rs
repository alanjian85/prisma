use clap::Parser;
use image::{Rgb, RgbImage};
use indicatif::ProgressBar;
use nalgebra::{Point2, Point3};
use palette::{LinSrgb, Srgb};
use prisma::cli::{Cli, Size};
use prisma::core::{Camera, Intersect, Ray, Scene};
use prisma::shapes::Sphere;
use prisma::utils;

fn compute_ray_color(ray: &Ray, scene: &Scene) -> LinSrgb<f64> {
    if let Some(intersection) = scene.intersect(ray, &(0.0..f64::INFINITY)) {
        return utils::unit_vec_to_rgb(intersection.normal).into();
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

    let camera = Camera::new(width, height, Point3::new(0.0, 0.0, 0.0), 1.0);
    let mut scene = Scene::new();
    scene.add(Box::new(Sphere::new(Point3::new(0.0, -100.5, -1.0), 100.0)));
    scene.add(Box::new(Sphere::new(Point3::new(0.0, 0.0, -1.0), 0.5)));

    for y in 0..height {
        for x in 0..width {
            let ray = camera.generate_ray(Point2::new(x, y));
            let color: Srgb<f64> = Srgb::from_linear(compute_ray_color(&ray, &scene));

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
