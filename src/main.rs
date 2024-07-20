use clap::Parser;
use image::{Rgb, RgbImage};
use indicatif::ProgressBar;
use nalgebra::Point2;
use palette::{LinSrgb, Srgb};
use prisma::config::{Config, Size};
use prisma::core::{Ray, Scene};
use prisma::scripting::Scripting;
use rand::prelude::*;
use rayon::prelude::*;
use std::sync::Mutex;
use std::{error, fs};

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
        if let Some((ray, color)) = intersection.material.scatter(rng, ray, &intersection) {
            return color * compute_ray_color(config, &ray, rng, scene, depth + 1);
        }
        return LinSrgb::new(0.0, 0.0, 0.0);
    }

    let dir = ray.dir.normalize();
    let alpha = 0.5 * (dir.y + 1.0);
    LinSrgb::new(1.0, 1.0, 1.0) * (1.0 - alpha) + LinSrgb::new(0.5, 0.7, 1.0) * alpha
}

fn main() -> Result<(), Box<dyn error::Error>> {
    let config = Config::parse();
    let Size { width, height } = config.size;

    let scripting = Scripting::new()?;
    let script = fs::read_to_string(&config.script)?;
    let (camera, scene) = scripting.load(&config, &script)?;

    let image = Mutex::new(RgbImage::new(width, height));
    let progress_bar = ProgressBar::new(height as u64);

    (0..height).into_par_iter().for_each(|y| {
        let mut rng = rand::thread_rng();
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

            let mut image = image.lock().unwrap();
            image.put_pixel(x, y, Rgb([r, g, b]));
        }
        progress_bar.inc(1);
    });

    progress_bar.finish();
    image.lock().unwrap().save(config.output)?;
    Ok(())
}
