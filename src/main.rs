mod cli;

use clap::Parser;
use cli::{Cli, Size};
use image::{Rgb, RgbImage};
use indicatif::ProgressBar;

fn main() {
    let cli = Cli::parse();
    let Size { width, height } = cli.size;

    let mut image = RgbImage::new(width, height);
    let progress_bar = ProgressBar::new(height as u64);

    for y in 0..height {
        for x in 0..width {
            let r = x as f64 / (width - 1) as f64;
            let g = y as f64 / (height - 1) as f64;
            let b = 0.0;

            let r = (255.999 * r) as u8;
            let g = (255.999 * g) as u8;
            let b = (255.999 * b) as u8;

            image.put_pixel(x, y, Rgb([r, g, b]));
        }
        progress_bar.inc(1);
    }

    image.save(cli.output).unwrap();
    progress_bar.finish();
}
