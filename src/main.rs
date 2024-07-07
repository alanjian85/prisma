use image::{Rgb, RgbImage};
use indicatif::ProgressBar;

fn main() {
    let image_width = 256;
    let image_height = 256;

    let mut image = RgbImage::new(image_width, image_height);
    let progress_bar = ProgressBar::new(image_height as u64);

    for y in 0..image_height {
        for x in 0..image_width {
            let r = x as f64 / (image_width - 1) as f64;
            let g = y as f64 / (image_height - 1) as f64;
            let b = 0.0;

            let r = (255.999 * r) as u8;
            let g = (255.999 * g) as u8;
            let b = (255.999 * b) as u8;

            image.put_pixel(x, y, Rgb([r, g, b]));
        }
        progress_bar.inc(1);
    }

    image.save("output.png").unwrap();
    progress_bar.finish();
}
