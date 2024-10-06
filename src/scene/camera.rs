use encase::ShaderType;
use glam::{Mat4, Vec3};

pub struct CameraBuilder {
    transform: Mat4,
    yfov: f32,
    aspect_ratio: Option<f32>,
}

impl CameraBuilder {
    pub fn new() -> CameraBuilder {
        Self::default()
    }

    pub fn transform(&mut self, transform: Mat4) -> &mut CameraBuilder {
        self.transform = transform;
        self
    }

    pub fn yfov(&mut self, yfov: f32) -> &mut CameraBuilder {
        self.yfov = yfov;
        self
    }

    pub fn aspect_ratio(&mut self, aspect_ratio: f32) -> &mut CameraBuilder {
        self.aspect_ratio = Some(aspect_ratio);
        self
    }

    pub fn build(&self, width: u32, height: u32) -> Camera {
        let aspect_ratio = if let Some(aspect_ratio) = self.aspect_ratio {
            aspect_ratio
        } else {
            width as f32 / height as f32
        };
        let viewport_height = 2.0 * (self.yfov / 2.0).tan();
        let viewport_width = aspect_ratio * viewport_height;

        let pix_du = Vec3::new(viewport_width, 0.0, 0.0);
        let pix_dv = Vec3::new(0.0, -viewport_height, 0.0);
        let pix_dx = pix_du / width as f32;
        let pix_dy = pix_dv / height as f32;
        let pix_orig = Vec3::new(0.0, 0.0, -1.0) - 0.5 * pix_du - 0.5 * pix_dv;

        Camera {
            transform: self.transform,
            pix_orig,
            pix_dx,
            pix_dy,
        }
    }
}

impl Default for CameraBuilder {
    fn default() -> Self {
        Self {
            transform: Mat4::IDENTITY,
            yfov: 90.0_f32.to_radians(),
            aspect_ratio: None,
        }
    }
}

#[derive(Default, ShaderType)]
pub struct Camera {
    transform: Mat4,
    pix_orig: Vec3,
    pix_dx: Vec3,
    pix_dy: Vec3,
}
