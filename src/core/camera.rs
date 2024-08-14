use encase::ShaderType;
use glam::Vec3;

pub struct CameraBuilder {
    pos: Vec3,
    center: Vec3,
    up: Vec3,
    fov: f32,
    focus_dist: f32,
    lens_angle: f32,
}

impl CameraBuilder {
    pub fn new() -> CameraBuilder {
        Self::default()
    }

    pub fn pos(&mut self, pos: Vec3) -> &mut CameraBuilder {
        self.pos = pos;
        self
    }

    pub fn center(&mut self, center: Vec3) -> &mut CameraBuilder {
        self.center = center;
        self
    }

    pub fn up(&mut self, up: Vec3) -> &mut CameraBuilder {
        self.up = up;
        self
    }

    pub fn fov(&mut self, fov: f32) -> &mut CameraBuilder {
        self.fov = fov;
        self
    }

    pub fn focus_dist(&mut self, focus_dist: f32) -> &mut CameraBuilder {
        self.focus_dist = focus_dist;
        self
    }

    pub fn lens_angle(&mut self, lens_angle: f32) -> &mut CameraBuilder {
        self.lens_angle = lens_angle;
        self
    }

    pub fn build(&self, width: u32, height: u32) -> Camera {
        let aspect_ratio = width as f32 / height as f32;
        let viewport_height = 2.0 * (self.fov / 2.0).tan() * self.focus_dist;
        let viewport_width = aspect_ratio * viewport_height;

        let front = (self.center - self.pos).normalize();
        let right = front.cross(self.up).normalize();
        let up = right.cross(front);

        let pix_delta_u = viewport_width * right;
        let pix_delta_v = viewport_height * -up;
        let pix_delta_x = pix_delta_u / width as f32;
        let pix_delta_y = pix_delta_v / height as f32;
        let pix_orig = self.pos + self.focus_dist * front - 0.5 * pix_delta_u - 0.5 * pix_delta_v;

        let lens_radius = (self.lens_angle / 2.0).tan() * self.focus_dist;
        let lens_delta_x = lens_radius * right;
        let lens_delta_y = lens_radius * -up;

        Camera {
            pos: self.pos,
            pix_orig,
            pix_delta_x,
            pix_delta_y,
            lens_radius,
            lens_delta_x,
            lens_delta_y,
        }
    }
}

impl Default for CameraBuilder {
    fn default() -> Self {
        Self {
            pos: Vec3::new(0.0, 0.0, 0.0),
            center: Vec3::new(0.0, 0.0, -1.0),
            up: Vec3::new(0.0, 1.0, 0.0),
            fov: 90.0_f32.to_radians(),
            focus_dist: 1.0,
            lens_angle: 0.0,
        }
    }
}

#[derive(Default, ShaderType)]
pub struct Camera {
    pos: Vec3,
    pix_orig: Vec3,
    pix_delta_x: Vec3,
    pix_delta_y: Vec3,
    lens_radius: f32,
    lens_delta_x: Vec3,
    lens_delta_y: Vec3,
}
