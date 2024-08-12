pub struct Scene {
    env_map: u32,
}

impl Scene {
    pub fn new() -> Self {
        Self { env_map: 0 }
    }

    pub fn set_env(&mut self, env_map: u32) {
        self.env_map = env_map;
    }
}
