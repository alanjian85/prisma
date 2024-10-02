struct Ray {
    orig: vec3f,
    dir: vec3f
}

fn ray_at(ray: Ray, t: f32) -> vec3f {
    return ray.orig + t * ray.dir;
}