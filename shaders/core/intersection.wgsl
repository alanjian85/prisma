struct Intersection {
    t: f32,
    normal: vec3f,
    front: bool,
    tex_coord: vec2f,
    transform: u32,
    material: u32,
}

fn intersection_flip_normal(intersection: ptr<function, Intersection>, ray: Ray) {
    let normal = (*intersection).normal;
    if dot(normal, ray.dir) < 0.0 {
        (*intersection).front = true;
    } else {
        (*intersection).front = false;
        (*intersection).normal = -normal;
    }
}