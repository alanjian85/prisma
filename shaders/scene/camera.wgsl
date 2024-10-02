struct Camera {
    pos: vec3f,
    pix_orig: vec3f,
    pix_delta_x: vec3f,
    pix_delta_y: vec3f,
    lens_delta_x: vec3f,
    lens_delta_y: vec3f
}

fn camera_gen_ray(camera: Camera, size: vec2u, pix: vec2u, rand_state: ptr<function, u32>) -> Ray {
    let ray_offset = rand_disk(rand_state);
    let ray_pos = camera.pos + ray_offset.x * camera.lens_delta_x + ray_offset.y * camera.lens_delta_y;

    let pix_xy = vec2f(pix) + rand_square(rand_state);
    let pix_pos = camera.pix_orig + pix_xy.x * camera.pix_delta_x + pix_xy.y * camera.pix_delta_y;

    return Ray(ray_pos, pix_pos - ray_pos);
}