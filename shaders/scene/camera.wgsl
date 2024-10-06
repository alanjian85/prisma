struct Camera {
    transform: mat4x4f,
    pix_orig: vec3f,
    pix_dx: vec3f,
    pix_dy: vec3f,
}

fn camera_gen_ray(camera: Camera, pix: vec2u, rand_state: ptr<function, u32>) -> Ray {
    let pix_xy = vec2f(pix) + rand_square(rand_state);
    let pix_pos = camera.pix_orig + pix_xy.x * camera.pix_dx + pix_xy.y * camera.pix_dy;
    return Ray((camera.transform * vec4(0.0, 0.0, 0.0, 1.0)).xyz, (camera.transform * vec4(pix_pos, 0.0)).xyz);
}