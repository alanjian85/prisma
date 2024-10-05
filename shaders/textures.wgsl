@group(3) @binding(0)
var textures: binding_array<texture_2d<f32>>;

fn sample_panorama(idx: u32, uvw: vec3f) -> vec3f {
    let texture_size = textureDimensions(textures[idx]);

    let theta = acos(-uvw.y);
    let phi = atan2(-uvw.z, uvw.x) + PI;

    let u = phi / (2.0 * PI);
    let v = theta / PI;

    let x = u32(u * f32(texture_size.x - 1));
    let y = u32((1.0 - v) * f32(texture_size.y - 1));

    return textureLoad(textures[idx], vec2(x, y), 0).xyz;
}

fn sample_texture(idx: u32, uv: vec2f) -> vec3f {
    let texture_size = textureDimensions(textures[idx]);

    let x = u32(fract(uv.x) * f32(texture_size.x - 1));
    let y = u32(fract(uv.y) * f32(texture_size.y - 1));

    return textureLoad(textures[idx], vec2(x, y), 0).xyz;
}