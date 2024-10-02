fn rand_init(id: vec2u, size: vec2u, frame: u32) -> u32 {
    let state = dot(id, vec2(1, size.x)) ^ jenkins_hash(frame);
    return jenkins_hash(state);
}

fn rand(state: ptr<function, u32>) -> f32 {
    *state = xorshift(*state);
    return u32_to_f32(*state);
}

fn rand_square(state: ptr<function, u32>) -> vec2f {
    return vec2(rand(state) - 0.5, rand(state) - 0.5);
}

fn rand_disk(state: ptr<function, u32>) -> vec2f {
    let r = sqrt(rand(state));
    let theta = 2.0 * PI * rand(state);
    return r * vec2(cos(theta), sin(theta));
}

fn rand_sphere(state: ptr<function, u32>) -> vec3f {
    let a = rand(state);
    let b = rand(state);
    let x = cos(2.0 * PI * a) * 2.0 * sqrt(b * (1 - b));
    let y = sin(2.0 * PI * a) * 2.0 * sqrt(b * (1 - b));
    let z = 1.0 - 2.0 * b;
    return vec3(x, y, z);
}

fn jenkins_hash(x: u32) -> u32 {
    var res = x + x << 10;
    res ^= res >> 6;
    res += res << 3;
    res ^= res >> 11;
    res += res << 15;
    return res;
}

fn xorshift(x: u32) -> u32 {
    var res = x ^ x << 13;
    res ^= x >> 17;
    res ^= x << 5;
    return res;
}

fn u32_to_f32(x: u32) -> f32 {
    return bitcast<f32>(0x3F800000 | (x >> 9)) - 1.0;
}