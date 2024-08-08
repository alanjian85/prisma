struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var vertices = array<VertexOutput, 3>(
        VertexOutput(vec4(-1.0, 3.0, 0.0, 1.0), vec2(0.0, 2.0)),
        VertexOutput(vec4(-1.0, -1.0, 0.0, 1.0), vec2(0.0, 0.0)),
        VertexOutput(vec4(3.0, -1.0, 0.0, 1.0), vec2(2.0, 0.0))
    );
    return vertices[in_vertex_index];
}

fn rand_init(uv: vec2u, res: vec2u, frame: u32) -> u32 {
    let state = dot(uv, vec2(1, res.x)) ^ jenkins_hash(frame);
    return jenkins_hash(state);
}

fn rand(state: ptr<function, u32>) -> f32 {
    *state = xorshift(*state);
    return u32_to_f32(*state);
}

fn rand_sphere(state: ptr<function, u32>) -> vec3f {
    var v = vec3f();
    loop {
        v = vec3(rand(state), rand(state), rand(state));
        if dot(v, v) <= 1.0 {
            break;
        }
    }
    return normalize(v);
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

struct Interval {
    min: f32,
    max: f32,
}

fn interval_surrounds(interval: Interval, x: f32) -> bool {
    return interval.min < x && x < interval.max;
}

struct Ray {
    orig: vec3f,
    dir: vec3f,
}

struct Intersection {
    t: f32,
    normal: vec3f,
    color: vec3f,
}

fn ray_at(ray: Ray, t: f32) -> vec3f {
    return ray.orig + t * ray.dir;
}

struct Sphere {
    center: vec3f,
    radius: f32,
    color: vec3f,
}

fn sphere_intersect(sphere: Sphere, ray: Ray, intersection: ptr<function, Intersection>, interval: Interval) -> bool {
    let oc = sphere.center - ray.orig;
    let a = dot(ray.dir, ray.dir);
    let b = dot(ray.dir, oc);
    let c = dot(oc, oc) - sphere.radius * sphere.radius;
    let discriminant = b * b - a * c;

    if discriminant < 0.0 {
        return false;
    }

    var t = (b - sqrt(discriminant)) / a;
    if !interval_surrounds(interval, t) {
        t = (b + sqrt(discriminant)) / a;
        if !interval_surrounds(interval, t) {
            return false;
        }
    }

    (*intersection).t = t;
    (*intersection).normal = (ray_at(ray, t) - sphere.center) / sphere.radius;
    (*intersection).color = sphere.color;
    return true;
}

fn generate_ray(uv: vec2f) -> Ray {
    let aspect_ratio = 16.0 / 9.0;
    let viewport_height = 2.0;
    let viewport_width = aspect_ratio * viewport_height;
    let focus_dist = 1.0;

    let pix_delta_u = vec3(viewport_width, 0.0, 0.0);
    let pix_delta_v = vec3(0.0, viewport_height, 0.0);
    let pix_orig = vec3(0.0, 0.0, -focus_dist) - 0.5 * pix_delta_u - 0.5 * pix_delta_v;

    let pix_pos = pix_orig + uv.x * pix_delta_u + uv.y * pix_delta_v;
    return Ray(vec3(0.0, 0.0, 0.0), pix_pos);
}

const NUM_SPHERES = 4;
const MAX_DEPTH = 50;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let width = 1920u;
    let height = 1080u;
    let frame = 0u;

    let x = u32(in.uv.x * f32(width));
    let y = u32(in.uv.y * f32(height));

    var rand_state = rand_init(vec2(x, y), vec2(width, height), frame);

    var spheres = array<Sphere, NUM_SPHERES>(
        Sphere(vec3(0.0, -100.5, -1.0), 100.0, vec3(0.8, 0.8, 0.0)),
        Sphere(vec3(0.0, 0.0, -1.2), 0.5, vec3(0.1, 0.2, 0.5)),
        Sphere(vec3(-1.0, 0.0, -1.0), 0.5, vec3(0.8, 0.8, 0.8)),
        Sphere(vec3(1.0, 0.0, -1.0), 0.5, vec3(0.8, 0.6, 0.2))
    );

    var ray = generate_ray(in.uv);
    var color = vec3(1.0, 1.0, 1.0);
    for (var i = 0; i < MAX_DEPTH; i++) {
        var intersection = Intersection();
        var intersected = false;
        intersection.t = 1000.0;
        for (var i = 0; i < NUM_SPHERES; i++) {
            let interval = Interval(0.001, intersection.t);
            if sphere_intersect(spheres[i], ray, &intersection, interval) {
                intersected = true;
            }
        }

        if intersected {
            let orig = ray_at(ray, intersection.t);
            let dir = normalize(reflect(ray.dir, intersection.normal)) + rand_sphere(&rand_state);
            ray = Ray(orig, dir);
            color *= intersection.color;
        } else {
            let alpha = 0.5 * (normalize(ray.dir).y + 1.0);
            color *= mix(vec3(1.0, 1.0, 1.0), vec3(0.5, 0.7, 1.0), alpha);
            return vec4(color, 1.0);
        }
    }

    return vec4(0.0, 0.0, 0.0, 1.0);
}
