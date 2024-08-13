const PI: f32 = 3.14159;

override MAX_DEPTH: u32;

@group(0) @binding(0)
var render_target: texture_storage_2d<rgba16float, read_write>;

@group(1) @binding(0)
var textures: binding_array<texture_2d<f32>>;

struct Camera {
    pos: vec3f,
    pix_orig: vec3f,
    pix_delta_x: vec3f,
    pix_delta_y: vec3f,
}

struct SceneUniform {
    camera: Camera,
    env_map: u32
}

@group(2) @binding(0)
var<uniform> scene: SceneUniform;

@group(2) @binding(1)
var<storage, read> primitives: array<Sphere>;

var<push_constant> sample: u32;

@compute
@workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(render_target);
    var rand_state = rand_init(id.xy, size, sample);

    var ray = generate_ray(size, id.xy, &rand_state);
    var color = vec3(1.0, 1.0, 1.0);
    for (var i = 0u; i < MAX_DEPTH; i++) {
        var intersection = Intersection();
        if scene_intersect(ray, &intersection) {
            intersection_flip_normal(&intersection, ray);
            ray.orig = ray_at(ray, intersection.t);
            ray.dir = intersection.normal + rand_sphere(&rand_state);
            color *= vec3(0.5, 0.5, 0.5);
        } else {
            let texture_size = textureDimensions(textures[scene.env_map]);

            let dir = normalize(ray.dir);
            let theta = acos(-dir.y);
            let phi = atan2(-dir.z, dir.x) + PI;

            let u = phi / (2.0 * PI);
            let v = theta / PI;

            let x = u32(u * f32(texture_size.x - 1));
            let y = u32((1.0 - v) * f32(texture_size.y - 1));

            color *= textureLoad(textures[scene.env_map], vec2(x, y), 0).xyz;
            break;
        }
    }

    let prev_color = textureLoad(render_target, id.xy);
    textureStore(render_target, id.xy, prev_color + vec4(color, 1.0));
}

struct Interval {
    min: f32,
    max: f32,
}

fn interval_surrounds(interval: Interval, x: f32) -> bool {
    return interval.min < x && x < interval.max;
}

fn rand_init(id: vec2u, size: vec2u, frame: u32) -> u32 {
    let state = dot(id, vec2(1, size.x)) ^ jenkins_hash(frame);
    return jenkins_hash(state);
}

fn rand(state: ptr<function, u32>) -> f32 {
    *state = xorshift(*state);
    return u32_to_f32(*state);
}

fn rand_sphere(state: ptr<function, u32>) -> vec3f {
    let a = rand(state);
    let b = rand(state);
    let x = cos(2 * PI * a) * 2 * sqrt(b * (1 - b));
    let y = sin(2 * PI * a) * 2 * sqrt(b * (1 - b));
    let z = 1 - 2 * b;
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

struct Ray {
    orig: vec3f,
    dir: vec3f
}

fn ray_at(ray: Ray, t: f32) -> vec3f {
    return ray.orig + t * ray.dir;
}

fn generate_ray(size: vec2u, pix: vec2u, rand_state: ptr<function, u32>) -> Ray {
    let camera = scene.camera;
    let x = f32(pix.x) + rand(rand_state) - 0.5;
    let y = f32(pix.y) + rand(rand_state) - 0.5;
    let pix_pos = camera.pix_orig + x * camera.pix_delta_x + y * camera.pix_delta_y;
    return Ray(camera.pos, pix_pos - camera.pos);
}

struct Intersection {
    t: f32,
    normal: vec3f,
    front: bool,
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

fn scene_intersect(ray: Ray, intersection: ptr<function, Intersection>) -> bool {
    (*intersection).t = 1000.0;
    var intersected = false;
    for (var i = 0u; i < arrayLength(&primitives); i++) {
        let interval = Interval(0.001, (*intersection).t);
        if sphere_intersect(primitives[i], ray, intersection, interval) {
            intersected = true;
        }
    }
    return intersected;
}

struct Sphere {
    center: vec3f,
    radius: f32,
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
    return true;
}