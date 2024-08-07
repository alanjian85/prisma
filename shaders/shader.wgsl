override MAX_DEPTH: u32 = 50u;

@group(0) @binding(0)
var render_target: texture_storage_2d<rgba32float, read_write>;

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
            let material = intersection.material;
            ray.orig = ray_at(ray, intersection.t);
            if material.diffuse {
                color *= material.albedo;
                ray.dir = intersection.normal + rand_sphere(&rand_state);
            } else {
                let dir = normalize(ray.dir);
                let eta = select(material.ior, 1.0 / material.ior, intersection.front);

                let cosine = dot(dir, intersection.normal);
                let sine = sqrt(1.0 - cosine * cosine);

                if eta * sine >= 1.0 {
                    ray.dir = reflect(dir, intersection.normal);
                } else {
                    ray.dir = refract(dir, intersection.normal, eta);
                }
            }
        } else {
            let alpha = 0.5 * (normalize(ray.dir).y + 1.0);
            color *= mix(vec3(1.0, 1.0, 1.0), vec3(0.5, 0.7, 1.0), alpha);
            break;
        }
    }

    let prev_color = textureLoad(render_target, id.xy);
    textureStore(render_target, id.xy, prev_color + vec4(color, 0.001));
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
    let t = 2 * sqrt(b * (1 - b));
    let x = cos(radians(a * 360)) * t;
    let y = sin(radians(b * 360)) * t;
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
    let aspect_ratio = f32(size.x) / f32(size.y);
    let viewport_height = 2.0;
    let viewport_width = aspect_ratio * viewport_height;
    let focus_dist = 1.0;

    let pix_delta_u = vec3(viewport_width, 0.0, 0.0);
    let pix_delta_v = vec3(0.0, -viewport_height, 0.0);
    let pix_delta_x = pix_delta_u / f32(size.x);
    let pix_delta_y = pix_delta_v / f32(size.y);
    let pix_orig = vec3(0.0, 0.0, -focus_dist) - 0.5 * pix_delta_u - 0.5 * pix_delta_v;

    let x = f32(pix.x) + rand(rand_state) - 0.5;
    let y = f32(pix.y) + rand(rand_state) - 0.5;

    let pix_pos = pix_orig + x * pix_delta_x + y * pix_delta_y;
    return Ray(vec3(0.0, 0.0, 0.0), pix_pos);
}

struct Intersection {
    t: f32,
    normal: vec3f,
    front: bool,
    material: Material,
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

const NUM_SPHERES = 4;

struct Scene {
    spheres: array<Sphere, NUM_SPHERES>
}

fn scene_intersect(ray: Ray, intersection: ptr<function, Intersection>) -> bool {
    let material_ground = Material(true, vec3(0.8, 0.8, 0.8), 0.0);
    let material_ball = Material(false, vec3(0.0, 0.0, 0.0), 1.5);
    let material_bubble = Material(false, vec3(0.0, 0.0, 0.0), 1.0 / 1.5);

    var spheres = array(
        Sphere(vec3(0.0, -100.5, -1.0), 100.0, material_ground),
        Sphere(vec3(0.0, 0.0, -1.2), 0.5, material_ball),
        Sphere(vec3(0.0, 0.0, -1.2), 0.4, material_bubble),
    );

    (*intersection).t = 1000.0;
    var intersected = false;
    for (var i = 0; i < 3; i++) {
        let interval = Interval(0.001, (*intersection).t);
        if sphere_intersect(spheres[i], ray, intersection, interval) {
            intersected = true;
        }
    }
    return intersected;
}

struct Sphere {
    center: vec3f,
    radius: f32,
    material: Material,
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
    (*intersection).material = sphere.material;
    return true;
}

struct Material {
    diffuse: bool,
    albedo: vec3f,
    ior: f32,
}