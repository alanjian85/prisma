const PI: f32 = 3.14159;

override MAX_DEPTH: u32;

@group(0) @binding(0)
var render_target: texture_storage_2d<rgba16float, read_write>;

@group(1) @binding(0)
var textures: binding_array<texture_2d<f32>>;

struct Material {
    ty: u32,
    albedo: vec3f,
    fuzziness: f32,
    ior: f32,
}

@group(2) @binding(0)
var<storage, read> materials: array<Material>;

struct Camera {
    pos: vec3f,
    pix_orig: vec3f,
    pix_delta_x: vec3f,
    pix_delta_y: vec3f,
    lens_radius: f32,
    lens_delta_x: vec3f,
    lens_delta_y: vec3f
}

struct SceneUniform {
    camera: Camera,
    env_map: u32
}

@group(3) @binding(0)
var<uniform> scene: SceneUniform;

struct Sphere {
    center: vec3f,
    radius: f32,
    material: u32,
}

@group(3) @binding(1)
var<storage, read> primitives: array<Sphere>;

struct Aabb3 {
    min: vec3f,
    max: vec3f,
}

fn aabb_intersect(aabb: Aabb3, ray: Ray, interval: Interval) -> bool {
    var t_interval = interval;
    for (var i = 0; i < 3; i++) {
        let inv_dir = 1.0 / ray.dir[i];

        let t0 = (aabb.min[i] - ray.orig[i]) * inv_dir;
        let t1 = (aabb.max[i] - ray.orig[i]) * inv_dir;

        if t0 < t1 {
            t_interval.min = max(t_interval.min, t0);
            t_interval.max = min(t_interval.max, t1);
        } else {
            t_interval.min = max(t_interval.min, t1);
            t_interval.max = min(t_interval.max, t0);
        }

        if t_interval.min >= t_interval.max {
            return false;
        }
    }
    return true;
}

struct BvhNode {
    aabb: Aabb3,
    rigth_idx: u32,
    primitive: u32,
}

@group(3) @binding(2)
var<storage, read> bvh_nodes: array<BvhNode>;

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
            let material = materials[intersection.material];

            ray.orig = ray_at(ray, intersection.t);
            if material.ty == 0 {
                ray.dir = intersection.normal + rand_sphere(&rand_state);
                color *= material.albedo;
            } else if material.ty == 1 {
                ray.dir = reflect(normalize(ray.dir), intersection.normal) + material.fuzziness * rand_sphere(&rand_state);
                if dot(ray.dir, intersection.normal) < 0.0 {
                    color *= vec3(0.0, 0.0, 0.0);
                    break;
                }
                color *= material.albedo;
            } else {
                let dir = normalize(ray.dir);
                let eta = select(material.ior, 1.0 / material.ior, intersection.front);

                let cosine = dot(-dir, intersection.normal);
                let sine = sqrt(1.0 - cosine * cosine);

                if eta * sine > 1.0 || rand(&rand_state) < reflectance(cosine, eta) {
                    ray.dir = reflect(dir, intersection.normal);
                } else {
                    ray.dir = refract(dir, intersection.normal, eta);
                }
            }
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

fn interval_overlaps(lhs: Interval, rhs: Interval) -> bool {
    return !(lhs.min > rhs.max || lhs.max < rhs.min);
}

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
    var v: vec2f;
    loop {
        v = rand_square(state);
        if dot(v, v) < 1.0 {
            break;
        }
    }
    return v;
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

    let ray_offset = camera.lens_radius * rand_disk(rand_state);
    let ray_pos = camera.pos + ray_offset.x * camera.lens_delta_x + ray_offset.y * camera.lens_delta_y;

    let pix_xy = vec2f(pix) + rand_square(rand_state);
    let pix_pos = camera.pix_orig + pix_xy.x * camera.pix_delta_x + pix_xy.y * camera.pix_delta_y;

    return Ray(ray_pos, pix_pos - ray_pos);
}

struct Intersection {
    t: f32,
    normal: vec3f,
    front: bool,
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

fn scene_intersect(ray: Ray, intersection: ptr<function, Intersection>) -> bool {
    var stack = array<u32, 32>();
    var stack_ptr = 1u;
    stack[0] = 0u;
    (*intersection).t = 1000.0;

    var node = 0u;
    var intersected = false;
    var idx = 0;
    loop {
        let interval = Interval(0.001, (*intersection).t);

        let left = node + 1;
        let right = bvh_nodes[node].rigth_idx;

        if right == 0 || !aabb_intersect(bvh_nodes[node].aabb, ray, interval) {
            if right == 0 && sphere_intersect(primitives[bvh_nodes[node].primitive], ray, intersection, interval) {
                intersected = true;
            }
            stack_ptr--;
            node = stack[stack_ptr];
        } else {
            node = left;
            stack[stack_ptr] = right;
            stack_ptr++;
        }

        idx++;

        if node == 0 {
            break;
        }
    }

    return intersected;
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

fn reflectance(cosine: f32, eta: f32) -> f32 {
    var r = (1.0 - eta) / (1.0 + eta);
    r *= r;
    return r + (1.0 - r) * pow(1.0 - cosine, 5.0);
}
