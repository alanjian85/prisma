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

struct Ray {
    orig: vec3f,
    dir: vec3f,
}

struct Intersection {
    normal: vec3f,
}

fn ray_at(ray: Ray, t: f32) -> vec3f {
    return ray.orig + t * ray.dir;
}

struct Sphere {
    center: vec3f,
    radius: f32,
}

fn sphere_intersect(sphere: Sphere, ray: Ray, intersection: ptr<function, Intersection>) -> bool {
    let oc = sphere.center - ray.orig;
    let a = dot(ray.dir, ray.dir);
    let b = dot(ray.dir, oc);
    let c = dot(oc, oc) - sphere.radius * sphere.radius;
    let discriminant = b * b - a * c;

    if discriminant >= 0.0 {
        let t = (b - sqrt(discriminant)) / a;
        (*intersection).normal = (ray_at(ray, t) - sphere.center) / sphere.radius;
        return true;
    } else {
        return false;
    }
}

const aspect_ratio = 16.0 / 9.0;

fn generate_ray(uv: vec2f) -> Ray {
    let viewport_height = 2.0;
    let viewport_width = aspect_ratio * viewport_height;
    let focus_dist = 1.0;

    let pix_delta_u = vec3(viewport_width, 0.0, 0.0);
    let pix_delta_v = vec3(0.0, viewport_height, 0.0);
    let pix_orig = vec3(0.0, 0.0, -focus_dist) - 0.5 * pix_delta_u - 0.5 * pix_delta_v;

    let pix_pos = pix_orig + uv.x * pix_delta_u + uv.y * pix_delta_v;
    return Ray(vec3(0.0, 0.0, 0.0), pix_pos);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let sphere = Sphere(vec3(0.0, 0.0, -1.0), 0.5);

    let ray = generate_ray(in.uv);
    var intersection = Intersection();
    if sphere_intersect(sphere, ray, &intersection) {
        return vec4(0.5 * (intersection.normal + 1.0), 1.0);
    } else {
        let alpha = 0.5 * (normalize(ray.dir).y + 1.0);
        return mix(vec4(1.0, 1.0, 1.0, 1.0), vec4(0.5, 0.7, 1.0, 1.0), alpha);
    }
}
