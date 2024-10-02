@group(3) @binding(0)
var<storage, read> vertices: array<Vertex>;

@group(3) @binding(1)
var<storage, read> offsets: array<u32>;

@group(3) @binding(2)
var<storage, read> material_indices: array<u32>;

struct Vertex {
    pos: vec3f,
    normal: vec3f,
}

struct Primitive {
    idx: u32,
    p0: u32,
    p1: u32,
    p2: u32,
}

fn primitive_intersect(primitive: Primitive, ray: Ray,
    intersection: ptr<function, Intersection>, interval: Interval) -> bool {
    let offset = offsets[primitive.idx];
    var p0 = vertices[primitive.p0 + offset].pos - ray.orig;
    var p1 = vertices[primitive.p1 + offset].pos - ray.orig;
    var p2 = vertices[primitive.p2 + offset].pos - ray.orig;

    let z = max_dim(abs(ray.dir));
    let x = (z + 1) % 3;
    let y = (x + 1) % 3;
    p0 = permute(p0, x, y, z);
    p1 = permute(p1, x, y, z);
    p2 = permute(p2, x, y, z);

    let sx = -ray.dir[x] / ray.dir[z];
    let sy = -ray.dir[y] / ray.dir[z];
    let sz = 1.0 / ray.dir[z];
    p0 += vec3(p0.z * vec2(sx, sy), 0.0);
    p1 += vec3(p1.z * vec2(sx, sy), 0.0);
    p2 += vec3(p2.z * vec2(sx, sy), 0.0);

    let e0 = p1.x * p2.y - p1.y * p2.x;
    let e1 = p2.x * p0.y - p2.y * p0.x;
    let e2 = p0.x * p1.y - p0.y * p1.x;
    if (e0 < 0 || e1 < 0 || e2 < 0) && (e0 > 0 || e1 > 0 || e2 > 0) {
        return false;
    }
    let det = e0 + e1 + e2;
    if det == 0 {
        return false;
    }

    p0.z *= sz;
    p1.z *= sz;
    p2.z *= sz;
    let t = (e0 * p0.z + e1 * p1.z + e2 * p2.z) / det;
    if !interval_surrounds(interval, t) {
        return false;
    }

    (*intersection).t = t;
    (*intersection).normal = normalize((e0 * vertices[primitive.p0 + offset].normal + e1 * vertices[primitive.p1 + offset].normal + e2 * vertices[primitive.p2 + offset].normal) / det);
    (*intersection).material = material_indices[primitive.idx];

    return true;
}

fn max_dim(v: vec3f) -> u32 {
    if v.x > v.y && v.x > v.z {
        return 0u;
    } else if v.y > v.z {
        return 1u;
    } else {
        return 2u;
    }
}

fn permute(v: vec3f, x: u32, y: u32, z: u32) -> vec3f {
    let vx = v[x];
    let vy = v[y];
    let vz = v[z];
    return vec3(vx, vy, vz);
}