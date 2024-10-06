@group(2) @binding(0)
var<storage, read> vertices: array<Vertex>;

@group(2) @binding(1)
var<storage, read> offsets: array<u32>;

@group(2) @binding(2)
var<storage, read> material_indices: array<u32>;

struct Vertex {
    pos: vec3f,
    normal: vec3f,
    tex_coord: vec2f
}

struct Triangle {
    primitive: u32,
    v0: u32,
    v1: u32,
    v2: u32,
}

fn triangle_intersect(triangle: Triangle, ray: Ray,
    intersection: ptr<function, Intersection>, interval: Interval) -> bool {
    let offset = offsets[triangle.primitive];
    var v0 = vertices[triangle.v0 + offset].pos - ray.orig;
    var v1 = vertices[triangle.v1 + offset].pos - ray.orig;
    var v2 = vertices[triangle.v2 + offset].pos - ray.orig;

    let z = max_dim(abs(ray.dir));
    let x = (z + 1) % 3;
    let y = (x + 1) % 3;
    v0 = permute(v0, x, y, z);
    v1 = permute(v1, x, y, z);
    v2 = permute(v2, x, y, z);

    let sx = -ray.dir[x] / ray.dir[z];
    let sy = -ray.dir[y] / ray.dir[z];
    let sz = 1.0 / ray.dir[z];
    v0 += vec3(v0.z * vec2(sx, sy), 0.0);
    v1 += vec3(v1.z * vec2(sx, sy), 0.0);
    v2 += vec3(v2.z * vec2(sx, sy), 0.0);

    let e0 = v1.x * v2.y - v1.y * v2.x;
    let e1 = v2.x * v0.y - v2.y * v0.x;
    let e2 = v0.x * v1.y - v0.y * v1.x;
    if (e0 < 0 || e1 < 0 || e2 < 0) && (e0 > 0 || e1 > 0 || e2 > 0) {
        return false;
    }
    let det = e0 + e1 + e2;
    if det == 0 {
        return false;
    }

    v0.z *= sz;
    v1.z *= sz;
    v2.z *= sz;
    let t = (e0 * v0.z + e1 * v1.z + e2 * v2.z) / det;
    if !interval_surrounds(interval, t) {
        return false;
    }

    (*intersection).t = t;
    (*intersection).normal = normalize((e0 * vertices[triangle.v0 + offset].normal + e1 * vertices[triangle.v1 + offset].normal + e2 * vertices[triangle.v2 + offset].normal) / det);
    (*intersection).tex_coord = (e0 * vertices[triangle.v0 + offset].tex_coord + e1 * vertices[triangle.v1 + offset].tex_coord + e2 * vertices[triangle.v2 + offset].tex_coord) / det;
    (*intersection).material = material_indices[triangle.primitive];

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