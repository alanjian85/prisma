@group(1) @binding(0)
var<uniform> scene: SceneUniform;

@group(1) @binding(1)
var<storage, read> triangles: array<Triangle>;

@group(1) @binding(2)
var<storage, read> bvh_nodes: array<BvhNode>;

@group(1) @binding(3)
var<storage, read> transforms: array<Transform>;

struct SceneUniform {
    camera: Camera,
    hdri: u32
}

struct Transform {
    transform: mat4x4f,
    inv_trans: mat4x4f,
}

struct BvhNode {
    aabb: Aabb3,
    rigth_idx: u32,
    triangle_start: u32,
    triangle_end: u32,
}

fn scene_intersect(ray: Ray, intersection: ptr<function, Intersection>) -> bool {
    var stack = array<u32, 32>();
    var stack_ptr = 1u;
    stack[0] = 0u;
    (*intersection).t = bitcast<f32>(0x7F800000);

    var node = 0u;
    var intersected = false;
    loop {
        var interval = Interval(0.001, (*intersection).t);

        let left = node + 1;
        let right = bvh_nodes[node].rigth_idx;

        if right == 0 || !aabb_intersect(bvh_nodes[node].aabb, ray, interval) {
            if right == 0 {
                for (var i = bvh_nodes[node].triangle_start; i < bvh_nodes[node].triangle_end; i++) {
                    if triangle_intersect(triangles[i], ray, intersection, interval) {
                        intersected = true;
                        interval = Interval(0.001, (*intersection).t);
                    }
                }
            }
            stack_ptr--;
            node = stack[stack_ptr];
        } else {
            node = left;
            stack[stack_ptr] = right;
            stack_ptr++;
        }

        if node == 0 {
            break;
        }
    }

    return intersected;
}