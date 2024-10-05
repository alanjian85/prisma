///#include "core/intersection.wgsl"
///#include "core/ray.wgsl"
///#include "core/triangle.wgsl"

///#include "scene/camera.wgsl"
///#include "scene/scene.wgsl"

///#include "utils/aabb.wgsl"
///#include "utils/constants.wgsl"
///#include "utils/interval.wgsl"
///#include "utils/rand.wgsl"

///#include "materials.wgsl"
///#include "textures.wgsl"

override MAX_DEPTH: u32;

@group(0) @binding(0)
var render_target: texture_storage_2d<rgba32float, read_write>;

var<push_constant> sample: u32;

@compute
@workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(render_target);
    var rand_state = rand_init(id.xy, size, sample);

    var ray = camera_gen_ray(scene.camera, size, id.xy, &rand_state);
    var color = vec3(1.0, 1.0, 1.0);
    for (var i = 0u; i < MAX_DEPTH; i++) {
        var intersection = Intersection();
        if scene_intersect(ray, &intersection) {
            intersection_flip_normal(&intersection, ray);

            color = sample_texture(3u, intersection.tex_coord);
            break;

//            let wi = normalize(intersection.normal + rand_sphere(&rand_state));
//            let wo = -normalize(ray.dir);
//
//            ray.orig = ray_at(ray, intersection.t);
//            ray.dir = wi;
//
//            color *= material_brdf(intersection.normal, wi, wo) * PI;
        } else {
            color *= sample_panorama(scene.hdri, normalize(ray.dir));
            break;
        }
    }

    let prev_color = textureLoad(render_target, id.xy);
    textureStore(render_target, id.xy, prev_color + vec4(color, 1.0));
}