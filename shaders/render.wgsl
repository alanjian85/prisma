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

struct Path {
    coefficient: vec3f,
    constant: vec3f,
}

@compute
@workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(render_target);
    var rand_state = rand_init(id.xy, size, sample);

    var ray = camera_gen_ray(scene.camera, id.xy, &rand_state);
    var paths = array<Path, 50>();
    var depth = 0u;
    for (; depth < MAX_DEPTH; depth++) {
        var intersection = Intersection();
        if scene_intersect(ray, &intersection) {
            intersection_flip_normal(&intersection, ray);
            let material = materials[intersection.material];
            let normal_in_tangent = sample_texture(material.normal_texture, intersection.tex_coord);
            let normal = normalize(normal_in_tangent.x * intersection.tangent +
                         normal_in_tangent.y * intersection.bitangent +
                         normal_in_tangent.z * intersection.normal);

            let wo = -normalize(ray.dir);
            var wi: vec3f;
            if rand(&rand_state) < 0.5 {
                wi = normalize(normal + rand_sphere(&rand_state));
            } else {
                var h = material_sample_h(intersection, &rand_state, normal);
                //if dot(wo, h) <= 0.0 {
                //    h = -h;
                //}
                wi = reflect(-wo, h);
                if dot(wo, h) <= 0.0 || dot(wo, wi) <= 0.0 {
                    paths[depth].coefficient = vec3(0.0, 0.0, 0.0);
                    paths[depth].constant = vec3(0.0, 0.0, 0.0);
                    break;
                }
            }
            let h = normalize(wo + wi);
            let pdf = 0.5 * (1.0 / PI + material_pdf(intersection, normal, wo, h));

            ray.orig = ray_at(ray, intersection.t);
            ray.dir = wi;

            paths[depth].coefficient = material_brdf(intersection, normal, wi, wo) / pdf;
            paths[depth].constant = vec3(0.0, 0.0, 0.0); //sample_texture(material.emissive_texture, intersection.tex_coord);
        } else {
            paths[depth].coefficient = sample_panorama(scene.hdri, normalize(ray.dir));
            paths[depth].constant = vec3(0.0, 0.0, 0.0);
            break;
        }
    }

    depth++;
    var color = vec3(1.0, 1.0, 1.0);
    for (; depth > 0; depth--) {
        let path = paths[depth - 1];
        color = path.coefficient * color + path.constant;
    }

    if !(color.r < 0.0 || 0.0 < color.r || color.r == 0.0) { color.r = 0.0; }
    if !(color.g < 0.0 || 0.0 < color.g || color.g == 0.0) { color.g = 0.0; }
    if !(color.b < 0.0 || 0.0 < color.b || color.b == 0.0) { color.b = 0.0; }

    if color.r == bitcast<f32>(0x7F800000) { color.r = 0.0; }
    if color.g == bitcast<f32>(0x7F800000) { color.g = 0.0; }
    if color.b == bitcast<f32>(0x7F800000) { color.b = 0.0; }

    if color.r == -bitcast<f32>(0x7F800000) { color.r = 0.0; }
    if color.g == -bitcast<f32>(0x7F800000) { color.g = 0.0; }
    if color.b == -bitcast<f32>(0x7F800000) { color.b = 0.0; }

    let prev_color = textureLoad(render_target, id.xy);
    textureStore(render_target, id.xy, prev_color + vec4(color, 1.0));
}
