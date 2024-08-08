@group(0) @binding(0)
var render_target: texture_storage_2d<rgba8unorm, write>;

@compute
@workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(render_target);
    let u = f32(id.x) / f32(size.x);
    let v = f32(id.y) / f32(size.y);
    textureStore(render_target, id.xy, vec4(u, v, 0.0, 1.0));
}
