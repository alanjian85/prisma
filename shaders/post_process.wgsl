override NUM_SAMPLES: u32;

@group(0) @binding(0)
var src_texture: texture_storage_2d<rgba16float, read>;

@group(0) @binding(1)
var dst_texture: texture_storage_2d<rgba8unorm, write>;

@compute
@workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    var color = textureLoad(src_texture, id.xy).xyz / f32(NUM_SAMPLES);
    color /= color + vec3(1.0);
    color = pow(color, vec3(1.0 / 2.2));
    textureStore(dst_texture, id.xy, vec4(color, 1.0));
}
