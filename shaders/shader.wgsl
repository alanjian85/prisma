struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var vertices = array<VertexOutput, 3>(
        VertexOutput(vec4<f32>(-1.0, 3.0, 0.0, 1.0), vec2<f32>(0.0, 2.0)),
        VertexOutput(vec4<f32>(-1.0, -1.0, 0.0, 1.0), vec2<f32>(0.0, 0.0)),
        VertexOutput(vec4<f32>(3.0, -1.0, 0.0, 1.0), vec2<f32>(2.0, 0.0))
    );
    return vertices[in_vertex_index];
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let u = in.uv.x;
    let v = in.uv.y;
    return vec4<f32>(u, v, 0.0, 1.0);
}
