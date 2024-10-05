//@group(2) @binding(0)
//var<storage, read> materials: array<Material>;

struct Material {
    diffuse: vec3f,
    emission: vec3f,
}

fn material_brdf(normal: vec3f, wi: vec3f, wo: vec3f) -> vec3f {
    let base_color = vec3(0.023, 0.023, 0.023);
    let metallic = 0.0;
    let roughness = 0.5;

    let h = normalize(wi + wo);
    let vdoth = dot(wo, h);
    let ndoth = dot(normal, h);
    let ndotl = dot(normal, wi);
    let ndotv = dot(normal, wo);

    let alpha = roughness * roughness;
    let f0 = mix(vec3(0.04), base_color, metallic);
    let f = f0 + (1.0 - f0) * pow(1.0 - vdoth, 5.0);

    let alpha2 = alpha * alpha;
    let diffuse = (1.0 - f) / PI * mix(base_color, vec3(0.0), metallic);
    var specular = f * microfacet_dist(alpha2, ndoth) * masking_shadowing(alpha2, ndotl, ndotv) / (4.0 * ndotl * ndotv);

    return diffuse + specular;
}

fn microfacet_dist(alpha2: f32, ndoth: f32) -> f32 {
    var denom = ndoth * ndoth * (alpha2 - 1.0) + 1.0;
    denom *= PI * denom;
    return alpha2 / denom;
}

fn masking_shadowing(alpha2: f32, ndotl: f32, ndotv: f32) -> f32 {
    let a = 2.0 * ndotl / (ndotl + sqrt(alpha2 + (1.0 - alpha2) * ndotl * ndotl));
    let b = 2.0 * ndotv / (ndotv + sqrt(alpha2 + (1.0 - alpha2) * ndotv * ndotv));
    return a * b;
}