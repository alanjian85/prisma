@group(3) @binding(0)
var<storage, read> materials: array<Material>;

struct Material {
    base_color_texture: u32,
    metallic_roughness_texture: u32,
    normal_texture: u32,
    emissive_texture: u32
}

fn material_brdf(intersection: Intersection, n: vec3f, wi: vec3f, wo: vec3f) -> vec3f {
    let material = materials[intersection.material];
    let base_color = sample_texture(material.base_color_texture, intersection.tex_coord);
    let metallic_roughness = sample_texture(material.metallic_roughness_texture, intersection.tex_coord);
    let metallic = metallic_roughness.b;
    let roughness = metallic_roughness.g;

    let h = normalize(wi + wo);
    let vdoth = dot(wo, h);
    let ndoth = dot(n, h);
    let ndotl = dot(n, wi);
    let ndotv = dot(n, wo);

    let alpha = roughness * roughness;
    let f0 = mix(vec3(0.04), base_color, metallic);
    let f = f0 + (1.0 - f0) * pow(1.0 - vdoth, 5.0);

    let alpha2 = alpha * alpha;
    let diffuse = (1.0 - f) / PI * mix(base_color, vec3(0.0), metallic);
    var specular = f * microfacet_dist(alpha2, ndoth) * masking_shadowing(alpha2, ndotl, ndotv) / (4.0 * ndotl * ndotv);

    return diffuse + specular;
}

fn material_sample_h(intersection: Intersection, state: ptr<function, u32>, n: vec3f) -> vec3f {
    let material = materials[intersection.material];
    let metallic_roughness = sample_texture(material.metallic_roughness_texture, intersection.tex_coord);
    let roughness = metallic_roughness.g;

    let alpha = roughness * roughness;
    let alpha2 = alpha * alpha;

    let rx = rand(state);
    let ry = rand(state);

    let phi = 2.0 * PI * rx;
    let cosine_phi = cos(phi);
    let sine_phi = sin(phi);

    let cosine_theta = sqrt((1.0 - ry) / (ry * (alpha2 - 1.0) + 1.0));
    let sine_theta = sqrt(1.0 - cosine_theta * cosine_theta);

    let x = cosine_phi * sine_theta;
    let y = cosine_theta;
    let z = sine_phi * sine_theta;

    let up = vec3(0.0, 1.0, 0.0);
    let t = normalize(cross(n, up));
    let b = cross(n, t);
    let h = normalize(x * t + y * n + z * b);
    return h;
}

fn material_pdf(intersection: Intersection, n: vec3f, wo: vec3f, h: vec3f) -> f32 {
    let material = materials[intersection.material];
    let metallic_roughness = sample_texture(material.metallic_roughness_texture, intersection.tex_coord);
    let roughness = metallic_roughness.g;

    let alpha = roughness * roughness;
    let alpha2 = alpha * alpha;
    let vdoth = dot(wo, h);
    let ndoth = dot(n, h);

    return microfacet_dist(alpha2, ndoth) / (4.0 * vdoth);
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