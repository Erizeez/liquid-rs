// Final output mask for a borderless transparent window.
//
// This is deliberately a separate present pass. The scene remains rectangular
// internally, while only the pixels handed to the platform surface receive the
// window's continuous-corner alpha mask.

@group(0) @binding(0) var u_source: texture_2d<f32>;
@group(0) @binding(1) var u_sampler: sampler;

struct WindowMaskUniform {
  size: vec2f,
  radius: f32,
  exponent: f32,
};

@group(0) @binding(2) var<uniform> u: WindowMaskUniform;

fn superellipse_distance(point: vec2f) -> f32 {
  let half_size = u.size * 0.5;
  let radius = min(u.radius, min(half_size.x, half_size.y));
  let q = abs(point - half_size) - (half_size - vec2f(radius));
  let outside = max(q, vec2f(0.0));
  let exponent = max(u.exponent, 2.0);
  let norm = pow(pow(outside.x, exponent) + pow(outside.y, exponent), 1.0 / exponent);
  return norm + min(max(q.x, q.y), 0.0) - radius;
}

@fragment
fn fs_main(@location(0) v_uv: vec2f) -> @location(0) vec4f {
  let source = textureSampleLevel(u_source, u_sampler, v_uv, 0.0);
  let distance = superellipse_distance(v_uv * u.size);
  let coverage = 1.0 - smoothstep(-1.0, 1.0, distance);
  return vec4f(source.rgb, source.a * coverage);
}
