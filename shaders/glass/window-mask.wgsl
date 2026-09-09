// Final output mask for a borderless transparent window.
//
// This is deliberately a separate present pass. The scene remains rectangular
// internally, while only the pixels handed to the platform surface receive the
// window's continuous-corner alpha mask evaluated through squircle-rs.

@group(0) @binding(0) var u_source: texture_2d<f32>;
@group(0) @binding(1) var u_sampler: sampler;

struct WindowMaskUniform {
  size: vec2f,
  radius: f32,
  exponent: f32,
};

@group(0) @binding(2) var<uniform> u: WindowMaskUniform;

@fragment
fn fs_main(@location(0) v_uv: vec2f) -> @location(0) vec4f {
  let source = textureSampleLevel(u_source, u_sampler, v_uv, 0.0);
  let half_size = u.size * 0.5;
  let p = v_uv * u.size - half_size;
  // Use squircle-rs analytical continuous curvature SDF with Apple standard smoothing (0.6)
  let distance = sd_squircle(p, half_size, u.radius, 0.6);
  let coverage = 1.0 - smoothstep(-1.0, 1.0, distance);
  return vec4f(source.rgb, source.a * coverage);
}
