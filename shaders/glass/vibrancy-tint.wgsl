// Uniform rectangular vibrancy surface. The blur itself is produced by the
// Dual Kawase pyramid from vibrancy-rs; this pass only applies the medium tint.

struct Uniforms {
  tint: vec4f,
};

@group(0) @binding(0) var u_blurred: texture_2d<f32>;
@group(0) @binding(1) var u_sampler: sampler;
@group(0) @binding(2) var<uniform> u: Uniforms;

@fragment
fn fs_main(@location(0) v_uv: vec2f) -> @location(0) vec4f {
  let blurred = textureSampleLevel(u_blurred, u_sampler, v_uv, 0.0);
  return vec4f(mix(blurred.rgb, u.tint.rgb, clamp(u.tint.a, 0.0, 1.0)), 1.0);
}
