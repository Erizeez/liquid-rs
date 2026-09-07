// Fullscreen copy used by the hierarchical glass compositor.

@group(0) @binding(0) var u_source: texture_2d<f32>;
@group(0) @binding(1) var u_sampler: sampler;

@fragment
fn fs_main(@location(0) v_uv: vec2f) -> @location(0) vec4f {
  return textureSampleLevel(u_source, u_sampler, v_uv, 0.0);
}
