// Fullscreen quad vertex shader for WebGPU.
// Ported from liquid-glass-studio/src/shaders-wgsl/vertex.wgsl.

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vs_main(@location(0) a_position: vec2f) -> VertexOutput {
  var out: VertexOutput;
  let uv = (a_position + 1.0) * 0.5;
  out.uv = vec2f(uv.x, 1.0 - uv.y);
  out.position = vec4f(a_position, 0.0, 1.0);
  return out;
}
