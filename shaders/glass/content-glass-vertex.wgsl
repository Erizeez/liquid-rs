// Vertex stage for the fitted content-glass material.
//
// Draws a quad covering the element's rect plus a small pad in canvas pixels
// and produces:
//   v_uv          element-local pixels, origin at the centre, Y up
//   v_backdrop_uv normalised canvas coordinates of the same point
//
// The pad is required because the macOS contact contour is a ~1px dark line
// drawn just OUTSIDE the silhouette; without it the line would fall outside
// the rasterized quad and never appear.
//
// The element rect, canvas size and half size live in the same uniform block
// as the fragment material (group 0 / binding 2), so one dynamic-offset buffer
// can draw any number of elements.

// Shared with the fragment stage: rect (x,y,w,h) canvas px + canvas size +
// element half size.
struct Uniforms {
  rect: vec4f,
  canvas: vec2f,
  half_size: vec2f,
  corner_radius: f32,
  dark: f32,
  _pad_frame: vec2f,
  inner_refract: vec4f,
  refract_params: vec4f,
  refract_angle: vec2f,
  _pad0: vec2f,
  displacement: vec4f,
  aberration: vec4f,
  aberration_angle: vec2f,
  _pad1: vec2f,
  blur_params: vec4f,
  edge_params: vec4f,
  face_cm0: vec4f,
  face_cm1: vec4f,
  face_cm2: vec4f,
  face_cm_dark0: vec4f,
  face_cm_dark1: vec4f,
  face_cm_dark2: vec4f,
  quad_light: array<vec4f, 6>,
  quad_dark: array<vec4f, 6>,
};

@group(0) @binding(2) var<uniform> frame: Uniforms;

struct VsOut {
  @builtin(position) pos: vec4f,
  @location(0) v_uv: vec2f,
  @location(1) v_backdrop_uv: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
  var q = array<vec2f, 4>(
    vec2f(-1.0, -1.0),
    vec2f(1.0, -1.0),
    vec2f(-1.0, 1.0),
    vec2f(1.0, 1.0));
  let p = q[vi];

  // Expand the quad so the outer contact line has room to rasterize.
  let pad = 3.0;
  let x0 = frame.rect.x - pad;
  let y0 = frame.rect.y - pad;
  let w = frame.rect.z + pad * 2.0;
  let h = frame.rect.w + pad * 2.0;

  let u = p.x * 0.5 + 0.5;
  let v = 0.5 - p.y * 0.5;
  let px = x0 + u * w;
  let py = y0 + v * h;

  // Element-local coordinates matching canvas pixel orientation (X right, Y down).
  // This ensures normal vectors point in the exact same direction as backdrop UVs
  // without any Y-flip inconsistencies.
  let cx = frame.rect.x + frame.rect.z * 0.5;
  let cy = frame.rect.y + frame.rect.w * 0.5;

  var out: VsOut;
  out.pos = vec4f(
    px / frame.canvas.x * 2.0 - 1.0,
    1.0 - py / frame.canvas.y * 2.0,
    0.0,
    1.0);
  out.v_uv = vec2f(px - cx, py - cy);
  out.v_backdrop_uv = vec2f(px / frame.canvas.x, py / frame.canvas.y);
  return out;
}
