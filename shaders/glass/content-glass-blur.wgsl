// Separable Gaussian blur for the content-glass backdrop.
//
// The measured macOS 27 backdrop blur is much closer to a Gaussian
// (best-fit sigma ~18px at 2x) than to the single mip-level box sample the
// earlier material used: fitting the body against a Gaussian probe drops the
// interior error from ~14/255 to ~5/255.
//
// Two passes (horizontal, then vertical) approximate the 2D Gaussian. The
// glass shader then samples the result at LOD 0.

struct Blur {
  // Texel step direction: (1, 0) horizontal, (0, 1) vertical.
  dir: vec2f,
  // Gaussian sigma in pixels.
  sigma: f32,
  // Sample radius in pixels (typically 2 * sigma).
  radius: f32,
};

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> u: Blur;

struct VsOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  let xy = p[vi];
  var out: VsOut;
  out.pos = vec4f(xy, 0.0, 1.0);
  out.uv = vec2f((xy.x + 1.0) * 0.5, 1.0 - (xy.y + 1.0) * 0.5);
  return out;
}

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  let texel = u.dir / vec2f(textureDimensions(src, 0));
  let sigma = max(u.sigma, 0.5);
  let r = i32(ceil(u.radius));
  let inv2s2 = 1.0 / (2.0 * sigma * sigma);

  var sum = textureSampleLevel(src, samp, uv, 0.0);
  var wsum = 1.0;
  for (var i = 1; i <= r; i += 1) {
    let fi = f32(i);
    let w = exp(-fi * fi * inv2s2);
    sum += textureSampleLevel(src, samp, uv + texel * fi, 0.0) * w;
    sum += textureSampleLevel(src, samp, uv - texel * fi, 0.0) * w;
    wsum += 2.0 * w;
  }
  return sum / wsum;
}
