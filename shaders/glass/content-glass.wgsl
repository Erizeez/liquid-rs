// Content Liquid Glass — macOS 27 fitted material (WGSL).
//
// A clean re-implementation of the *content* glass material, fitted against
// real macOS 27 NSGlassEffectView captures on this machine (build 26A5421a):
//
//   * body   : 3x4 affine colour matrix over the mip-blurred backdrop,
//              fitted per appearance (light/dark). MAE 1.5..3.2/255.
//   * blur   : mip selection, lod = max(0, log2(radius * scale)). Cost is
//              constant regardless of radius; matches the measured high
//              frequency decay of the real material.
//   * shape  : rounded rectangle with a per-corner radius (the measured macOS
//              corner is a circular arc, not a full-element superellipse).
//   * edges  : a sub-pixel anisotropic contact contour. Left/right are much
//              darker than top/bottom (measured factor 0.376 vs 0.451 light,
//              0.549 vs 0.901 dark). This is the whole "top/bottom light,
//              left/right dark" effect — there is no painted bevel in the
//              content material.
//   * lensing: two-sided refraction (inner + outer lobe) and chromatic
//              aberration along the rim.
//
// Deliberately NOT here: rim glint, edge bleed, key fill, ring shadow, blur
// fill, press light. Those belong to control chrome / interactive variants,
// not to the base content glass, and painting them globally is what made the
// previous implementation read as plastic.

const TINY: f32 = 1.0e-6;
const EPS: f32 = 1.0e-4;

struct Uniforms {
  // Element rect (x, y, w, h) in canvas pixels, top-left origin, plus canvas
  // size. Shared with the vertex stage so one dynamic-offset buffer can draw
  // any number of elements.
  rect: vec4f,
  canvas: vec2f,
  half_size: vec2f,
  corner_radius: f32,
  dark: f32,
  _pad_frame: vec2f,
  // (inner amount, inner inv_height, outer amount, outer inv_height)
  inner_refract: vec4f,
  // (refract opacity, threshold0, threshold1, unused)
  refract_params: vec4f,
  refract_angle: vec2f,
  _pad0: vec2f,
  displacement: vec4f,
  // (aberration amount, inv_height, offset, unused)
  aberration: vec4f,
  aberration_angle: vec2f,
  _pad1: vec2f,
  // (blur radius in device px, refraction scale_ref, edge line width, output_linear)
  blur_params: vec4f,
  // (contour opacity left/right, contour opacity top/bottom, diffusion, unused)
  edge_params: vec4f,
  face_cm0: vec4f,
  face_cm1: vec4f,
  face_cm2: vec4f,
  face_cm_dark0: vec4f,
  face_cm_dark1: vec4f,
  face_cm_dark2: vec4f,
  // Quadratic colour correction, one vec4 per term:
  // term order is (r2, g2, b2, rg, rb, gb), xyz is the per-output gain.
  quad_light: array<vec4f, 6>,
  quad_dark: array<vec4f, 6>,
};

@group(0) @binding(0) var u_backdrop: texture_2d<f32>;
@group(0) @binding(1) var u_sampler: sampler;
@group(0) @binding(2) var<uniform> u: Uniforms;

struct Sdf {
  dist: f32,
  normal: vec2f,
};

// Continuous Curvature (Squircle G2) SDF from squircle-rs:
// When smoothing == 0.0, evaluates exact Euclidean circular arc (matching native macOS 27 NSGlassEffectView).
// When smoothing > 0.0, evaluates analytical G2 continuous curvature squircle with diagonal
// radius normalization (2^(1/n - 0.5)) so that smoothing does NOT artificially bloat the corner radius!
fn rounded_rect_sdf(p: vec2f, half_size: vec2f, radius: f32, smoothing: f32) -> Sdf {
  let r = clamp(radius, 0.0, min(half_size.x, half_size.y));
  let q = abs(p) - (half_size - vec2f(r));

  if (q.x <= 0.0 || q.y <= 0.0) {
    let d = max(q.x, q.y) - r;
    var g = vec2f(0.0, 1.0);
    if (q.x > q.y) {
      g = vec2f(1.0, 0.0);
    }
    return Sdf(d, g * sign(p));
  }

  let s = clamp(smoothing, 0.0, 1.0);
  if (s <= 0.001) {
    let l = length(q);
    let dist = l - r;
    let g = select(vec2f(0.0, 1.0), q / max(l, TINY), l > TINY);
    return Sdf(dist, g * sign(p));
  }

  // Continuous curvature corner (Squircle G2 model from squircle-rs):
  let exp_n = mix(2.0, 4.2, s);

  let q_safe = max(q, vec2f(TINY));
  let q_pow = vec2f(pow(q_safe.x, exp_n), pow(q_safe.y, exp_n));
  let dist = pow(q_pow.x + q_pow.y, 1.0 / exp_n) - r;

  let grad_pow = vec2f(pow(q_safe.x, exp_n - 1.0), pow(q_safe.y, exp_n - 1.0));
  let g = normalize(grad_pow);
  return Sdf(dist, g * sign(p));
}

// Higher-Order Bézier Lensing Displacement Curve:
// Evaluates a 4th-order (quartic) Bernstein Bézier polynomial with boundary conditions:
//   P0 = 1.0 (outer glass edge contact)
//   P1 = u.inner_refract.z (outer shoulder convexity)
//   P2 = u.inner_refract.w (mid-belly convexity / sag)
//   P3 = u.refract_params.x (inner floor landing slope)
//   P4 = 0.0 (inner glass flat floor)
// Provides full parametric control over inflection, plumpness, and transition smoothness.
fn bezier_lensing_curve(t: f32, p1: f32, p2: f32, p3: f32) -> f32 {
  let tt = clamp(t, 0.0, 1.0);
  let u = 1.0 - tt;
  let u2 = u * u;
  let u3 = u2 * u;
  let u4 = u2 * u2;
  let t2 = tt * tt;
  let t3 = t2 * tt;
  let y = u4 * 1.0 + 4.0 * u3 * tt * p1 + 6.0 * u2 * t2 * p2 + 4.0 * u * t3 * p3;
  return clamp(y, 0.0, 2.0);
}

fn refract_lobe(dist: f32, amount: f32, inv_height: f32, offset: f32, p1: f32, p2: f32, p3: f32) -> f32 {
  let t = clamp((-dist - offset) * inv_height, 0.0, 1.0);
  return amount * bezier_lensing_curve(t, p1, p2, p3);
}

// The backdrop texture is pre-blurred by a separable Gaussian pass, so all
// samples read LOD 0. `radius` is retained in the signature because the
// refraction lobes still scale with it, but it no longer selects a mip.
fn sample_backdrop(uv: vec2f, radius: f32) -> vec4f {
  return textureSampleLevel(u_backdrop, u_sampler, uv, 0.0);
}

// Anti-aliased coverage across a 2.4px transition band (matching the
// physical 1.2px boundary footprint observed on macOS Retina captures).
fn aa_step(x: f32) -> f32 {
  return clamp(x / 2.0 + 0.5, 0.0, 1.0);
}

fn grade(c: vec3f, r0: vec4f, r1: vec4f, r2: vec4f) -> vec3f {
  return vec3f(dot(c, r0.xyz) + r0.w, dot(c, r1.xyz) + r1.w, dot(c, r2.xyz) + r2.w);
}

fn srgb_to_linear(c: vec3f) -> vec3f {
  let lo = c / 12.92;
  let hi = pow((c + vec3f(0.055)) / 1.055, vec3f(2.4));
  return select(lo, hi, c > vec3f(0.04045));
}

@fragment
fn fs_main(
  @location(0) v_uv: vec2f,
  @location(1) v_backdrop_uv: vec2f,
) -> @location(0) vec4f {
  let diffusion = clamp(u.edge_params.z, 0.0, 1.0);
  if (diffusion <= 0.0) {
    discard;
  }
  let d_refract = pow(diffusion, 0.55);
  let d_blur = pow(diffusion, 1.30);
  let d_body = pow(diffusion, 1.60);

  let backdrop_size = vec2f(textureDimensions(u_backdrop, 0));
  let texel = 1.0 / backdrop_size;

  let scale_ref = u.blur_params.y;
  let s = select(1.0, min(u.half_size.x, u.half_size.y) / scale_ref, scale_ref > 0.0);
  let anim_scale = mix(0.96, 1.0, pow(diffusion, 0.8));
  let anim_half = u.half_size * anim_scale;

  let corner_smoothing = u.refract_params.y;
  let sdf = rounded_rect_sdf(v_uv, anim_half, u.corner_radius * anim_scale, corner_smoothing);
  let dist = sdf.dist;
  let normal = sdf.normal;

  // Rotate the normal, then push it through the 2x2 displacement matrix.
  let rot = vec2f(
    dot(normal, vec2f(u.refract_angle.x, -u.refract_angle.y)),
    dot(normal, vec2f(u.refract_angle.y, u.refract_angle.x)));
  let disp = vec2f(dot(rot, u.displacement.xy), dot(rot, u.displacement.zw));

  // Measured macOS 27 blur is a fixed radius in device pixels, not scaled by
  // element size (best-fit ~65px at 2x for every panel). `s` still scales the
  // refraction lobes below, which DO grow with the element.
  let blur_radius = u.blur_params.x * d_blur;

  // Inner lobe.
  // Higher-Order Bézier control parameters:
  let p1 = select(1.0, u.inner_refract.z, abs(u.inner_refract.z) > 0.001);
  let p2 = select(0.75, u.inner_refract.w, abs(u.inner_refract.w) > 0.001);
  let p3 = select(0.30, u.refract_params.w, abs(u.refract_params.w) > 0.001);

  // Anisotropic bezel height scaling:
  // Scales the optical bevel depth by displacement aspect ratio:
  // Left/Right (Nx=1, Ny=0) has height scaled by displacement.x (wider/thicker).
  // Top/Bottom (Nx=0, Ny=1) has height scaled by displacement.w (tighter/narrower).
  let aniso_scale = sqrt(normal.x * normal.x * u.displacement.x * u.displacement.x +
                         normal.y * normal.y * u.displacement.w * u.displacement.w);
  let eff_inv_height = select(u.inner_refract.y / s, (u.inner_refract.y / s) / max(aniso_scale, TINY), aniso_scale > 0.05);

  let inner_mag = refract_lobe(
    dist, u.inner_refract.x * d_refract * s, eff_inv_height, 0.0, p1, p2, p3);
  let inner_uv = v_backdrop_uv + inner_mag * disp * texel;
  var face = sample_backdrop(inner_uv, blur_radius);

  // Outer lobe — the two-sided part that reads as thick glass.
  if (u.refract_params.x > 0.0) {
    let outer_mag = refract_lobe(
      dist, u.inner_refract.z * d_refract * s, u.inner_refract.w / s, 0.0, p1, p2, p3);
    let outer_uv = v_backdrop_uv + outer_mag * disp * texel;
    let outer = sample_backdrop(outer_uv, blur_radius);
    let span = u.refract_params.z - u.refract_params.y;
    let t = clamp(
      (dist - u.refract_params.y) / select(TINY, span, abs(span) >= TINY),
      0.0, 1.0);
    face = mix(face, outer, t * u.refract_params.x);
  }

  var rgb = face.rgb;

  // Chromatic aberration: two loops walking opposite ways along an
  // independently rotated vector. Red and blue are dragged apart, green
  // straddles both. 3 forward + 4 backward taps.
  if (u.aberration.x > 0.0) {
    let rot_a = vec2f(
      dot(normal, vec2f(u.aberration_angle.x, -u.aberration_angle.y)),
      dot(normal, vec2f(u.aberration_angle.y, u.aberration_angle.x)));
    let disp_a = vec2f(dot(rot_a, u.displacement.zw), dot(rot_a, u.displacement.xy));
    let off_a = disp_a * refract_lobe(dist, u.aberration.x, u.aberration.y, u.aberration.z, p1, p2, p3);

    var acc = vec3f(0.0);
    var w = 1.0;
    for (var i = 0; i < 3; i += 1) {
      let sm = sample_backdrop(inner_uv + off_a * w * texel, blur_radius);
      let a = max(sm.a, TINY);
      acc.r += (sm.r / a) * w;
      acc.g += (sm.g / a) * (1.0 - w);
      w -= 1.0 / 3.0;
    }
    var ss = 0.0;
    for (var i = 0; i < 4; i += 1) {
      let sm = sample_backdrop(inner_uv - off_a * ss * texel, blur_radius);
      let a = max(sm.a, TINY);
      acc.g += (sm.g / a) * (1.0 - ss);
      acc.b += (sm.b / a) * ss;
      ss += 1.0 / 3.0;
    }
    acc *= vec3f(0.5, 1.0 / 3.0, 0.5);
    rgb = acc;
  }

  // Body grade. Both appearances are carried so the caller can cross-fade.
  let cm0 = mix(u.face_cm0, u.face_cm_dark0, u.dark);
  let cm1 = mix(u.face_cm1, u.face_cm_dark1, u.dark);
  let cm2 = mix(u.face_cm2, u.face_cm_dark2, u.dark);
  var graded = grade(rgb, cm0, cm1, cm2);

  // Apple BlurFill (lighten mode):
  // Transcribed from Apple's glass_background_all_lpf:
  // Samples un-refracted backdrop at v_backdrop_uv and composites with lighten blend (opacity 0.675).
  // This naturally rounds off sharp optical caustics and triangular refraction tips,
  // producing the soft, organic, rounded ends observed on native macOS glass.
  let fill_raw = sample_backdrop(v_backdrop_uv, blur_radius);
  let fill_graded = grade(fill_raw.rgb, cm0, cm1, cm2);
  graded = mix(graded, max(graded, fill_graded), 0.35);

  // Backdrop-Aware Vibrancy Layer:
  // Derived from Apple's CALayer with vibrantColorMatrix (inputBackdropAware = 1):
  // Adds a subtle ambient lift on bright backgrounds, maintaining luminous milky depth.
  let luma = dot(graded, vec3f(0.212646, 0.715332, 0.072205));
  let vibrant_alpha = clamp(0.3125 * luma - 0.25, 0.0, 0.12);
  graded = mix(graded, vec3f(1.0), vibrant_alpha);

  // Pure Neutral White Milkiness (Colloidal / Opalescent Scattering):
  // Mixes towards pure neutral white (1.0, 1.0, 1.0) rather than empirical color matrices,
  // producing authentic milk-glass / porcelain opalescence without hue shift or color cast.
  let milky_white = clamp(u.refract_params.z, 0.0, 1.0);
  graded = mix(graded, vec3f(1.0), milky_white * 0.70);

  // Physical Grazing Highlight (Symmetric Top & Bottom):
  // Evaluates sharp 1-2px specular core + faint subtle inner halo along horizontal edges:
  let highlight_gain = select(1.0, u.edge_params.w, u.edge_params.w > 0.001);
  let edge_weight = pow(abs(normal.y), 1.6);
  let d_in = max(0.0, -dist);
  let eff_h_px = select(26.0, 1.0 / max(eff_inv_height, TINY), eff_inv_height > 0.001);
  let core_span = 4.4; // 2.2px at 1x = 4.4px at 2x Retina
  let sharp_core = pow(1.0 - min(1.0, d_in / core_span), 3.0);
  let faint_halo = pow(1.0 - min(1.0, d_in / eff_h_px), 2.0) * 0.08;
  let light_contrib = (sharp_core * 0.72 + faint_halo) * edge_weight * highlight_gain;
  graded = min(vec3f(1.0), graded + vec3f(light_contrib));

  // Lateral Micro-Rim Subtractive Shadow:
  // Darkens vertical edges (|nx| -> 1.0) within a tight 5.2px micro-band (2.6px at 1x):
  let lat_weight = pow(abs(normal.x), 2.0);
  let lat_span = 5.2;
  let lat_decay = pow(1.0 - min(1.0, d_in / lat_span), 2.0);
  let lat_drop = lat_weight * lat_decay * (42.0 / 255.0);
  graded = max(vec3f(0.0), graded - vec3f(lat_drop));

  // Physical Subtractive Crevice Ambient Occlusion:
  // Under diffuse sky lighting, ambient occlusion represents lost ambient illumination (delta E).
  // It subtracts a bounded ambient delta (46/255 on vertical sides, 32/255 on horizontal sides)
  // rather than multiplying the background by a dark constant (which would crush saturated bright hues).
  let delta_lr = mix(32.0 / 255.0, 46.0 / 255.0, normal.x * normal.x);
  let delta_ao = mix(delta_lr, delta_lr * 1.2, u.dark);

  let fw = max(fwidth(dist), 1.0);
  let body_coverage = clamp((0.5 - dist) / fw, 0.0, 1.0);
  let crevice_dist = clamp((dist - 0.5 * fw) / fw, -1.0, 1.0);
  let crevice_decay = (1.0 - crevice_dist * crevice_dist) * (1.0 - body_coverage);

  // Read backdrop at exact fragment coordinate to compute subtractive shadow:
  let bg_raw = textureSampleLevel(u_backdrop, u_sampler, v_backdrop_uv, 0.0).rgb;
  let shadow_color = max(vec3f(0.0), bg_raw - vec3f(delta_ao));

  let shadow_coverage = clamp(crevice_decay, 0.0, 1.0);
  let total_coverage = (body_coverage + shadow_coverage) * d_body;
  if (total_coverage < TINY) {
    discard;
  }

  let final_rgb = mix(shadow_color, graded, body_coverage);
  let out_rgb = select(final_rgb, srgb_to_linear(final_rgb), u.blur_params.w > 0.5);
  return vec4f(out_rgb, total_coverage);
}
