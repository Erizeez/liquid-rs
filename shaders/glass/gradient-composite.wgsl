// Mixes the locally blurred image with the original image. The fallback keeps
// transparent or unavailable samples from turning the transition black.

@group(0) @binding(0) var u_blurred: texture_2d<f32>;
@group(0) @binding(1) var u_original: texture_2d<f32>;
@group(0) @binding(2) var u_sampler: sampler;

struct Uniforms {
  fallback: vec4f,
  output_size: vec2f,
  _pad: vec2f,
  // [start_y, end_y, unused, unused] in physical top-origin pixels.
  gradient: vec4f,
};

@group(0) @binding(3) var<uniform> u: Uniforms;

@fragment
fn fs_main(@location(0) v_uv: vec2f) -> @location(0) vec4f {
  let blurred = textureSampleLevel(u_blurred, u_sampler, v_uv, 0.0);
  let original = textureSampleLevel(u_original, u_sampler, v_uv, 0.0);
  // Invalid transparent samples fall back to the known sidebar medium instead
  // of importing black RGB from the transparent-window compositor.
  // The fallback belongs only to the blurred sample. The clear endpoint must
  // retain the original alpha; otherwise fallback.a leaves a translucent
  // film at the bottom of the gradient even when the blur amount is zero.
  let sampledBlur = select(u.fallback.rgb, blurred.rgb, blurred.a > 0.001);
  let sampledBlurAlpha = select(u.fallback.a, blurred.a, blurred.a > 0.001);
  let pixelY = v_uv.y * u.output_size.y;
  let gradientHeight = max(u.gradient.y - u.gradient.x, 0.000001);
  let linearAmount = select(
    1.0,
    clamp((u.gradient.y - pixelY) / gradientHeight, 0.0, 1.0),
    pixelY > u.gradient.x,
  );
  let amount = linearAmount * linearAmount * (3.0 - 2.0 * linearAmount);
  let blurMix = mix(u.gradient.z, 1.0, amount);
  // Replace the sharp source with the blur according to the vertical amount.
  // This prevents a translucent overlay from leaving a sharp copy of the
  // scrolling text visible underneath the blurred result.
  // Preserve the translucent sidebar medium. Only scrolling application
  // content is blurred here; WindowServer's live backdrop stays underneath.
  return vec4f(
    mix(original.rgb, sampledBlur, blurMix),
    mix(original.a, sampledBlurAlpha, blurMix),
  );
}
