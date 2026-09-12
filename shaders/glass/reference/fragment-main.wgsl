// Main glass composition shader ported from liquid-glass-studio.

const PI: f32 = 3.14159265359;
const N_R: f32 = 0.98;
const N_G: f32 = 1.0;
const N_B: f32 = 1.02;
// The SDF is lifted into a shallow, rounded optical profile. A rough
// dielectric interface evaluates both reflected and transmitted energy.
const SURFACE_BEVEL_MIN: f32 = 3.0;
const SURFACE_BEVEL_MAX: f32 = 11.0;
const SURFACE_MIN_ROUGHNESS: f32 = 0.09;
const SURFACE_MAX_ROUGHNESS: f32 = 0.20;
const EXTERNAL_TAIL_STRENGTH: f32 = 0.55;
const EXTERNAL_TAIL_ONSET: f32 = 3.0;
const TRAFFIC_LIGHT_CLEAR_COAT_WIDTH: f32 = 1.20;
// Deliberately wider than the clear coat: this is the dark material body
// response, not the outer antialiased highlight. It scales with the control
// diameter through `trafficLightSizeScale()` below.
const TRAFFIC_LIGHT_COATING_BAND: f32 = 4.80;
const TRAFFIC_LIGHT_BASE_FRESNEL_RANGE: f32 = 24.0;
const TRAFFIC_LIGHT_MIN_THICKNESS: f32 = 0.58;
const TRAFFIC_LIGHT_VERTICAL_LIGHT_FACTOR: f32 = 0.18;
const TRAFFIC_LIGHT_EDGE_DARKNESS: f32 = 1.08;
const TRAFFIC_LIGHT_EDGE_SHARPNESS: f32 = 8.00;
const TRAFFIC_LIGHT_VERTICAL_EDGE_WIDTH: f32 = 0.42;
const TRAFFIC_LIGHT_VERTICAL_EDGE_SHARPNESS: f32 = 8.00;
const TRAFFIC_LIGHT_PHYSICAL_TINT_COVERAGE: f32 = 0.95;
// Press-state lift measured from the native control: normal red is about
// sRGB (242, 94, 83), while the pressed state is about (255, 119, 104).
// These deltas are linear-light gains; R reaches the SDR ceiling while G/B
// rise only enough to reproduce the native pressed-state softness.
const TRAFFIC_LIGHT_RED_PRESS_LIFT: vec3f = vec3f(0.11, 0.072, 0.052);

struct Uniforms {
  u_resolution: vec2f,
  u_dpr: f32,
  // Device pixels per logical point. Not a shape scale: node geometry is
  // already physical. Used where a material response is authored in points.
  u_renderScale: f32,
  u_mouse: vec2f,
  u_mouseSpring: vec2f,
  u_shapeWidth: f32,
  u_shapeHeight: f32,
  u_shapeRadius: f32,
  u_shapeRoundness: f32,
  u_capsuleBezierX: vec4f,
  u_capsuleBezierY: vec4f,
  u_mergeRate: f32,
  u_press: f32,
  u_shadowExpand: f32,
  u_shadowFactor: f32,
  u_shadowPosition: vec2f,
  u_bgTextureRatio: f32,
  u_bgType: i32,
  u_bgTextureReady: i32,
  u_showShape1: i32,
  u_blurRadius: i32,
  u_featureFlags: i32,
  u_tint: vec4f,
  u_refThickness: f32,
  u_refFactor: f32,
  u_refDispersion: f32,
  u_refFresnelRange: f32,
  u_refFresnelHardness: f32,
  u_refFresnelFactor: f32,
  u_glareRange: f32,
  u_glareHardness: f32,
  u_glareConvergence: f32,
  u_glareOppositeFactor: f32,
  u_glareFactor: f32,
  u_refStrength: f32,
  u_opacity: f32,
  u_interaction: f32,
  u_environmentLuminance: f32,
  u_environmentContrast: f32,
  u_adaptive: vec4f,
  u_fusedBounds: array<vec4f, 4>,
  u_fusedGeometry: array<vec4f, 4>,
  u_interactionState: vec4f,
  u_trafficLight: vec4f,
  u_trafficLightLight: vec4f,
  u_trafficLightEdge: vec4f,
  // x: hover gain, y: press gain, z: tint-hued press lift
  // (`GlassMaterial::interaction`).
  u_interactionResponse: vec4f,
  // x: uniform incident field, y: lower-hemisphere thin light gain
  // (`GlassMaterial::core_light`).
  u_coreLight: vec4f,
  // x: core lift, y: axial glow, z: horizontal roll-off power
  u_coreLightGradient: vec4f,
  // x: lateral power, y: vertical floor, z: grazing power
  // (`GlassMaterial::rim_profile`).
  u_rimProfile: vec4f,
  // Reference bead: x/y/z droplet control points, w centre glow.
  u_beadA: vec4f,
  // x saturation lift, y highlight, z dark rim, w core span factor.
  u_beadB: vec4f,
  // x rim span factor, y caustic light, z caustic dark, w mode (1 = dark).
  u_beadC: vec4f,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var u_blurredBg: texture_2d<f32>;
@group(0) @binding(2) var u_bg: texture_2d<f32>;
@group(0) @binding(3) var u_sampler: sampler;

const FEATURE_EDGE_BLUR: i32 = 1;
const FEATURE_REDUCED_TRANSPARENCY: i32 = 2;
const FEATURE_INCREASED_CONTRAST: i32 = 4;
const FEATURE_REDUCED_MOTION: i32 = 8;
const FEATURE_CLEAR_VARIANT: i32 = 16;
const FEATURE_TRAFFIC_LIGHT: i32 = 32;
const FEATURE_TRAFFIC_LIGHT_PHYSICAL: i32 = 64;
const FEATURE_TRAFFIC_LIGHT_REFERENCE: i32 = 128;
const FEATURE_TRAFFIC_LIGHT_BEAD: i32 = 256;
// Texture sampling from the renderer's sRGB target returns linear values.
// These are the linear-light equivalents of the light macOS window substrate
// used to calibrate the traffic-light material (0.95, 0.95, 0.965 sRGB).
const TRAFFIC_LIGHT_REFERENCE_BACKDROP: vec3f = vec3f(0.8900, 0.8900, 0.9180);

fn featureEnabled(flag: i32) -> bool {
  return (u.u_featureFlags & flag) != 0;
}

fn isTrafficLightBead() -> bool {
  return featureEnabled(FEATURE_TRAFFIC_LIGHT_BEAD);
}

fn usesTrafficLightReferenceBackdrop() -> bool {
  return featureEnabled(FEATURE_TRAFFIC_LIGHT_PHYSICAL)
    && featureEnabled(FEATURE_TRAFFIC_LIGHT_REFERENCE);
}

fn trafficLightReferenceBackdrop() -> vec3f {
  return TRAFFIC_LIGHT_REFERENCE_BACKDROP;
}

fn sampleActualBackdrop(v_uv: vec2f) -> vec4f {
  return textureSampleLevel(u_bg, u_sampler, v_uv, 0.0);
}

fn sdCircle(p: vec2f, r: f32) -> f32 {
  return length(p) - r;
}

fn continuousCornerSDF(p_in: vec2f, r: f32, n: f32) -> f32 {
  let p = abs(p_in);
  let exponent = max(n, 2.0);
  let v = pow(pow(p.x, exponent) + pow(p.y, exponent), 1.0 / exponent);
  return v - r;
}

fn roundedRectSDF(p_in: vec2f, center: vec2f, width: f32, height: f32, cornerRadius: f32, n: f32) -> f32 {
  let p = p_in - center;
  let cr = cornerRadius * u.u_dpr;
  let d = abs(p) - vec2f(width * u.u_dpr, height * u.u_dpr) * 0.5;
  var dist: f32;
  if (d.x > -cr && d.y > -cr) {
    let cornerCenter = sign(p) * (vec2f(width * u.u_dpr, height * u.u_dpr) * 0.5 - vec2f(cr));
    let cornerP = p - cornerCenter;
    dist = continuousCornerSDF(cornerP, cr, n);
  } else {
    dist = min(max(d.x, d.y), 0.0) + length(max(d, vec2f(0.0)));
  }
  return dist;
}

fn smin(a: f32, b: f32, k: f32) -> f32 {
  let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
  return mix(b, a, h) - k * h * (1.0 - h);
}

fn cubicCoordinate(points: vec4f, t: f32) -> f32 {
  let oneMinusT = 1.0 - t;
  return points.x * oneMinusT * oneMinusT * oneMinusT
    + 3.0 * points.y * oneMinusT * oneMinusT * t
    + 3.0 * points.z * oneMinusT * t * t
    + points.w * t * t * t;
}

fn cubicDerivative(points: vec4f, t: f32) -> f32 {
  let oneMinusT = 1.0 - t;
  return 3.0 * (points.y - points.x) * oneMinusT * oneMinusT
    + 6.0 * (points.z - points.y) * oneMinusT * t
    + 3.0 * (points.w - points.z) * t * t;
}

fn capsuleBezierParameter(longitudinal: f32) -> f32 {
  let points = u.u_capsuleBezierX;
  var t = clamp((longitudinal - points.x) / max(points.w - points.x, 0.000001), 0.0, 1.0);
  for (var iteration = 0; iteration < 5; iteration += 1) {
    let derivative = cubicDerivative(points, t);
    if (abs(derivative) > 0.000001) {
      t = clamp(t - (cubicCoordinate(points, t) - longitudinal) / derivative, 0.0, 1.0);
    }
  }
  return t;
}

fn continuousCapsuleSDF(
  p_in: vec2f,
  center: vec2f,
  width: f32,
  height: f32,
) -> f32 {
  let p = abs(p_in - center);
  let halfSize = vec2f(width, height) * u.u_dpr * 0.5;
  let radius = min(halfSize.x, halfSize.y);
  let horizontal = halfSize.x >= halfSize.y;
  let halfLong = select(halfSize.y, halfSize.x, horizontal);
  let longitudinal = select(p.y, p.x, horizontal);
  let transverse = select(p.x, p.y, horizontal);
  let capCenter = max(halfLong - radius, 0.0);

  if (capCenter <= 0.000001) {
    return sdCircle(p, radius);
  }

  let normalizedLongitudinal = (longitudinal - capCenter) / radius;
  if (normalizedLongitudinal <= u.u_capsuleBezierX.x) {
    return transverse - radius;
  }
  if (normalizedLongitudinal >= u.u_capsuleBezierX.w) {
    return length(vec2f(longitudinal - capCenter, transverse)) - radius;
  }

  let t = capsuleBezierParameter(normalizedLongitudinal);
  let curveY = cubicCoordinate(u.u_capsuleBezierY, t);
  let derivativeX = cubicDerivative(u.u_capsuleBezierX, t);
  let derivativeY = cubicDerivative(u.u_capsuleBezierY, t);
  let profile = radius * (1.0 - curveY);
  let slope = -derivativeY / max(derivativeX, 0.000001);
  return (transverse - profile) / sqrt(1.0 + slope * slope);
}

/// The node's own primary silhouette, before shape 1 and any fused layer are
/// merged in.
///
/// The reference bead carves its disc from this rather than from the merged
/// field, so neither a fused secondary shape nor the shape-1 debug circle can
/// widen its coverage.
fn shapeSDF(p2: vec2f, p: vec2f) -> f32 {
  let p2n = p2 + p / u.u_resolution.y;
  if (u.u_shapeRoundness < 0.0) {
    return continuousCapsuleSDF(
      p2n,
      vec2f(0.0),
      u.u_shapeWidth / u.u_resolution.y,
      u.u_shapeHeight / u.u_resolution.y,
    );
  }
  return roundedRectSDF(
    p2n,
    vec2f(0.0),
    u.u_shapeWidth / u.u_resolution.y,
    u.u_shapeHeight / u.u_resolution.y,
    u.u_shapeRadius / u.u_resolution.y,
    u.u_shapeRoundness,
  );
}

fn mainSDF(p1: vec2f, p2: vec2f, p: vec2f) -> f32 {
  let p1n = p1 + p / u.u_resolution.y;
  var d1: f32;
  if (u.u_showShape1 == 1) {
    d1 = sdCircle(p1n, 100.0 * u.u_dpr / u.u_resolution.y);
  } else {
    d1 = 1.0;
  }
  var merged = smin(d1, shapeSDF(p2, p), u.u_mergeRate);
  for (var fusedIndex = 0; fusedIndex < 4; fusedIndex += 1) {
    if (u.u_fusedGeometry[fusedIndex].z > 0.5) {
      let fusedCenter = (vec2f(0.0) - u.u_fusedBounds[fusedIndex].xy) / u.u_resolution.y;
      let fusedP = fusedCenter + p / u.u_resolution.y;
      let fused = roundedRectSDF(
        fusedP,
        vec2f(0.0),
        u.u_fusedBounds[fusedIndex].z / u.u_resolution.y,
        u.u_fusedBounds[fusedIndex].w / u.u_resolution.y,
        u.u_fusedGeometry[fusedIndex].x / u.u_resolution.y,
        u.u_fusedGeometry[fusedIndex].y,
      );
      merged = smin(merged, fused, u.u_mergeRate);
    }
  }
  return merged;
}

fn shadowSDFAt(pixel: vec2f, shadowPosition: vec2f) -> f32 {
  let shadowP1 = (vec2f(0.0) - u.u_resolution * 0.5
    + vec2f(shadowPosition.x * u.u_dpr, shadowPosition.y * u.u_dpr))
    / u.u_resolution.y;
  let shadowP2 = (vec2f(0.0) - u.u_mouseSpring
    + vec2f(shadowPosition.x * u.u_dpr, shadowPosition.y * u.u_dpr))
    / u.u_resolution.y;
  return mainSDF(shadowP1, shadowP2, pixel);
}

fn shadowSDF(pixel: vec2f) -> f32 {
  return shadowSDFAt(pixel, u.u_shadowPosition);
}

fn surfaceSDF(pixel: vec2f) -> f32 {
  // Keep the unshifted surface SDF available to clip the cast shadow to the
  // real outside half-space. The elevation offset must never darken the
  // material interior.
  return shadowSDFAt(pixel, vec2f(0.0));
}

fn shadowStrength(pixel: vec2f) -> f32 {
  // A cast shadow belongs strictly to the outside half-space. Its soft tail
  // follows the elevation offset; the material's inner curved absorption is
  // handled by the glass pass itself. Nothing here can darken the material
  // interior or create a second contour around it.
  let surfaceMerged = surfaceSDF(pixel);
  if (surfaceMerged <= 0.0) {
    return 0.0;
  }
  let pixelsPerSdfUnit = u.u_resolution.y / u.u_dpr;
  let surfaceDistance = surfaceMerged * pixelsPerSdfUnit;
  let castDistance = shadowSDF(pixel) * pixelsPerSdfUnit;
  let penumbra = max(min(u.u_shadowExpand, 12.0), 1.0);
  // A blurred cast silhouette is coverage of the shifted signed-distance
  // field, not an exponential drawn around its outside. The latter creates
  // a second capsule even when its edge is soft.
  let castCoverage = 1.0 - smoothstep(-penumbra, penumbra, castDistance);
  let onset = smoothstep(0.0, EXTERNAL_TAIL_ONSET, surfaceDistance);
  // The SDF difference identifies the side toward which the silhouette was
  // shifted. Suppressing zero/negative deltas removes the upper and lateral
  // halo while preserving the projected shadow beneath the control.
  let offsetMagnitude = max(length(u.u_shadowPosition), 0.5);
  let projectedSide = smoothstep(
    0.0,
    offsetMagnitude,
    surfaceDistance - castDistance,
  );
  let soft = onset * castCoverage * projectedSide * EXTERNAL_TAIL_STRENGTH;
  // The traffic-light body already owns its inner curved absorption. A second
  // contact term here would place another dark contour just outside it and
  // produce the doubled top/bottom edge seen on compact controls. Keep the
  // projected shadow tail, which is the part that belongs on the backdrop.
  return clamp(soft * u.u_shadowFactor, 0.0, 1.0);
}

fn safeAsin(x: f32) -> f32 {
  return asin(clamp(x, -1.0, 1.0));
}

fn getNormal(p1: vec2f, p2: vec2f, p: vec2f) -> vec2f {
  let h = vec2f(1.0, 1.0);
  let grad = vec2f(
    mainSDF(p1, p2, p + vec2f(h.x, 0.0)) - mainSDF(p1, p2, p - vec2f(h.x, 0.0)),
    mainSDF(p1, p2, p + vec2f(0.0, h.y)) - mainSDF(p1, p2, p - vec2f(0.0, h.y)),
  ) / (2.0 * h);
  return safeNormalize(grad);
}

// Preserve the source renderer's optical normal scale for its Snell-law
// displacement. The normalized surface normal above is appropriate for the
// physical interface lighting, but replacing this tuned gradient with a unit
// vector reduced edge displacement from tens of pixels to only a few pixels
// and removed the characteristic liquid-lens response.
fn sourceOpticalNormal(p1: vec2f, p2: vec2f, p: vec2f) -> vec2f {
  let h = vec2f(1.0, 1.0);
  let grad = vec2f(
    mainSDF(p1, p2, p + vec2f(h.x, 0.0)) - mainSDF(p1, p2, p - vec2f(h.x, 0.0)),
    mainSDF(p1, p2, p + vec2f(0.0, h.y)) - mainSDF(p1, p2, p - vec2f(0.0, h.y)),
  ) / (2.0 * h);
  return grad * 1.414213562 * 1000.0;
}

fn safeNormalize(v: vec2f) -> vec2f {
  let len = length(v);
  if (len < 1e-8) {
    return vec2f(0.0);
  }
  return v / len;
}

struct SurfaceProfile {
  normal: vec3f,
  rim: f32,
  highlightRim: f32,
  fresnel: f32,
};

fn trafficLightSizeScale() -> f32 {
  // The material encodes the logical control-size ratio in the Fresnel
  // range. Keep every traffic-light edge width on that same ratio so a
  // 64-point inspection control is not left with a native-size bevel.
  return clamp(
    u.u_refFresnelRange / TRAFFIC_LIGHT_BASE_FRESNEL_RANGE,
    1.0,
    4.0,
  );
}

fn clearCoatWidth() -> f32 {
  let width = clamp(u.u_refFresnelRange * 0.08, 1.25, 2.0);
  let trafficLightScale = trafficLightSizeScale();
  return select(width, TRAFFIC_LIGHT_CLEAR_COAT_WIDTH * trafficLightScale, isTrafficLight());
}

fn surfaceBevelWidth() -> f32 {
  let width = clamp(u.u_refThickness * 0.42, SURFACE_BEVEL_MIN, SURFACE_BEVEL_MAX);
  let trafficLightScale = trafficLightSizeScale();
  let trafficLightWidth = clamp(
    u.u_refThickness * 0.18 * trafficLightScale,
    1.8 * trafficLightScale,
    3.2 * trafficLightScale,
  );
  return select(width, trafficLightWidth, isTrafficLight());
}

fn refractionBodyWidth() -> f32 {
  // Refraction belongs to the glass body, not just its clear-coat rim. Span
  // the short-axis half-width so content under a compact capsule bends
  // continuously from either edge toward the neutral centre line.
  let halfShortAxis = min(u.u_shapeWidth, u.u_shapeHeight) * u.u_dpr * 0.5;
  return max(halfShortAxis, surfaceBevelWidth());
}

fn smootherStep01(value: f32) -> f32 {
  let x = clamp(value, 0.0, 1.0);
  return x * x * x * (x * (x * 6.0 - 15.0) + 10.0);
}

fn surfaceProfile(merged: f32, p1: vec2f, p2: vec2f, pixel: vec2f) -> SurfaceProfile {
  let insideDistance = max(-merged * (u.u_resolution.y / u.u_dpr), 0.0);
  let bevelWidth = surfaceBevelWidth();
  let bevelPosition = clamp(insideDistance / bevelWidth, 0.0, 1.0);
  // A quarter-circle section transitions from a steep outer rim to a flat
  // central pane. Keeping some Z at the rim avoids singular refraction.
  let lateralSlope = cos(bevelPosition * PI * 0.5) * 0.94;
  let planarNormal = getNormal(p1, p2, pixel);
  var surfaceNormal = normalize(vec3f(
    planarNormal * lateralSlope,
    sqrt(max(1.0 - lateralSlope * lateralSlope, 0.001)),
  ));
  // The physical traffic-light experiment uses the exact circular SDF as a
  // shallow sphere. It gets no painted edge or directional-light response;
  // only the interface normal changes from the generic rounded-surface
  // profile to the spherical one.
  if (featureEnabled(FEATURE_TRAFFIC_LIGHT_PHYSICAL)) {
    surfaceNormal = refractionBodyNormal(merged, p1, p2, pixel);
  }
  // The refractive body can remain several pixels thick, but the optically
  // smooth outer coating is much thinner. Using Fresnel range as the coating
  // width keeps this a single physical interface instead of painting a
  // second outline over the broad glass bevel.
  let coatingWidth = clearCoatWidth();
  let coatingPosition = clamp(insideDistance / coatingWidth, 0.0, 1.0);
  let coatingFocus = mix(2.2, 1.35, clamp(u.u_refFresnelHardness, 0.0, 1.0));
  let rim = pow(1.0 - smoothstep(0.0, 1.0, coatingPosition), coatingFocus);
  // The bright strip is convolved over the broader curved bevel. It creates
  // a gradual microfacet shoulder toward the interior without widening the
  // dark lateral clear-coat boundary.
  let highlightWidth = select(bevelWidth * 1.50, bevelWidth * 1.15, isTrafficLight());
  let highlightRim = 1.0 - smootherStep01(insideDistance / highlightWidth);
  let f0 = pow((max(u.u_refFactor, 1.0001) - 1.0) / (max(u.u_refFactor, 1.0001) + 1.0), 2.0);
  let fresnel = f0 + (1.0 - f0) * pow(1.0 - clamp(surfaceNormal.z, 0.0, 1.0), 5.0);
  return SurfaceProfile(surfaceNormal, rim, highlightRim, fresnel);
}

fn refractionBodyNormal(merged: f32, p1: vec2f, p2: vec2f, pixel: vec2f) -> vec3f {
  let insideDistance = max(-merged * (u.u_resolution.y / u.u_dpr), 0.0);
  let bodyPosition = clamp(insideDistance / refractionBodyWidth(), 0.0, 1.0);
  // A quarter-circle body keeps meaningful curvature through the middle of
  // the control. Multiplying the cosine by the outward SDF normal produces a
  // continuous signed displacement through the capsule centre line.
  let lateralSlope = cos(bodyPosition * PI * 0.5) * 0.90;
  let sphericalDepth = sqrt(max(1.0 - lateralSlope * lateralSlope, 0.001));
  // Scale the view-axis radius independently from the visible circular
  // outline. This turns the optical body into an ellipsoid: a value below one
  // flattens it and a value above one gives it more depth.
  let bodyThickness = clamp(u.u_trafficLightLight.y, 0.25, 3.0);
  let planarNormal = getNormal(p1, p2, pixel);
  return normalize(vec3f(
    planarNormal * lateralSlope,
    sphericalDepth / bodyThickness,
  ));
}

fn trafficLightPhysicalLowerLight(merged: f32, p1: vec2f, p2: vec2f, pixel: vec2f) -> f32 {
  if (!featureEnabled(FEATURE_TRAFFIC_LIGHT_PHYSICAL) || merged >= 0.0) {
    return 0.0;
  }

  // The light comes from below and slightly toward the viewer. This is an
  // object-space incident direction: only the lower-facing part of the sphere
  // receives it, while the upper hemisphere remains unlit. Keeping the test
  // in normal space avoids a screen-space ellipse or a moving painted glow.
  let bodyNormal = refractionBodyNormal(merged, p1, p2, pixel);
  let lightAngle = clamp(u.u_trafficLightLight.x, 0.0, 1.0) * (PI * 0.5);
  let incidentDirection = vec3f(0.0, -sin(lightAngle), cos(lightAngle));
  let incidence = max(dot(bodyNormal, incidentDirection), 0.0);
  // Model a finite area-light source rather than blurring the finished pixel.
  // Its angular radius broadens the incident field while preserving the
  // normal-derived spherical placement of the lower illumination.
  let softness = clamp(u.u_trafficLightEdge.y, 0.0, 1.0);
  let lowerThreshold = mix(0.48, 0.20, softness);
  let upperThreshold = mix(0.86, 1.00, softness);
  return smoothstep(lowerThreshold, upperThreshold, incidence);
}

fn trafficLightPhysicalEdgeAbsorption(merged: f32, p1: vec2f, p2: vec2f, pixel: vec2f) -> f32 {
  if (!featureEnabled(FEATURE_TRAFFIC_LIGHT_PHYSICAL) || merged >= 0.0) {
    return 0.0;
  }

  let bodyNormal = refractionBodyNormal(merged, p1, p2, pixel);
  // Keep a closed contour, but make the left/right wall materially broader
  // and darker than the upper/lower arc. The Gaussian is in normal space, so
  // the anisotropy follows the curved body instead of becoming a pair of
  // screen-space side bars.
  // Use two angular Gaussian lobes centred on the left/right horizontal
  // poles. The side angle is their standard deviation in degrees, so the
  // thickness falls continuously instead of having a start/end sector.
  let sideSigma = clamp(u.u_trafficLightEdge.w, 10.0, 80.0) * PI / 180.0;
  let sideDistance = acos(clamp(abs(bodyNormal.x), 0.0, 1.0));
  let gaussianCoordinate = sideDistance / sideSigma;
  let horizontalWeight = exp(-0.5 * gaussianCoordinate * gaussianCoordinate);
  let sideBias = clamp(u.u_trafficLightEdge.z, 0.0, 1.0);
  let grazing = pow(clamp(1.0 - bodyNormal.z, 0.0, 1.0), 0.70);
  let sizeScale = clamp(
    min(u.u_shapeWidth, u.u_shapeHeight) / 64.0,
    0.35,
    1.0,
  );
  let edgeWidth = clamp(
    TRAFFIC_LIGHT_COATING_BAND
      * sizeScale
      * clamp(u.u_trafficLightEdge.x, 0.25, 3.0)
      * mix(
        1.0,
        mix(0.58, 1.55, horizontalWeight),
        sideBias,
      ),
    0.75,
    32.0,
  );
  let insideDistance = max(-merged * (u.u_resolution.y / u.u_dpr), 0.0);
  let edgeSharpness = mix(
    TRAFFIC_LIGHT_VERTICAL_EDGE_SHARPNESS,
    TRAFFIC_LIGHT_EDGE_SHARPNESS,
    mix(0.5, horizontalWeight, sideBias),
  );
  // Preserve the configured physical band width, but concentrate most of
  // the transition toward its inner shoulder. A plain smoothstep spread the
  // tint across the whole band and made the contour read as fog.
  let edgeBand = 1.0 - pow(
    smoothstep(0.0, edgeWidth, insideDistance),
    edgeSharpness,
  );
  let darkness = clamp(u.u_trafficLightLight.w, 0.0, 4.0);
  let baseDarkness = clamp(darkness / 2.0, 0.0, 1.0);
  let extendedDarkness = clamp((darkness - 2.0) / 2.0, 0.0, 1.0);
  let materialDepth =
    mix(0.42, 1.16, baseDarkness) + 0.50 * extendedDarkness;
  let directionalDepth = mix(
    1.0,
    mix(0.65, 1.0, horizontalWeight),
    sideBias,
  );
  return clamp(
    edgeBand
      * grazing
      * materialDepth
      * directionalDepth,
    0.0,
    0.92,
  );
}

const D65_WHITE: vec3f = vec3f(0.95045592705, 1.0, 1.08905775076);
const RGB_TO_XYZ_M_COL0: vec3f = vec3f(0.4124, 0.3576, 0.1805);
const RGB_TO_XYZ_M_COL1: vec3f = vec3f(0.2126, 0.7152, 0.0722);
const RGB_TO_XYZ_M_COL2: vec3f = vec3f(0.0193, 0.1192, 0.9505);
const XYZ_TO_RGB_M_COL0: vec3f = vec3f(3.2406255, -1.537208, -0.4986286);
const XYZ_TO_RGB_M_COL1: vec3f = vec3f(-0.9689307, 1.8757561, 0.0415175);
const XYZ_TO_RGB_M_COL2: vec3f = vec3f(0.0557101, -0.2040211, 1.0569959);

fn RGB_TO_XYZ(rgb: vec3f) -> vec3f {
  return vec3f(dot(rgb, RGB_TO_XYZ_M_COL0), dot(rgb, RGB_TO_XYZ_M_COL1), dot(rgb, RGB_TO_XYZ_M_COL2));
}

fn XYZ_TO_RGB(xyz: vec3f) -> vec3f {
  return vec3f(dot(xyz, XYZ_TO_RGB_M_COL0), dot(xyz, XYZ_TO_RGB_M_COL1), dot(xyz, XYZ_TO_RGB_M_COL2));
}

fn XYZ_TO_LAB_F(x: f32) -> f32 {
  if (x > 0.00885645167) { return pow(x, 0.333333333); }
  return 7.78703703704 * x + 0.13793103448;
}

fn XYZ_TO_LAB(xyz: vec3f) -> vec3f {
  let xyz_scaled = vec3f(XYZ_TO_LAB_F(xyz.x / D65_WHITE.x), XYZ_TO_LAB_F(xyz.y / D65_WHITE.y), XYZ_TO_LAB_F(xyz.z / D65_WHITE.z));
  return vec3f(116.0 * xyz_scaled.y - 16.0, 500.0 * (xyz_scaled.x - xyz_scaled.y), 200.0 * (xyz_scaled.y - xyz_scaled.z));
}

fn RGB_TO_LAB(rgb: vec3f) -> vec3f { return XYZ_TO_LAB(RGB_TO_XYZ(rgb)); }
fn LAB_TO_LCH(Lab: vec3f) -> vec3f { return vec3f(Lab.x, sqrt(dot(Lab.yz, Lab.yz)), atan2(Lab.z, Lab.y) * 57.2957795131); }
fn RGB_TO_LCH(rgb: vec3f) -> vec3f { return LAB_TO_LCH(RGB_TO_LAB(rgb)); }

fn LAB_TO_XYZ_F(x: f32) -> f32 {
  if (x > 0.206897) { return x * x * x; }
  return 0.12841854934 * (x - 0.137931034);
}

fn LAB_TO_XYZ(Lab: vec3f) -> vec3f {
  let w = (Lab.x + 16.0) / 116.0;
  return D65_WHITE * vec3f(LAB_TO_XYZ_F(w + Lab.y / 500.0), LAB_TO_XYZ_F(w), LAB_TO_XYZ_F(w - Lab.z / 200.0));
}

fn LAB_TO_RGB(lab: vec3f) -> vec3f { return XYZ_TO_RGB(LAB_TO_XYZ(lab)); }

fn LCH_TO_LAB(LCh: vec3f) -> vec3f {
  return vec3f(LCh.x, LCh.y * cos(LCh.z * 0.01745329251), LCh.y * sin(LCh.z * 0.01745329251));
}

fn LCH_TO_RGB(lch: vec3f) -> vec3f { return LAB_TO_RGB(LCH_TO_LAB(lch)); }

fn vec2ToAngle(v: vec2f) -> f32 {
  var angle = atan2(v.y, v.x);
  if (angle < 0.0) { angle += 2.0 * PI; }
  return angle;
}

fn sampleBlurred(v_uv: vec2f, offset: vec2f) -> vec4f {
  if (usesTrafficLightReferenceBackdrop()) {
    return vec4f(trafficLightReferenceBackdrop(), 1.0);
  }
  if (u.u_bgType == 12 && u.u_bgTextureReady != 1) {
    // The OS owns the pixels behind a transparent window. Without a captured
    // desktop frame, keep a tinted translucent surface instead of painting a
    // bright white placeholder over the window.
    return vec4f(u.u_tint.rgb, 1.0);
  }
  let blurred = textureSampleLevel(u_blurredBg, u_sampler, v_uv + offset, 0.0);
  return blurred;
}

fn getTextureDispersion(v_uv: vec2f, mixRate: f32, offset: vec2f, factor: f32) -> vec4f {
  var pixel = vec4f(1.0);
  if (usesTrafficLightReferenceBackdrop()) {
    // Keep this reference substrate spatially uniform. The traffic-light
    // body still evaluates its normal/refraction path, but dark-mode scene
    // pixels cannot alter the calibrated button material.
    return vec4f(trafficLightReferenceBackdrop(), 1.0);
  }
  if (u.u_bgType == 12 && u.u_bgTextureReady != 1) {
    // The first transparent layer has no sampleable desktop backdrop yet.
    // Later layers use bgType 13 and continue sampling the prior composite.
    pixel = vec4f(u.u_tint.rgb, 1.0);
    return pixel;
  }
  let bgR = textureSampleLevel(u_bg, u_sampler, v_uv + offset * (1.0 - (N_R - 1.0) * factor), 0.0).r;
  let bgG = textureSampleLevel(u_bg, u_sampler, v_uv + offset * (1.0 - (N_G - 1.0) * factor), 0.0).g;
  let bgB = textureSampleLevel(u_bg, u_sampler, v_uv + offset * (1.0 - (N_B - 1.0) * factor), 0.0).b;
  let blurR = textureSampleLevel(u_blurredBg, u_sampler, v_uv + offset * (1.0 - (N_R - 1.0) * factor), 0.0).r;
  let blurG = textureSampleLevel(u_blurredBg, u_sampler, v_uv + offset * (1.0 - (N_G - 1.0) * factor), 0.0).g;
  let blurB = textureSampleLevel(u_blurredBg, u_sampler, v_uv + offset * (1.0 - (N_B - 1.0) * factor), 0.0).b;
  pixel.r = mix(bgR, blurR, mixRate);
  pixel.g = mix(bgG, blurG, mixRate);
  pixel.b = mix(bgB, blurB, mixRate);
  return pixel;
}

fn luminance(color: vec3f) -> f32 {
  return dot(color, vec3f(0.2126, 0.7152, 0.0722));
}

fn backdropStatsActual(v_uv: vec2f) -> vec2f {
  let texel = 1.0 / u.u_resolution;
  let center = sampleActualBackdrop(v_uv).rgb;
  let north = sampleActualBackdrop(v_uv + vec2f(0.0, texel.y * 2.0)).rgb;
  let south = sampleActualBackdrop(v_uv - vec2f(0.0, texel.y * 2.0)).rgb;
  let east = sampleActualBackdrop(v_uv + vec2f(texel.x * 2.0, 0.0)).rgb;
  let west = sampleActualBackdrop(v_uv - vec2f(texel.x * 2.0, 0.0)).rgb;
  let center_luma = luminance(center);
  let neighbor_luma = vec4f(luminance(north), luminance(south), luminance(east), luminance(west));
  let local_min = min(center_luma, min(min(neighbor_luma.x, neighbor_luma.y), min(neighbor_luma.z, neighbor_luma.w)));
  let local_max = max(center_luma, max(max(neighbor_luma.x, neighbor_luma.y), max(neighbor_luma.z, neighbor_luma.w)));
  return vec2f(center_luma, local_max - local_min);
}

fn backdropStats(v_uv: vec2f) -> vec2f {
  if (usesTrafficLightReferenceBackdrop()) {
    return vec2f(luminance(trafficLightReferenceBackdrop()), 0.0);
  }
  if (u.u_bgType == 12 && u.u_bgTextureReady != 1) {
    return vec2f(u.u_environmentLuminance, 0.0);
  }
  let texel = 1.0 / u.u_resolution;
  let center = textureSampleLevel(u_bg, u_sampler, v_uv, 0.0).rgb;
  let north = textureSampleLevel(u_bg, u_sampler, v_uv + vec2f(0.0, texel.y * 2.0), 0.0).rgb;
  let south = textureSampleLevel(u_bg, u_sampler, v_uv - vec2f(0.0, texel.y * 2.0), 0.0).rgb;
  let east = textureSampleLevel(u_bg, u_sampler, v_uv + vec2f(texel.x * 2.0, 0.0), 0.0).rgb;
  let west = textureSampleLevel(u_bg, u_sampler, v_uv - vec2f(texel.x * 2.0, 0.0), 0.0).rgb;
  let center_luma = luminance(center);
  let neighbor_luma = vec4f(luminance(north), luminance(south), luminance(east), luminance(west));
  let local_min = min(center_luma, min(min(neighbor_luma.x, neighbor_luma.y), min(neighbor_luma.z, neighbor_luma.w)));
  let local_max = max(center_luma, max(max(neighbor_luma.x, neighbor_luma.y), max(neighbor_luma.z, neighbor_luma.w)));
  return vec2f(center_luma, local_max - local_min);
}

fn ambientBackdrop(v_uv: vec2f) -> vec3f {
  if (usesTrafficLightReferenceBackdrop()) {
    return trafficLightReferenceBackdrop();
  }
  if (u.u_bgType == 12 && u.u_bgTextureReady != 1) {
    return u.u_tint.rgb;
  }
  let texel = 1.0 / u.u_resolution;
  let spread = texel * (6.0 + 4.0 * clamp(u.u_interactionState.z, 0.0, 1.0));
  let samples = textureSampleLevel(u_bg, u_sampler, v_uv + vec2f(spread.x, 0.0), 0.0).rgb
    + textureSampleLevel(u_bg, u_sampler, v_uv - vec2f(spread.x, 0.0), 0.0).rgb
    + textureSampleLevel(u_bg, u_sampler, v_uv + vec2f(0.0, spread.y), 0.0).rgb
    + textureSampleLevel(u_bg, u_sampler, v_uv - vec2f(0.0, spread.y), 0.0).rgb;
  // Preserve extended-linear values from lower glass layers. SDR backdrop
  // captures remain within 0..1, while EDR reflections may legitimately
  // exceed paper white.
  return max(samples * 0.25, vec3f(0.0));
}

fn interactionLight(pixel: vec2f) -> f32 {
  if (u.u_interaction <= 0.0001 || featureEnabled(FEATURE_REDUCED_MOTION)) {
    return 0.0;
  }
  // Traffic-light hover is coordinated at the group level so its glyphs reveal
  // together; letting that flag drive a per-control optical spot would light up
  // every sibling button. Their press response is a whole-body brightening (see
  // the compose stage below), so this pointer-localised excitation does not
  // apply to them at all.
  if (isTrafficLight()) {
    return 0.0;
  }
  let radius = max(min(u.u_shapeWidth, u.u_shapeHeight) * u.u_dpr * 0.70, 32.0);
  let springPointer = mix(u.u_mouse, u.u_interactionState.xy, 0.65);
  return exp(-length(pixel - springPointer) / radius) * u.u_interaction;
}

fn pressLight(pixel: vec2f) -> f32 {
  if (u.u_press <= 0.0001 || featureEnabled(FEATURE_REDUCED_MOTION)) {
    return 0.0;
  }
  // A press briefly excites the whole control, with a mild concentration at
  // the pointer. This is intentionally broad: the traffic-light button reads
  // as a thin material being energized, not as a painted cursor spot.
  let radius = max(min(u.u_shapeWidth, u.u_shapeHeight) * u.u_dpr * 0.78, 32.0);
  let springPointer = mix(u.u_mouse, u.u_interactionState.xy, 0.65);
  let localFocus = exp(-length(pixel - springPointer) / radius);
  return u.u_press * (0.62 + 0.38 * localFocus);
}

struct InterfaceResponse {
  radiance: vec3f,
  reflectance: f32,
};

struct BodyResponse {
  transmission: f32,
  uniformLight: f32,
  highlight: f32,
  sideShadow: f32,
  curvatureShadow: f32,
  upperAbsorption: f32,
};

fn isTrafficLight() -> bool {
  return featureEnabled(FEATURE_TRAFFIC_LIGHT)
    && abs(u.u_shapeWidth - u.u_shapeHeight) < 0.5
    && u.u_shapeRadius > min(u.u_shapeWidth, u.u_shapeHeight) * 0.45;
}

fn trafficLightLowerFacing(merged: f32, p1: vec2f, p2: vec2f, pixel: vec2f) -> f32 {
  if (!isTrafficLight()) {
    return 0.0;
  }
  let bodyNormal = refractionBodyNormal(merged, p1, p2, pixel);
  // The lower-facing part of the curved body is the direction in which the
  // native control becomes more transmissive. Deriving this from the normal
  // keeps the response tied to the optical surface instead of painting a
  // screen-space vertical gradient.
  return clamp(-bodyNormal.y, 0.0, 1.0);
}

fn trafficLightThinness(merged: f32, p1: vec2f, p2: vec2f, pixel: vec2f) -> f32 {
  if (!isTrafficLight()) {
    return 0.0;
  }
  // Treat the lower-facing part of the curved body as the thinner side of
  // the laminated control. The smoothstep is the thickness transition: it
  // changes the optical depth continuously instead of introducing a painted
  // lower ellipse or a hard opacity boundary.
  return smoothstep(0.12, 0.82, trafficLightLowerFacing(merged, p1, p2, pixel));
}

fn trafficLightThicknessFactor(merged: f32, p1: vec2f, p2: vec2f, pixel: vec2f) -> f32 {
  return mix(
    1.0,
    TRAFFIC_LIGHT_MIN_THICKNESS,
    trafficLightThinness(merged, p1, p2, pixel),
  );
}

fn trafficLightTransmissionBackdrop(bodyColor: vec3f, backdrop: vec3f) -> vec3f {
  // Lower thickness is allowed to reveal the backdrop's colour and texture,
  // but a dark backdrop must not turn the thinner hemisphere into a dark
  // stripe. Preserve the pre-transmission luminance and only compensate the
  // transmitted source when it is darker; this is a luminance guard, not a
  // painted white highlight or an emissive ellipse.
  let body = max(bodyColor, vec3f(0.0));
  let source = max(backdrop, vec3f(0.0));
  let bodyLuminance = luminance(body);
  let sourceLuminance = luminance(source);
  if (sourceLuminance <= 0.0001) {
    return body;
  }
  let gain = clamp(bodyLuminance / sourceLuminance, 1.0, 3.0);
  return source * gain;
}

fn trafficLightPigment() -> vec3f {
  // The red source has a full red channel already, so increasing that channel
  // cannot make it brighter in SDR. Work in linear light instead: preserve
  // the hue, give red a controlled exposure gain, and add a small red-only
  // saturation lift before the material is composed with the backdrop.
  let tint = u.u_tint.rgb;
  let secondary = max(tint.g, tint.b);
  let redSignal = smoothstep(0.65, 0.90, tint.r - secondary);
  let luminanceValue = luminance(tint);
  let saturation = 1.0 + redSignal * 0.22;
  let saturatedTint = max(
    mix(vec3f(luminanceValue), tint, saturation),
    vec3f(0.0),
  );
  let exposure = 1.0 + redSignal * 0.55;
  return saturatedTint * exposure;
}

fn calibrateTrafficLightColor(color: vec3f) -> vec3f {
  // The center-point calibration is performed in linear light. The current
  // red control already clips its R channel at the SDR display limit, while
  // its G/B channels remain too high: that is why increasing red exposure was
  // visually ineffective and made the control look washed out instead.
  //
  // These gains map the measured 64 pt center (approximately sRGB
  // 254/119/106) toward the native macOS center (approximately 241/93/84).
  // Keep the correction gated by the tint hue so yellow and green controls,
  // and every non-traffic-light glass role, retain their existing response.
  let tint = u.u_tint.rgb;
  let secondary = max(tint.g, tint.b);
  let redSignal = smoothstep(0.65, 0.90, tint.r - secondary);
  let calibrated = color * vec3f(0.89, 0.59, 0.62);
  let pressed = calibrated
    + TRAFFIC_LIGHT_RED_PRESS_LIFT * clamp(u.u_press, 0.0, 1.0);
  return mix(color, pressed, redSignal);
}

fn trafficLightBodyResponse(merged: f32, p1: vec2f, p2: vec2f, pixel: vec2f) -> BodyResponse {
  // Traffic lights use a spherical body. Keep this response
  // disabled for capsules and panels so their existing material profiles are
  // unchanged.
  if (!isTrafficLight() || merged >= 0.0 || featureEnabled(FEATURE_REDUCED_TRANSPARENCY)) {
    return BodyResponse(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  }

  let bodyNormal = refractionBodyNormal(merged, p1, p2, pixel);
  let lowerFacing = clamp(-bodyNormal.y, 0.0, 1.0);

  // Keep the source light uniform. The lower half becomes brighter because
  // its continuous thickness field transmits more of the already-lit
  // backdrop; adding a separate lower area light would create a painted
  // ellipse and confuse transmission with emission.
  let thinness = trafficLightThinness(merged, p1, p2, pixel);
  let transmission = lowerFacing * mix(0.05, 0.22, thinness);
  // The incident field is uniform across the material. The lower hemisphere
  // releases slightly more of that same energy because its optical thickness
  // is lower; there is no screen-space position or hand-painted highlight.
  // Ease the lower release once more so the incident field reads as a broad,
  // soft material response instead of a concentrated lower-half glow.
  let softThinness = smoothstep(0.0, 1.0, thinness);
  // Centre light, from `GlassMaterial::core_light`: a flat incident field, a
  // smaller release toward the thinner lower hemisphere, and the droplet core.
  //
  // The reference appearance describes the core in screen-plane terms
  // (`dy/r`, `dx/r`, `d/r`); on the sphere those are exactly the body normal's
  // components, so the graded terms are derived from `bodyNormal` instead:
  // `-z` is the radial falloff from the optical centre outward, `y` the
  // vertical axis and `x` the horizontal roll-off.
  let coreRadial = 1.0 - clamp(bodyNormal.z, 0.0, 1.0);
  let coreFalloff = pow(coreRadial, clamp(u.u_coreLight.z, 1.0, 8.0));
  let coreLift = (1.0 - coreFalloff) * u.u_coreLightGradient.x;
  let vertical = clamp((bodyNormal.y + 0.15) / 1.15, 0.0, 1.0);
  let verticalGlow = pow(vertical, clamp(u.u_coreLight.w, 0.25, 4.0));
  let horizontalRolloff = pow(
    max(1.0 - bodyNormal.x * bodyNormal.x, 0.0),
    clamp(u.u_coreLightGradient.z, 0.05, 2.0),
  );
  let axialGlow = verticalGlow * horizontalRolloff * u.u_coreLightGradient.y;
  let uniformLight = (
    u.u_coreLight.x
      + u.u_coreLight.y * softThinness
      + coreLift
      + axialGlow
  ) * clamp(u.u_tint.a, 0.0, 1.0) * u.u_adaptive.y;
  let highlight = 0.0;
  // Lateral grazing normals receive the stronger side attenuation. There is
  // no separately painted white outline on the upper/lower contour: the
  // surrounding dark response stays inside the curved material body.
  let sideShadow = 0.0;
  // At the curved perimeter the surface normal turns away from the viewer,
  // so less of the coloured substrate is visible. Express the dark contour
  // through that grazing-angle response and the SDF distance; this is not an
  // additional geometric stroke and it does not create a white rim.
  // The dark perimeter is the absorbing side of the curved coating. Its
  // width is expressed in physical pixels and its strength comes from the
  // grazing normal, so this is a material response rather than a black
  // stroke drawn on top of the circle.
  // Keep the absorbing coat compact, but wide enough to survive the traffic
  // antialiasing and read as a continuous material edge at 1x and 2x DPR.
  let insideDistance = max(-merged * (u.u_resolution.y / u.u_dpr), 0.0);
  let trafficLightScale = trafficLightSizeScale();
  // The upper/lower arcs turn toward the viewer more quickly than the
  // lateral sides. Give those arcs a shorter optical falloff so the top does
  // not become a broad grey strip while the side wall keeps its thickness.
  // Keep the dark side wall out of most of the upper/lower arcs. The
  // transition is centred around a 45-degree surface angle instead of
  // gradually spreading over the entire quadrant.
  let lateralAngle = clamp(abs(bodyNormal.x), 0.0, 1.0);
  // The reference weights the lateral wall by a power of the body normal
  // (`|nx|^2`) rather than a fixed transition, which concentrates the absorbing
  // rim on the left and right while leaving a residual on the vertical arcs.
  let lateralEdge = pow(lateralAngle, clamp(u.u_rimProfile.x, 0.25, 16.0));
  let edgeWidthScale = mix(
    TRAFFIC_LIGHT_VERTICAL_EDGE_WIDTH,
    1.0,
    lateralEdge,
  );
  let edgeSharpness = mix(
    TRAFFIC_LIGHT_VERTICAL_EDGE_SHARPNESS,
    TRAFFIC_LIGHT_EDGE_SHARPNESS,
    lateralEdge,
  );
  // Use the same focused SDF transition as the search field. The previous
  // smoother-step made this inner absorption feather across too many pixels
  // and read as a soft duplicate contour at the top and bottom.
  let coatingPosition = clamp(
    insideDistance
      / (TRAFFIC_LIGHT_COATING_BAND * trafficLightScale * edgeWidthScale),
    0.0,
    1.0,
  );
  // Hold the dark material response across most of the widened band, then
  // return to the chromatic body decisively near its inner edge. This keeps
  // the border broad without turning it into a low-contrast soft shadow.
  let coatingBand = 1.0 - pow(
    smoothstep(0.0, 1.0, coatingPosition),
    edgeSharpness,
  );
  let grazingCurvature = pow(
    clamp(1.0 - bodyNormal.z, 0.0, 1.0),
    clamp(u.u_rimProfile.z, 0.1, 4.0),
  );
  // Match the search field's directional edge response: the lateral sides
  // turn away from the incident field and absorb more, while the upper and
  // lower contour keeps a softer residual edge. This is derived from the
  // curved body normal, not a screen-space black stroke.
  // At the upper/lower contour the search-field environment response already
  // supplies the pale rim. Leave only a small residual absorption there so it
  // does not sit underneath that rim as a second dark line.
  // Keep a visible absorption floor on the upper/lower arc as well. A zero
  // floor makes the dark material response collapse into two lateral bars;
  // the native control reads as one continuous rounded perimeter, with the
  // sides still substantially darker than the top and bottom.
  let directionalAbsorption = mix(
    clamp(u.u_rimProfile.y, 0.0, 1.0),
    1.0,
    lateralEdge,
  );
  let coatingAbsorption = coatingBand
    * mix(0.24, 0.82, grazingCurvature)
    * directionalAbsorption
    * TRAFFIC_LIGHT_EDGE_DARKNESS;
  let curvatureShadow = clamp(coatingAbsorption, 0.0, 0.92);
  // The upper-facing hemisphere has no light source in this material model.
  // Apply only a small normal-derived absorption there so the saturated base
  // colour does not read as an unintended upper highlight.
  let upperFacing = clamp(bodyNormal.y, 0.0, 1.0);
  let upperAbsorption = smoothstep(0.08, 0.78, upperFacing) * 0.06;
  return BodyResponse(transmission, uniformLight, highlight, sideShadow, curvatureShadow, upperAbsorption);
}

fn interfaceResponseForProfile(
  profile: SurfaceProfile,
  localEnvironment: vec3f,
  stats: vec2f,
) -> InterfaceResponse {
  let roughnessControl = clamp(u.u_glareConvergence, 0.0, 1.0);
  let coatingRoughness = mix(0.16, SURFACE_MIN_ROUGHNESS, roughnessControl);
  let roughness = mix(SURFACE_MAX_ROUGHNESS, coatingRoughness, profile.rim);
  let view = vec3f(0.0, 0.0, 1.0);
  let reflectedDirection = reflect(-view, profile.normal);
  // Approximate the convolution of two wide horizontal strip lights with the
  // rough GGX lobe. A power lobe made one strip collapse to a dim hairline;
  // the widened transition keeps both horizontal rims crisp and luminous.
  // The clear-coat is spatially narrow, so its strip-light lobe must overlap
  // the steep outer normal. If the angular gate starts too late, the normal
  // reaches the light only after the coating weight has already vanished.
  let verticalLobeStart = mix(0.50, 0.40, roughness / SURFACE_MAX_ROUGHNESS);
  let verticalLobeEnd = mix(0.72, 0.62, roughness / SURFACE_MAX_ROUGHNESS);
  let verticalAxis = clamp(abs(reflectedDirection.y), 0.0, 1.0);
  let verticalCore = smoothstep(
    verticalLobeStart,
    verticalLobeEnd,
    verticalAxis,
  );
  // A rough microfacet lobe has a low-energy tail rather than a hard angular
  // cutoff. The quadratic toe is zero on the lateral sides, so it softens the
  // upper/lower transition without washing out the dark vertical boundary.
  let verticalEnvironment = mix(verticalAxis * verticalAxis, verticalCore, 0.64);
  // The analytic environment has bright upper/lower strips and a much dimmer
  // lateral field. Roughness convolves those strips into a frosted response;
  // no orientation is explicitly multiplied by black.
  let lateralEnvironment = localEnvironment * 0.045 + vec3f(0.004);
  // Give the ceiling strip more energy for broad glass surfaces. Compact
  // traffic lights retain only a reduced fraction of this same strip field:
  // their lower illumination still comes from the curved-body response,
  // while the coat remains a dark grazing interface at the lateral sides.
  let broadVerticalIntensity = select(2.10, 2.55, reflectedDirection.y >= 0.0);
  // The search field uses this strip-light environment to keep its top and
  // bottom rims pale while the lateral environment remains dim. Traffic
  // lights use the same physical profile at reduced energy so their compact
  // contour does not become a bright painted band.
  let verticalIntensity = broadVerticalIntensity * select(
    1.0,
    TRAFFIC_LIGHT_VERTICAL_LIGHT_FACTOR,
    isTrafficLight(),
  );
  let localContrast = clamp(
    max(stats.y, abs(stats.x - u.u_environmentLuminance) * 0.5),
    0.0,
    1.0,
  );
  let surfaceGain = u.u_adaptive.y * mix(0.84, 1.16, localContrast);
  // Treat the broad strips as one prefiltered rough-environment lobe. Glare
  // changes that lobe's energy instead of adding another specular peak at a
  // different normal angle, which would read as a second concentric rim.
  let stripEnergy = 1.0 + 0.24 * u.u_glareFactor * surfaceGain;
  let verticalRadiance = vec3f(verticalIntensity * stripEnergy);
  let environmentRadiance = mix(lateralEnvironment, verticalRadiance, verticalEnvironment);

  // A laminated system control has more than the bare-air/glass interface:
  // its smooth outer coat and dim substrate reinforce grazing reflectance.
  // Strength represents that layered-interface gain, while Fresnel and the
  // surface normal still determine where the response can occur.
  let fresnelGain = mix(1.25, 10.0, clamp(u.u_refFresnelFactor, 0.0, 1.0));
  let reflectance = clamp(profile.fresnel * profile.rim * fresnelGain, 0.0, 0.55);

  // Lateral directions only see the dim environment, so their Fresnel energy
  // remains dark and frosted without a separately painted side shadow.
  // The rough vertical microfacets contribute a low-energy radiance tail over
  // the curved bevel. Keeping it out of `reflectance` preserves the narrow
  // dark boundary and prevents the shoulder from saturating into a white band.
  let highlightTail = verticalRadiance
    * verticalEnvironment
    * profile.highlightRim
    * select(0.030, 0.0, isTrafficLight());
  let reflectedRadiance = environmentRadiance * reflectance + highlightTail;
  return InterfaceResponse(reflectedRadiance, reflectance);
}

fn materialInterfaceResponse(
  merged: f32,
  p1: vec2f,
  p2: vec2f,
  pixel: vec2f,
  v_uv: vec2f,
  stats: vec2f,
) -> InterfaceResponse {
  let pixelsPerSdfUnit = u.u_resolution.y / u.u_dpr;
  let signedDistancePixels = merged * pixelsPerSdfUnit;
  if (signedDistancePixels > 1.0) {
    return InterfaceResponse(vec3f(0.0), 0.0);
  }

  let localEnvironment = ambientBackdrop(v_uv);
  if (abs(signedDistancePixels) <= surfaceBevelWidth() * 1.50 + 1.0) {
    // Integrate the full optical bevel over a 4x4 stratified subpixel grid.
    // Limiting these extra samples to the bevel keeps the rest of the glass
    // and the window at one sample per pixel while eliminating stair-steps
    // from both the narrow clear-coat and its soft highlight shoulder.
    let sampleOffsets = array<vec2f, 16>(
      vec2f(-0.375, -0.375),
      vec2f(-0.125, -0.375),
      vec2f( 0.125, -0.375),
      vec2f( 0.375, -0.375),
      vec2f(-0.375, -0.125),
      vec2f(-0.125, -0.125),
      vec2f( 0.125, -0.125),
      vec2f( 0.375, -0.125),
      vec2f(-0.375,  0.125),
      vec2f(-0.125,  0.125),
      vec2f( 0.125,  0.125),
      vec2f( 0.375,  0.125),
      vec2f(-0.375,  0.375),
      vec2f(-0.125,  0.375),
      vec2f( 0.125,  0.375),
      vec2f( 0.375,  0.375),
    );
    var integratedRadiance = vec3f(0.0);
    var integratedReflectance = 0.0;
    for (var sampleIndex = 0; sampleIndex < 16; sampleIndex += 1) {
      let samplePixel = pixel + sampleOffsets[sampleIndex] * u.u_dpr;
      let sampleMerged = mainSDF(p1, p2, samplePixel);
      if (sampleMerged < 0.0) {
        let response = interfaceResponseForProfile(
          surfaceProfile(sampleMerged, p1, p2, samplePixel),
          localEnvironment,
          stats,
        );
        integratedRadiance += response.radiance;
        integratedReflectance += response.reflectance;
      }
    }
    return InterfaceResponse(integratedRadiance * 0.0625, integratedReflectance * 0.0625);
  }

  if (merged >= 0.0) {
    return InterfaceResponse(vec3f(0.0), 0.0);
  }
  return interfaceResponseForProfile(
    surfaceProfile(merged, p1, p2, pixel),
    localEnvironment,
    stats,
  );
}

fn encodeSrgbChannel(value: f32) -> f32 {
  let v = clamp(value, 0.0, 1.0);
  if (v <= 0.0031308) {
    return v * 12.92;
  }
  return 1.055 * pow(v, 1.0 / 2.4) - 0.055;
}

fn decodeSrgbChannel(value: f32) -> f32 {
  let v = clamp(value, 0.0, 1.0);
  if (v <= 0.04045) {
    return v / 12.92;
  }
  return pow((v + 0.055) / 1.055, 2.4);
}

fn encodeSrgb(colour: vec3f) -> vec3f {
  return vec3f(
    encodeSrgbChannel(colour.r),
    encodeSrgbChannel(colour.g),
    encodeSrgbChannel(colour.b),
  );
}

fn decodeSrgb(colour: vec3f) -> vec3f {
  return vec3f(
    decodeSrgbChannel(colour.r),
    decodeSrgbChannel(colour.g),
    decodeSrgbChannel(colour.b),
  );
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4f, @location(0) v_uv: vec2f) -> @location(0) vec4f {
  let u_resolution1x = u.u_resolution / u.u_dpr;
  let pixel = vec2f(frag_coord.x, u.u_resolution.y - frag_coord.y);
  let p1 = (vec2f(0.0) - u.u_resolution * 0.5) / u.u_resolution.y;
  let p2 = (vec2f(0.0) - u.u_mouseSpring) / u.u_resolution.y;
  let merged = mainSDF(p1, p2, pixel);
  let antialiasWidth = select(
    max(fwidth(merged) * 1.80, 0.90 / u.u_resolution.y),
    max(fwidth(merged) * 1.15, 0.55 / u.u_resolution.y),
    isTrafficLight(),
  );
  let shapeAlpha = 1.0 - smoothstep(-antialiasWidth, antialiasWidth, merged);

  // Flat reference bead: the pre-rasterised traffic-light appearance evaluated
  // per pixel. It deliberately bypasses the physical composition below -- there
  // is no backdrop transmission, refraction, or Fresnel -- so the control reads
  // as a baked bead rather than a glass node.
  if (isTrafficLightBead()) {
    let beadBackdrop = sampleActualBackdrop(v_uv);
    // Carve the disc from the node's own silhouette. The merged field also
    // contains shape 1 and every fused layer, so using it as the coverage made
    // the bead paint whatever the group happened to merge into.
    let beadSdf = shapeSDF(p2, pixel);
    let beadCoverage = 1.0 - smoothstep(-antialiasWidth, antialiasWidth, beadSdf);
    // Derive the offset from the same control point the SDF uses (`p2` is the
    // shape centre in the space `mainSDF` consumes), so the bead can never
    // disagree with the shape's position and radius. Working in that normalised
    // space avoids re-deriving the screen transform by hand.
    let beadOffset = (vec2f(0.0) - pixel) / u.u_resolution.y - p2;
    let beadRadiusN = max(
      min(u.u_shapeWidth, u.u_shapeHeight) * 0.5 / u.u_resolution.y,
      1e-6,
    );
    let nx = beadOffset.x / beadRadiusN;
    let ny = beadOffset.y / beadRadiusN;
    // Back to physical pixels for the pixel-wide rim spans.
    let beadRadius = beadRadiusN * u.u_resolution.y;
    let inside = beadRadius - length(beadOffset) * u.u_resolution.y;

    var bead = u.u_tint.rgb;
    if (inside > 0.0) {
      // B's rasteriser ran every one of these constants on sRGB-encoded channel
      // values and wrote the result out as sRGB bytes. The shader otherwise
      // works in linear light, where the same numbers land noticeably lighter
      // (the 0.88 base is a 12% reduction in sRGB but only 1.5% in linear).
      // Encode into that space here and decode at the end, so the render target
      // receives the value the rasteriser would have written.
      let tintSrgb = encodeSrgb(u.u_tint.rgb);
      let tDist = clamp(inside / beadRadius, 0.0, 1.0);
      let q = 1.0 - tDist;
      let falloff = q * q * q * q
        + 4.0 * q * q * q * tDist * u.u_beadA.x
        + 6.0 * q * q * tDist * tDist * u.u_beadA.y
        + 4.0 * q * tDist * tDist * tDist * u.u_beadA.z;

      let dark = clamp(u.u_beadC.w, 0.0, 1.0);
      let glow = clamp(u.u_beadA.w, 0.0, 2.0) * mix(u.u_beadC.y, u.u_beadC.z, dark);
      // Vertical axial internal glow, not a circular one.
      let vertical = clamp((ny + 0.15) / 1.15, 0.0, 1.0);
      let axial = pow(vertical, 1.35)
        * pow(max(1.0 - nx * nx, 0.0), 0.25) * 0.42 * glow;
      let coreLift = (1.0 - falloff) * clamp(u.u_beadB.x, 0.0, 0.5) * glow;
      var colour = tintSrgb * (0.88 + coreLift + axial);

      // Whole-control pointer response, shared with every other variant.
      let engaged = clamp(u.u_interaction, 0.0, 1.0);
      let pressed = clamp(u.u_press, 0.0, 1.0);
      colour *= 1.0
        + u.u_interactionResponse.x * engaged
        + u.u_interactionResponse.y * pressed;
      colour += tintSrgb * u.u_interactionResponse.z * pressed;

      // Mode-exclusive rim: a dark wall on the lateral sides for the light
      // appearance, a bright edge on the vertical arcs for the dark one.
      // The spans are authored in logical points and grow with the control, so
      // convert them to physical pixels with the surface's own scale factor.
      let renderScale = max(u.u_renderScale, 1.0);
      let logicalSize = max(beadRadius * 2.0 / renderScale, 1.0);
      let span = max(1.8, 2.4 * sqrt(logicalSize / 14.0)) * renderScale;
      let rimSpan = max(span * clamp(u.u_beadC.x, 0.5, 2.0), 1.0);
      let sideWall = pow(clamp(abs(nx), 0.0, 1.0), 2.0);
      let darkDrop = pow(1.0 - clamp(inside / rimSpan, 0.0, 1.0), 2.0)
        * (52.0 / 255.0) * sideWall * clamp(u.u_beadB.z, 0.0, 3.0);
      let edgeSpan = max(span * clamp(u.u_beadB.w, 0.5, 2.0), 1.0);
      let sharp = pow(1.0 - clamp(inside / edgeSpan, 0.0, 1.0), 3.0) * 0.95;
      let halo = pow(1.0 - clamp(inside / beadRadius, 0.0, 1.0), 2.0) * 0.06;
      let bright = (sharp + halo)
        * pow(clamp(abs(ny), 0.0, 1.0), 1.8)
        * clamp(u.u_beadB.y, 0.0, 2.0);
      colour = max(colour - vec3f(darkDrop * (1.0 - dark)), vec3f(0.0));
      colour = min(colour + vec3f(bright * dark), vec3f(1.0));
      bead = decodeSrgb(colour);
    }

    // Re-emit the sampled backdrop outside the disc. The glass pass replaces
    // the target instead of blending into it, so returning the bead tint with a
    // zero alpha still painted the full effect quad: that is what left a solid
    // tinted square behind the control. Compositing here is the same contract
    // the physical path below uses.
    let beadOpacity = beadCoverage * clamp(u.u_opacity, 0.0, 1.0);
    return vec4f(
      mix(beadBackdrop.rgb, bead, beadOpacity),
      mix(beadBackdrop.a, 1.0, beadOpacity),
    );
  }
  let stats = backdropStats(v_uv);
  let contentContrast = clamp(max(stats.y, abs(stats.x - u.u_environmentLuminance) * 0.35), 0.0, 1.0);
  let variantTintFactor = select(1.0, 0.78, featureEnabled(FEATURE_CLEAR_VARIANT));
  let physicalLowerLight = trafficLightPhysicalLowerLight(merged, p1, p2, pixel);
  // Native traffic lights keep a dense chromatic upper body, while the lower
  // curved hemisphere transmits more of the underlying layer. The gradient is
  // derived from the surface normal, not painted in screen space.
  let trafficLightTintCoverage = 0.995 - trafficLightThinness(merged, p1, p2, pixel) * 0.24;
  let physicalTintCoverage = mix(
    TRAFFIC_LIGHT_PHYSICAL_TINT_COVERAGE,
    clamp(u.u_trafficLight.z, 0.0, 1.0),
    physicalLowerLight,
  );
  let physicalOrDefaultTintCoverage = select(
    0.8,
    physicalTintCoverage,
    featureEnabled(FEATURE_TRAFFIC_LIGHT_PHYSICAL),
  );
  let tintCoverage = select(physicalOrDefaultTintCoverage, trafficLightTintCoverage, isTrafficLight());
  let adaptiveTintAlpha = clamp(
    u.u_tint.a * mix(0.86, 1.14, contentContrast) * u.u_adaptive.x * variantTintFactor,
    0.0,
    1.0,
  );
  var outColor: vec4f;
  var surfaceReflection = vec3f(0.0);
  var surfaceReflectance = 0.0;

  if (merged < 0.005) {
    if (u.u_refStrength <= 0.0001) {
      outColor = sampleBlurred(v_uv, vec2f(0.0));
      outColor = mix(outColor, vec4f(u.u_tint.r, u.u_tint.g, u.u_tint.b, 1.0), adaptiveTintAlpha * tintCoverage);
    } else {
      // Start from the source repository's liquid optical base. Its nonlinear
      // Snell profile creates the strong magnifying fold at the contour; the
      // enhanced environment interface is layered on only after transmission.
      let insideDistance = -merged * u_resolution1x.y;
      let refractionThickness = u.u_refThickness * trafficLightThicknessFactor(merged, p1, p2, pixel);
      let incidenceRatio = 1.0 - insideDistance / refractionThickness;
      let thetaI = safeAsin(pow(incidenceRatio, 2.0));
      let thetaT = safeAsin(1.0 / max(u.u_refFactor, 1.0001) * sin(thetaI));
      var edgeFactor = -tan(thetaT - thetaI);
      if (insideDistance >= refractionThickness) {
        edgeFactor = 0.0;
      }

      let opticalNormal = sourceOpticalNormal(p1, p2, pixel);
      let refOffset = -opticalNormal
        * edgeFactor
        * 0.05
        * u.u_refStrength
        * u.u_dpr
        * vec2f(
          u.u_resolution.y / (u_resolution1x.x * u.u_dpr),
          -1.0,
      );
      var blurMixRate = clamp(insideDistance / u.u_refThickness, 0.0, 1.0);
      if (featureEnabled(FEATURE_TRAFFIC_LIGHT_PHYSICAL)) {
        // Physical traffic lights expose the clear/scattered blend directly.
        // This remains independent from the radius of the blurred backdrop:
        // zero is a clear refracted sample and one is the current fully
        // scattered material response.
        blurMixRate = clamp(u.u_trafficLightLight.z, 0.0, 1.0);
      } else if (featureEnabled(FEATURE_EDGE_BLUR)) {
        blurMixRate = 1.0;
      }
      if (isTrafficLight()) {
        // A thinner lower hemisphere should reveal more of the unblurred
        // source. Keep the control opaque, but reduce the scattering mix
        // continuously instead of making the lower half a transparent cutout.
        let thinness = trafficLightThinness(merged, p1, p2, pixel);
        blurMixRate = mix(blurMixRate, blurMixRate * 0.45, thinness);
      }
      let refracted = getTextureDispersion(
        v_uv,
        blurMixRate,
        refOffset,
        u.u_refDispersion,
      );
      outColor = mix(refracted, vec4f(u.u_tint.rgb, 1.0), adaptiveTintAlpha * tintCoverage);

      // The source renderer's white Fresnel shoulder is useful for broad
      // capsules, but it creates a white halo before the dark perimeter on
      // traffic lights. Their edge is already supplied by the
      // material clear-coat below, so keep the source shoulder for capsules
      // and omit it for traffic lights.
      if (!isTrafficLight()) {
        let fresnelFactor = clamp(
          pow(
            1.0
              + merged * u_resolution1x.y / 1500.0
                * pow(500.0 / u.u_refFresnelRange, 2.0)
                + u.u_refFresnelHardness,
            5.0,
          ),
          0.0,
          1.0,
        );
        var fresnelTintLCH = RGB_TO_LCH(
          mix(vec3f(1.0), u.u_tint.rgb, adaptiveTintAlpha * 0.5),
        );
        fresnelTintLCH.x = clamp(
          fresnelTintLCH.x + 20.0 * fresnelFactor * u.u_refFresnelFactor,
          0.0,
          100.0,
        );
        outColor = mix(
          outColor,
          vec4f(LCH_TO_RGB(fresnelTintLCH), 1.0),
          fresnelFactor
            * u.u_refFresnelFactor
            * 0.7
            * length(opticalNormal),
        );
      }
    }
  } else {
    outColor = textureSampleLevel(u_bg, u_sampler, v_uv, 0.0);
  }

  // The physical traffic-light path still needs an absorptive coloured body;
  // tint coverage alone only attenuates the refracted backdrop. Stacking this
  // substrate over the existing 0.95 tint coverage leaves about 0.6% of the
  // original transmission in the active centre while keeping the spherical
  // interface response below intact.
  if (featureEnabled(FEATURE_TRAFFIC_LIGHT_PHYSICAL) && merged < 0.0) {
    let pigment = trafficLightPigment();
    let substrateCoverage = mix(
      clamp(u.u_trafficLight.x, 0.0, 1.0),
      clamp(u.u_trafficLight.y, 0.0, 1.0),
      physicalLowerLight,
    );
    let effectiveSubstrateCoverage = clamp(
      adaptiveTintAlpha * substrateCoverage,
      0.0,
      1.0,
    );
    outColor = vec4f(
      mix(outColor.rgb, pigment, effectiveSubstrateCoverage),
      outColor.a,
    );
    let angularLight =
      clamp(u.u_trafficLight.w, 0.0, 2.0)
      * pow(physicalLowerLight, 0.85)
      * u.u_adaptive.y;
    outColor = vec4f(outColor.rgb + pigment * angularLight, outColor.a);
  }

  if (merged < 0.0) {
    let edgeProximity = exp(-abs(merged) * u_resolution1x.y / 10.0);
    let ambientStrength = clamp(
      u.u_adaptive.y * (0.010 + contentContrast * 0.030) * edgeProximity,
      0.0,
      0.08,
    );
    outColor = vec4f(mix(outColor.rgb, ambientBackdrop(v_uv), ambientStrength), outColor.a);
    let interaction = interactionLight(pixel);
    if (!isTrafficLight()) {
      outColor = vec4f(mix(outColor.rgb, vec3f(1.0), clamp(interaction * 0.12, 0.0, 0.12)), outColor.a);
    }
    let press = pressLight(pixel);
    // Traffic lights use the calibrated press-state lift below. Keep the
    // legacy pigment excitation for other glass variants only; applying it
    // here as well would double-count the red press gain and over-raise G/B.
    if (!isTrafficLight()) {
      outColor = vec4f(
        min(outColor.rgb + trafficLightPigment() * press * 0.60, vec3f(1.0)),
        outColor.a,
      );
    }

    let body = trafficLightBodyResponse(merged, p1, p2, pixel);
    if (body.transmission > 0.0 || body.uniformLight > 0.0 || body.highlight > 0.0 || body.sideShadow > 0.0 || body.curvatureShadow > 0.0 || body.upperAbsorption > 0.0) {
      // Add the uniform incident field in linear light before transmission
      // and edge absorption. The tint supplies the wavelength, while the
      // scalar body response supplies only the material energy.
      outColor = vec4f(outColor.rgb + u.u_tint.rgb * body.uniformLight, outColor.a);
      if (u.u_bgTextureReady == 1 || usesTrafficLightReferenceBackdrop()) {
        let clearBackdrop = select(
          sampleActualBackdrop(v_uv).rgb,
          trafficLightReferenceBackdrop(),
          usesTrafficLightReferenceBackdrop(),
        );
        let compensatedBackdrop = trafficLightTransmissionBackdrop(outColor.rgb, clearBackdrop);
        outColor = vec4f(mix(outColor.rgb, compensatedBackdrop, body.transmission), outColor.a);
      }
      // Keep the lower-middle optical highlight visible without laying a
      // broad white veil over the calibrated traffic-light colour.
      let bodyHighlight = clamp(body.highlight * 0.11, 0.0, 0.15);
      outColor = vec4f(
        mix(outColor.rgb, vec3f(1.0), bodyHighlight)
          * (1.0 - body.sideShadow)
          * (1.0 - body.curvatureShadow)
          * (1.0 - body.upperAbsorption),
        outColor.a,
      );
    }

    if (isTrafficLight() && !featureEnabled(FEATURE_REDUCED_MOTION)) {
      // Whole-control interaction brightening, driven by
      // `GlassMaterial::interaction`.
      //
      // This runs *after* the body response on purpose: the sphere's lower half
      // is transmission-mixed and its rim carries the absorbing curvature
      // shadow, so an earlier gain would brighten only part of the control. A
      // single uniform gain on the composed body is what makes hover and press
      // read as "the control lit up" instead of "a highlight moved in". It is
      // applied after the shadow terms too, so the dark perimeter keeps its
      // relative depth while the whole control rises together.
      //
      // `u_interaction` is `max(hover, press, focus)`, so the hover term is
      // already engaged while the control is held; the press terms add to it.
      let engaged = clamp(u.u_interaction, 0.0, 1.0);
      let trafficPress = clamp(u.u_press, 0.0, 1.0);
      let gain = u.u_interactionResponse.x * engaged
        + u.u_interactionResponse.y * trafficPress;
      let lift = u.u_interactionResponse.z * trafficPress;
      if (gain > 0.0 || lift > 0.0) {
        outColor = vec4f(
          outColor.rgb * (1.0 + gain) + u.u_tint.rgb * lift,
          outColor.a,
        );
      }
    }
  }

  if (shapeAlpha > 0.0) {
    let surfaceInterface = materialInterfaceResponse(merged, p1, p2, pixel, v_uv, stats);
    surfaceReflection = surfaceInterface.radiance;
    surfaceReflectance = surfaceInterface.reflectance;
  }

  if (featureEnabled(FEATURE_INCREASED_CONTRAST)) {
    let border = 1.0 - smoothstep(0.15, 1.35, abs(merged) * u_resolution1x.y);
    var borderColor = vec3f(0.0);
    if (stats.x < 0.5) { borderColor = vec3f(1.0); }
    outColor = vec4f(mix(outColor.rgb, borderColor, border * 0.38), outColor.a);
  }

  // Keep the contour approximately one physical pixel wide at every surface
  // scale and DPR. A fixed SDF threshold made compact controls look blurry.
  // The final composite always uses the real scene backdrop. Only the
  // traffic-light material's internal optical reads use the light reference.
  let backdrop = sampleActualBackdrop(v_uv);
  let opacity = shapeAlpha * clamp(u.u_opacity, 0.0, 1.0);
  let transmitted = mix(backdrop.rgb, outColor.rgb, opacity);
  // Fresnel partitions the interface energy: reflected energy replaces the
  // corresponding transmitted backdrop energy. The rough environment strips
  // can add radiance above that term.
  // The interface response already contains subpixel coverage, so applying
  // shapeAlpha again would darken partially covered edge pixels twice.
  let interfaceReflectance = clamp(surfaceReflectance, 0.0, 0.55);
  let reflected = max(
    transmitted * (1.0 - interfaceReflectance) + surfaceReflection,
    vec3f(0.0),
  );
  // Apply the closed contour after the interface response. This keeps the
  // edge chromatic: it is the button tint at a darker value, rather than the
  // backdrop/reflection being multiplied into an achromatic gray. The amount
  // still follows the physical absorption field and the control opacity.
  let edgeAbsorption = trafficLightPhysicalEdgeAbsorption(merged, p1, p2, pixel);
  let darkness = clamp(u.u_trafficLightLight.w, 0.0, 4.0);
  let baseDarkness = clamp(darkness / 2.0, 0.0, 1.0);
  let extendedDarkness = clamp((darkness - 2.0) / 2.0, 0.0, 1.0);
  let edgeTintValue = clamp(
    mix(0.70, 0.36, baseDarkness) - 0.24 * extendedDarkness,
    0.12,
    0.70,
  );
  let edgeTint = u.u_tint.rgb * edgeTintValue;
  let edgeBlend = clamp(edgeAbsorption * opacity, 0.0, 0.92);
  let finished = mix(reflected, edgeTint, edgeBlend);
  // The draw call is intentionally larger than the circle so blur and
  // shadows have room to render. Keep the chromatic calibration inside the
  // SDF coverage; otherwise the red correction tints that rectangular effect
  // region and leaves a visible red box behind the control.
  let calibrated = calibrateTrafficLightColor(finished);
  let finalColor = mix(finished, calibrated, shapeAlpha);
  return vec4f(finalColor, mix(backdrop.a, 1.0, opacity));
}

// Composites the current node's SDF shadow onto the finished layer. The
// caller renders this after the glass pass over the node's expanded effect
// region, so the shadow never becomes an input to the material blur.
@fragment
fn fs_shadow(@builtin(position) frag_coord: vec4f, @location(0) v_uv: vec2f) -> @location(0) vec4f {
  let backdrop = sampleActualBackdrop(v_uv);
  let pixel = vec2f(frag_coord.x, u.u_resolution.y - frag_coord.y);
  // External shadows adapt to the actual scene, not to the light reference
  // substrate used inside a traffic-light button.
  let stats = backdropStatsActual(v_uv);
  let lightBackground = smoothstep(0.58, 0.96, stats.x);
  let contentContrast = clamp(max(stats.y, abs(stats.x - u.u_environmentLuminance) * 0.5), 0.0, 1.0);
  let backgroundAdaptation = mix(1.24, 0.68, lightBackground) * mix(0.86, 1.20, contentContrast);
  let shadow = clamp(shadowStrength(pixel) * backgroundAdaptation * u.u_adaptive.z, 0.0, 1.0);
  return vec4f(backdrop.rgb * (1.0 - shadow), max(backdrop.a, shadow));
}
