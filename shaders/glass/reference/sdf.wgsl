// Signed distance functions ported from liquid-glass-studio.

fn sdCircle(p: vec2f, r: f32) -> f32 {
  return length(p) - r;
}

fn continuousCornerSDF(p_in: vec2f, r: f32, n: f32) -> f32 {
  let p = abs(p_in);
  let exponent = max(n, 2.0);
  let v = pow(pow(p.x, exponent) + pow(p.y, exponent), 1.0 / exponent);
  return v - r;
}

fn roundedRectSDF(
  p_in: vec2f,
  center: vec2f,
  width: f32,
  height: f32,
  cornerRadius: f32,
  n: f32,
) -> f32 {
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

fn mainSDF(p1: vec2f, p2: vec2f, p: vec2f) -> f32 {
  let p1n = p1 + p / u.u_resolution.y;
  let p2n = p2 + p / u.u_resolution.y;
  var d1: f32;
  if (u.u_showShape1 == 1) {
    d1 = sdCircle(p1n, 100.0 * u.u_dpr / u.u_resolution.y);
  } else {
    d1 = 1.0;
  }
  var d2: f32;
  if (u.u_shapeRoundness < 0.0) {
    d2 = continuousCapsuleSDF(
      p2n,
      vec2f(0.0),
      u.u_shapeWidth / u.u_resolution.y,
      u.u_shapeHeight / u.u_resolution.y,
    );
  } else {
    d2 = roundedRectSDF(
      p2n,
      vec2f(0.0),
      u.u_shapeWidth / u.u_resolution.y,
      u.u_shapeHeight / u.u_resolution.y,
      u.u_shapeRadius / u.u_resolution.y,
      u.u_shapeRoundness,
    );
  }
  return smin(d1, d2, u.u_mergeRate);
}
