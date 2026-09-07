// Math helpers shared by the reference glass passes.

fn safeAsin(x: f32) -> f32 {
  return asin(clamp(x, -1.0, 1.0));
}
