// Shared WGSL helpers. Backend bindings will be added with the first wgpu pass.

fn saturate(value: f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

