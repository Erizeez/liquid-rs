# liquid-rs

[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org)

Framework-agnostic real-time Liquid Glass rendering engine and GPU shader compositor for Rust.

`liquid-rs` is completely decoupled from any UI frameworks (such as Iced or Slint) and platform windowing layers (such as Winit). It operates purely at the math, geometry, scene description, and WGPU rendering pipeline levels.

---

## Features

- **Apple Squircle & Continuous Curvature**: First-class support for Apple-style squircle corners (G2 continuous curvature, super-ellipse) and analytical SDF computation.
- **Physical Optical Glass Pipeline**: Realistic multi-pass rendering featuring surface refraction, spectral dispersion, fresnel rim lighting, adaptive ambient specular glares, and real-time backdrop sampling.
- **Dual Kawase Backdrop Blur**: High-performance downsampled dual-pass Kawase blur integration powered by [`vibrancy-rs`](https://github.com/Erizeez/vibrancy-rs).
- **Scene Graph & Automatic Shape Fusion**: Declarative scene graph supporting nested glass nodes, depth z-ordering, unified backdrop capture bounds, and zero-flicker adjacent shape fusion.
- **Physics-Based Spring Animations**: Built-in spring dynamics engine for fluid interactive surface reactions.

---

## Architecture

`liquid-rs` is organized as a workspace of focused, modular crates:

```text
liquid-rs (Facade)
├── liquid-glass-animation  # Spring dynamics and physics transitions
├── liquid-glass-geometry   # Continuous-corner squircles, capsules, SDFs
├── liquid-glass-scene      # Declarative scene graph, material optics, node hierarchy
└── liquid-glass-render     # WGPU 4-pass compositor, texture pooling, Kawase blur
```

- **[`liquid-glass-geometry`](crates/liquid-glass-geometry)**: Continuous-corner (Apple squircle) and capsule geometry primitives with analytical Signed Distance Fields (SDF).
- **[`liquid-glass-scene`](crates/liquid-glass-scene)**: Renderer-independent scene graph, physical material parameters (`specular`, `translucency`, `refraction`, `dispersion`, `glare`), and layer composition models.
- **[`liquid-glass-render`](crates/liquid-glass-render)**: `wgpu`-based multi-pass compositor integrating Dual Kawase backdrop blur from `vibrancy-rs` and real-time liquid glass surface WGSL shaders.
- **[`liquid-glass-animation`](crates/liquid-glass-animation)**: Spring dynamics for fluid transitions and physics.

---

## Usage

Add `liquid-rs` to your `Cargo.toml`:

```toml
[dependencies]
liquid-rs = { git = "https://github.com/Erizeez/liquid-rs", tag = "v0.1.0" }
```

Or depend on individual lightweight crates according to your needs:

```toml
[dependencies]
liquid-glass-geometry = { git = "https://github.com/Erizeez/liquid-rs", tag = "v0.1.0" }
liquid-glass-scene = { git = "https://github.com/Erizeez/liquid-rs", tag = "v0.1.0" }
liquid-glass-render = { git = "https://github.com/Erizeez/liquid-rs", tag = "v0.1.0" }
```

### Quick Example

```rust
use liquid_rs::{
    scene::{Color, GlassMaterial, GlassNode, GlassScene, GlassShape, Rect},
    render::{GpuRenderer, GpuSize, LiquidRenderer},
};

// 1. Build a declarative glass scene
let mut scene = GlassScene::new();
let material = GlassMaterial::regular();

let node = GlassNode::new(
    Rect::new(20.0, 20.0, 320.0, 64.0),
    material,
)
.with_shape(GlassShape::Capsule);

scene.add_node(node);

// 2. The scene can then be rendered with GpuRenderer over any WGPU render target
```

---

## Upstream Dependencies

- [`squircle-rs`](https://github.com/Erizeez/squircle-rs): Apple-style squircle path and curvature calculation.
- [`vibrancy-rs`](https://github.com/Erizeez/vibrancy-rs): Real-time GPU Dual Kawase blur passes.

---

## Acknowledgements

- Core WGSL shaders and signed distance field formulation adapted from [liquid-glass-studio](https://github.com/charlesyin/liquid-glass-studio) by Charles Yin (licensed under MIT).

---

## License

Dual-licensed under either of:

- [MIT License](LICENSE-MIT)
- [Apache License, Version 2.0](LICENSE-APACHE)

at your option.
