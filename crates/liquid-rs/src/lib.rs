//! # liquid-rs
//!
//! Framework-agnostic real-time Liquid Glass rendering engine and GPU shader compositor.
//!
//! Provides renderer-independent continuous-corner geometry, physical scene graph,
//! and WGPU multi-pass compositor for Liquid Glass materials without framework or window bindings.

pub use liquid_glass_animation as animation;
pub use liquid_glass_geometry as geometry;
pub use liquid_glass_render as render;
pub use liquid_glass_scene as scene;
