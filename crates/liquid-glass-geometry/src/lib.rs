//! Renderer-independent continuous-corner geometry.
//!
//! The public geometry implementation is maintained in our dedicated
//! [`squircle-rs`](https://github.com/Erizeez/squircle-rs) repository. This
//! crate keeps the workspace dependency boundary stable while making that
//! tagged implementation the single source of truth for UI corner geometry.

#![deny(unsafe_code)]

pub use squircle_rs::{
    APPLE_CORNER_SMOOTHING, CornerRadii, CornerSegment, CubicBezier, PathCommand, Point,
    ProcessedCorner, SquircleParams, corner_lead_distance, generate_squircle_svg_path,
    glsl_squircle_sdf_source, sd_squircle, squircle_alpha, squircle_border_coverage,
    squircle_path_commands, wgsl_squircle_sdf_source,
};

/// The exponent used by the renderer's compact SDF representation for the
/// upstream library's default Apple smoothing ratio.
///
/// `squircle-rs` exposes the smoothing ratio for path generation and maps it
/// to the same exponent in its shader helper. Keeping this conversion here
/// lets the scene model and the GPU shader share the tagged library default.
pub const DEFAULT_SQUIRCLE_EXPONENT: f32 = 2.0 + (4.2 - 2.0) * APPLE_CORNER_SMOOTHING;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_default_is_the_tagged_apple_smoothing() {
        assert!((APPLE_CORNER_SMOOTHING - 0.6).abs() < f32::EPSILON);
        assert!((DEFAULT_SQUIRCLE_EXPONENT - 3.32).abs() < f32::EPSILON);

        let params = SquircleParams::new(160.0, 96.0, 18.0);
        assert!(!squircle_path_commands(&params).is_empty());
    }
}
