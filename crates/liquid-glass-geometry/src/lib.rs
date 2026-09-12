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

/// Balanced smoothing factor for nested glass plates and Dock containers (0.20).
///
/// Preserves a dominant 80% true circular arc body (72° span) for crisp, distinct
/// circular dome recognition, with subtle 20% tangent ease-in at the ends to eliminate seam creases.
pub const APPLE_DOCK_SMOOTHING: f32 = 0.20;

/// Concentric nested corner geometry calculations.
///
/// Ensures that when rounded rectangles are nested (e.g. icons inside a dock, cards inside a window),
/// their centers of curvature coincide in 2D space: `C_outer == C_inner`.
pub mod concentric {
    /// Calculates the concentric outer corner radius: `R_outer = padding + r_inner`.
    #[inline]
    #[must_use]
    pub const fn outer_radius(inner_radius: f32, padding: f32) -> f32 {
        inner_radius + padding
    }

    /// Calculates the concentric inner corner radius: `r_inner = max(0.0, R_outer - padding)`.
    #[inline]
    #[must_use]
    pub fn inner_radius(outer_radius: f32, padding: f32) -> f32 {
        (outer_radius - padding).max(0.0)
    }

    /// Computes the radial clearance between nested concentric corners at any angle theta (strictly constant = padding).
    #[inline]
    #[must_use]
    pub const fn radial_gap(padding: f32) -> f32 {
        padding
    }
}

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

    #[test]
    #[allow(clippy::float_cmp)]
    fn concentric_curvature_centers_invariant() {
        let r_inner = 10.0;
        let padding = 13.0;
        let r_outer = concentric::outer_radius(r_inner, padding);
        assert_eq!(r_outer, 23.0);
        assert_eq!(concentric::inner_radius(r_outer, padding), 10.0);
        assert_eq!(concentric::radial_gap(padding), 13.0);
    }
}
