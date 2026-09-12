//! Renderer-independent data model for Liquid Glass.
//!
//! This crate deliberately does not depend on `wgpu` or Iced. It is the
//! contract between the UI adapter and the compositor.

#![deny(unsafe_code)]

use std::fmt;

pub use liquid_glass_geometry::{
    APPLE_CORNER_SMOOTHING, PathCommand, SquircleParams, squircle_path_commands,
};

/// A stable identifier for a glass element.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct GlassId(pub u64);

/// A logical-pixel rectangle.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Rect {
    #[must_use]
    pub const fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self { x, y, width, height }
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.width <= 0.0 || self.height <= 0.0
    }

    #[must_use]
    pub fn expand(self, amount: f32) -> Self {
        Self {
            x: self.x - amount,
            y: self.y - amount,
            width: self.width + amount * 2.0,
            height: self.height + amount * 2.0,
        }
    }
}

/// An sRGB RGBA color in the 0..=1 range.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    #[must_use]
    pub const fn rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    #[must_use]
    pub const fn white() -> Self {
        Self::rgba(1.0, 1.0, 1.0, 1.0)
    }

    #[must_use]
    pub const fn transparent() -> Self {
        Self::rgba(0.0, 0.0, 0.0, 0.0)
    }
}

impl Default for Color {
    fn default() -> Self {
        Self::transparent()
    }
}

/// Public Liquid Glass material variants.
///
/// `Regular` is the more substantial system surface. `Clear` keeps the
/// backdrop more legible and is intended for controls that should sit lightly
/// above content. `TrafficLight` is a deliberately narrow semantic variant
/// for the enhanced macOS-style red/yellow/green window buttons. The physical
/// experiment uses `TrafficLightPhysical`: it selects only a spherical SDF
/// normal while retaining the stock material response.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum GlassVariant {
    #[default]
    Regular,
    Clear,
    TrafficLight,
    TrafficLightPhysical,
}

/// Optical controls specific to the spherical macOS traffic-light material.
///
/// These values are carried through the regular glass material so a demo or a
/// platform adapter can tune the control without changing the shader's global
/// defaults or affecting ordinary glass surfaces.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrafficLightStyle {
    /// Coverage of the coloured substrate away from the lower light.
    pub substrate_coverage: f32,
    /// Coverage of the coloured substrate where the lower light is strongest.
    pub lower_substrate_coverage: f32,
    /// Base tint coverage at the lower-facing part of the sphere.
    pub lower_tint_coverage: f32,
    /// Strength of the normal-derived lower incident light.
    pub angular_light: f32,
    /// Normalized angle from the view axis toward the lower hemisphere.
    /// `0.0` is frontal light and `1.0` is light coming directly from below.
    pub light_angle: f32,
    /// Angular radius of the lower light source. Larger values spread the
    /// incident field over a wider part of the curved body.
    pub light_softness: f32,
    /// Depth-axis radius relative to the visible short axis. `1.0` is a
    /// sphere, values below it flatten the body, and values above it thicken
    /// the ellipsoid.
    pub body_thickness: f32,
    /// Blend from a clear refracted sample to the blurred backdrop sample.
    /// `1.0` preserves the fully scattered default.
    pub internal_scattering: f32,
    /// Absorption strength of the closed spherical edge; the lateral wall
    /// receives the stronger anisotropic response.
    pub side_edge_darkness: f32,
    /// Relative width of the closed absorbing band.
    pub side_edge_width: f32,
    /// Bias from a uniform contour toward a thicker/deeper lateral wall.
    /// Zero is uniform around the circle; one concentrates the response
    /// toward the left and right sides.
    pub edge_side_bias: f32,
    /// Standard deviation in degrees of the Gaussian thickness distribution
    /// around each horizontal pole.
    pub edge_side_angle: f32,
}

impl TrafficLightStyle {
    #[must_use]
    pub const fn physical() -> Self {
        Self {
            substrate_coverage: 0.88,
            lower_substrate_coverage: 0.54,
            lower_tint_coverage: 0.66,
            angular_light: 0.055,
            light_angle: 0.52,
            light_softness: 1.0,
            body_thickness: 0.79,
            internal_scattering: 1.0,
            side_edge_darkness: 4.0,
            side_edge_width: 1.26,
            edge_side_bias: 1.0,
            edge_side_angle: 23.0,
        }
    }
}

impl Default for TrafficLightStyle {
    fn default() -> Self {
        Self::physical()
    }
}

/// Accessibility preferences that affect the optical treatment of glass.
///
/// These are renderer inputs rather than UI-only flags, so every backend can
/// make the same decision about transparency, contrast, and motion.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[allow(clippy::struct_excessive_bools)]
pub struct GlassAccessibility {
    pub reduced_transparency: bool,
    pub increased_contrast: bool,
    pub reduced_motion: bool,
}

impl GlassAccessibility {
    #[must_use]
    pub const fn none() -> Self {
        Self { reduced_transparency: false, increased_contrast: false, reduced_motion: false }
    }
}

/// Low-frequency information about the content behind a glass surface.
///
/// Values are normalized to the `0..=1` range. The renderer may replace this
/// fallback with a more current estimate when it owns the backdrop upload.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlassEnvironment {
    pub ambient_color: Color,
    pub luminance: f32,
    pub contrast: f32,
}

impl GlassEnvironment {
    #[must_use]
    pub const fn neutral() -> Self {
        Self {
            ambient_color: Color::rgba(0.5, 0.5, 0.5, 1.0),
            // Relative luminance is linear-light even though public colors
            // use sRGB components. 0.5 sRGB decodes to roughly 0.214.
            luminance: 0.214_041_14,
            contrast: 0.0,
        }
    }

    /// Estimates ambient color, luminance, and contrast from an RGBA8 frame.
    ///
    /// The estimate intentionally samples a sparse set of pixels. It is used
    /// for slow-changing material adaptation and should not become another
    /// full-frame operation in a scrolling render loop.
    #[must_use]
    pub fn from_rgba8(rgba8: &[u8]) -> Self {
        let mut color_sum = [0.0_f32; 3];
        let mut luminance_sum = 0.0_f32;
        let mut minimum = 1.0_f32;
        let mut maximum = 0.0_f32;
        let mut samples = 0.0_f32;

        for pixel in rgba8.chunks_exact(4).step_by(16) {
            let color = [
                f32::from(pixel[0]) / 255.0,
                f32::from(pixel[1]) / 255.0,
                f32::from(pixel[2]) / 255.0,
            ];
            let linear = color.map(srgb_channel_to_linear);
            let luminance = linear[0] * 0.2126 + linear[1] * 0.7152 + linear[2] * 0.0722;
            color_sum[0] += color[0];
            color_sum[1] += color[1];
            color_sum[2] += color[2];
            luminance_sum += luminance;
            minimum = minimum.min(luminance);
            maximum = maximum.max(luminance);
            samples += 1.0;
        }

        if samples <= f32::EPSILON {
            return Self::neutral();
        }

        Self {
            ambient_color: Color::rgba(
                color_sum[0] / samples,
                color_sum[1] / samples,
                color_sum[2] / samples,
                1.0,
            ),
            luminance: (luminance_sum / samples).clamp(0.0, 1.0),
            contrast: (maximum - minimum).clamp(0.0, 1.0),
        }
    }
}

impl Default for GlassEnvironment {
    fn default() -> Self {
        Self::neutral()
    }
}

fn srgb_channel_to_linear(value: f32) -> f32 {
    let value = value.clamp(0.0, 1.0);
    if value <= 0.04045 { value / 12.92 } else { ((value + 0.055) / 1.055).powf(2.4) }
}

/// Renderer-wide inputs shared by all nodes in one composition.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GlassRenderOptions {
    pub accessibility: GlassAccessibility,
    pub environment: GlassEnvironment,
}

/// Pointer and state information used by a glass node's internal lighting.
///
/// `pointer` is local to the node's logical bounds. The pointer does not move
/// the SDF; it only controls the interaction light that is rendered inside
/// the surface. This keeps content and geometry stable while the surface
/// reacts to hover, press, or focus.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlassInteraction {
    pub pointer: [f32; 2],
    /// A caller-advanced spring position. It defaults to `pointer`, allowing
    /// applications to use `liquid-glass-animation` without shader changes.
    pub spring: [f32; 2],
    /// Amount of low-frequency parallax used by ambient spill, in `0..=1`.
    pub parallax: f32,
    pub hover: f32,
    pub press: f32,
    pub focus: f32,
}

impl GlassInteraction {
    #[must_use]
    pub const fn inactive() -> Self {
        Self {
            pointer: [0.5, 0.5],
            spring: [0.5, 0.5],
            parallax: 0.0,
            hover: 0.0,
            press: 0.0,
            focus: 0.0,
        }
    }

    #[must_use]
    pub fn normalized(pointer: [f32; 2], hover: f32, press: f32, focus: f32) -> Self {
        Self {
            pointer: [pointer[0].clamp(0.0, 1.0), pointer[1].clamp(0.0, 1.0)],
            spring: [pointer[0].clamp(0.0, 1.0), pointer[1].clamp(0.0, 1.0)],
            parallax: 0.0,
            hover: hover.clamp(0.0, 1.0),
            press: press.clamp(0.0, 1.0),
            focus: focus.clamp(0.0, 1.0),
        }
    }

    #[must_use]
    pub fn with_spring(mut self, spring: [f32; 2], parallax: f32) -> Self {
        self.spring = [spring[0].clamp(0.0, 1.0), spring[1].clamp(0.0, 1.0)];
        self.parallax = parallax.clamp(0.0, 1.0);
        self
    }

    #[must_use]
    pub const fn strength(self) -> f32 {
        self.hover.max(self.press).max(self.focus)
    }
}

impl Default for GlassInteraction {
    fn default() -> Self {
        Self::inactive()
    }
}

/// Per-material gain for effects that adapt to the backdrop and surface size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdaptiveStyle {
    pub tint: f32,
    pub ambient: f32,
    pub shadow: f32,
    pub size: f32,
}

impl AdaptiveStyle {
    #[must_use]
    pub const fn system() -> Self {
        Self { tint: 1.0, ambient: 1.0, shadow: 1.0, size: 1.0 }
    }
}

impl Default for AdaptiveStyle {
    fn default() -> Self {
        Self::system()
    }
}

/// The curvature model used by radius-based rounded rectangles.
/// Circles and ellipses preserve their exact circular geometry. The default
/// continuous curve is derived from `squircle-rs`'s Apple smoothing ratio.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CornerCurve {
    /// A quarter-circle corner with a curvature discontinuity at the join.
    Circular,
    /// A continuous squircle corner represented by the upstream SDF exponent.
    Continuous { exponent: f32 },
}

impl CornerCurve {
    /// The SDF exponent corresponding to `squircle-rs`'s Apple smoothing
    /// default (`APPLE_CORNER_SMOOTHING == 0.6`).
    pub const DEFAULT_CONTINUOUS_EXPONENT: f32 = liquid_glass_geometry::DEFAULT_SQUIRCLE_EXPONENT;

    #[must_use]
    pub const fn continuous() -> Self {
        Self::Continuous { exponent: Self::DEFAULT_CONTINUOUS_EXPONENT }
    }

    #[must_use]
    pub const fn continuous_with_exponent(exponent: f32) -> Self {
        Self::Continuous { exponent }
    }

    #[must_use]
    pub const fn exponent(self) -> f32 {
        match self {
            Self::Circular => 2.0,
            Self::Continuous { exponent } => {
                if exponent >= 2.0 {
                    exponent
                } else {
                    2.0
                }
            }
        }
    }
}

impl Default for CornerCurve {
    fn default() -> Self {
        Self::continuous()
    }
}

/// A shape evaluated by the SDF shader.
#[derive(Clone, Debug, PartialEq)]
pub enum GlassShape {
    RoundedRect { radius: f32 },
    Superellipse { exponent: f32 },
    Capsule,
    Circle,
    Ellipse,
}

/// A secondary shape fused into one glass node.
///
/// Keeping the shapes in one node is the compositor equivalent of Apple's
/// glass effect container: the backdrop is sampled once and the union is
/// evaluated before the optical response, avoiding a glass-on-glass seam.
#[derive(Clone, Debug, PartialEq)]
pub struct GlassShapeLayer {
    pub bounds: Rect,
    pub shape: GlassShape,
    pub corner_curve: CornerCurve,
}

impl GlassShapeLayer {
    #[must_use]
    pub fn new(bounds: Rect, shape: GlassShape) -> Self {
        Self { bounds, shape, corner_curve: CornerCurve::default() }
    }

    #[must_use]
    pub const fn corner_curve(mut self, corner_curve: CornerCurve) -> Self {
        self.corner_curve = corner_curve;
        self
    }
}

impl Default for GlassShape {
    fn default() -> Self {
        Self::RoundedRect { radius: 16.0 }
    }
}

/// Blur configuration used by a glass material.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlurStyle {
    pub radius: f32,
    pub edge_blur: bool,
}

/// Refraction configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RefractionStyle {
    pub thickness: f32,
    pub index: f32,
    pub strength: f32,
}

/// Chromatic dispersion configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DispersionStyle {
    pub strength: f32,
    pub spread: f32,
}

/// Fresnel highlight configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FresnelStyle {
    pub range: f32,
    pub hardness: f32,
    pub strength: f32,
}

/// Directional edge light used by the stable, system-style glass highlight.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlareStyle {
    pub range: f32,
    pub hardness: f32,
    pub convergence: f32,
    pub opposite_factor: f32,
    pub factor: f32,
}

impl GlareStyle {
    #[must_use]
    pub const fn system() -> Self {
        Self { range: 30.0, hardness: 0.2, convergence: 0.5, opposite_factor: 0.8, factor: 0.45 }
    }
}

/// Soft cast shadow produced by a glass surface.
///
/// The shadow is intentionally separate from tint and opacity: a transparent
/// material can still sit above the backdrop and establish its elevation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadowStyle {
    /// Width of the exponential falloff in logical pixels.
    pub expand: f32,
    /// Shadow strength, normally in the `0..=0.6` range.
    pub factor: f32,
    /// Logical-pixel offset of the shadow, usually a small downward shift.
    pub offset: [f32; 2],
}

impl ShadowStyle {
    /// No cast shadow.
    #[must_use]
    pub const fn none() -> Self {
        Self { expand: 25.0, factor: 0.0, offset: [0.0, 0.0] }
    }

    /// The restrained elevation cue used by regular glass surfaces.
    #[must_use]
    pub const fn subtle() -> Self {
        Self { expand: 25.0, factor: 0.15, offset: [0.0, 2.0] }
    }

    /// A slightly stronger cue for floating interactive glass.
    #[must_use]
    pub const fn elevated() -> Self {
        Self { expand: 28.0, factor: 0.30, offset: [0.0, 3.0] }
    }

    /// The compact, higher-contrast elevation cue used by system-like fields.
    #[must_use]
    pub const fn control() -> Self {
        // Keep the unshifted contact edge visible while avoiding a second,
        // downward-biased dark contour beneath compact controls.
        Self { expand: 25.0, factor: 0.24, offset: [0.0, 1.0] }
    }
}

/// The centre light of a spherical glass body.
///
/// The sphere is lit from inside: a uniform incident field across the body,
/// plus a smaller release toward the optically thinner lower hemisphere. These
/// are the strengths a host tunes to judge how lit the centre of a control
/// reads against its rim.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoreLight {
    /// Strength of the uniform incident field across the body. This is the
    /// dominant term for how bright the centre reads.
    pub uniform_light: f32,
    /// Additional release of the same field toward the thinner lower
    /// hemisphere.
    pub thin_light_gain: f32,
}

impl CoreLight {
    /// The calibrated default.
    #[must_use]
    pub const fn new() -> Self {
        Self { uniform_light: 0.045, thin_light_gain: 0.055 }
    }

    /// Clamps both strengths into range.
    #[must_use]
    pub fn clamped(self) -> Self {
        Self {
            uniform_light: self.uniform_light.clamp(0.0, 1.0),
            thin_light_gain: self.thin_light_gain.clamp(0.0, 1.0),
        }
    }
}

impl Default for CoreLight {
    fn default() -> Self {
        Self::new()
    }
}

/// How a glass control's *material* responds to pointer engagement.
///
/// Engagement is a whole-control brightening, not a highlight: the composed
/// colour rises uniformly, with no pointer falloff, which is what makes hover
/// and press read as "the control lit up" instead of "a highlight moved in".
///
/// These are material parameters rather than runtime interaction values, so a
/// playground can tune the response live through the material.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InteractionResponse {
    /// Spatially uniform multiplicative gain applied while the pointer is over
    /// the control.
    pub hover_gain: f32,
    /// Spatially uniform multiplicative gain applied while the control is held.
    pub press_gain: f32,
    /// Spatially uniform, tint-hued brightness lift added while held. Preserves
    /// hue where a channel is already clipped at the SDR ceiling, which a
    /// multiplier alone cannot raise.
    pub press_lift: f32,
}

impl InteractionResponse {
    /// The calibrated default: a hover lift, a smaller press gain, and a
    /// smaller still tint-hued press lift.
    #[must_use]
    pub const fn new() -> Self {
        Self { hover_gain: 0.22, press_gain: 0.12, press_lift: 0.06 }
    }

    /// Disables the response.
    #[must_use]
    pub const fn none() -> Self {
        Self { hover_gain: 0.0, press_gain: 0.0, press_lift: 0.0 }
    }

    /// Clamps every gain into range.
    #[must_use]
    pub fn clamped(self) -> Self {
        Self {
            hover_gain: self.hover_gain.clamp(0.0, 1.0),
            press_gain: self.press_gain.clamp(0.0, 1.0),
            press_lift: self.press_lift.clamp(0.0, 1.0),
        }
    }
}

impl Default for InteractionResponse {
    fn default() -> Self {
        Self::new()
    }
}

/// A complete, shape-independent glass material.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlassMaterial {
    /// Public system variant used by the shader's adaptive branch.
    pub variant: GlassVariant,
    /// Optional semantic controls for the spherical traffic-light variant.
    pub traffic_light: TrafficLightStyle,
    pub blur: BlurStyle,
    pub tint: Color,
    /// Neutral white mixed into the tint independently of its hue.
    pub whiteness: f32,
    pub refraction: RefractionStyle,
    pub dispersion: DispersionStyle,
    pub fresnel: FresnelStyle,
    pub glare: GlareStyle,
    pub opacity: f32,
    /// Smooth-min blend radius used by the reference two-shape composition.
    ///
    /// When [`Self::show_shape1`] is enabled, the compositor combines a
    /// 100px circle with the node's shape before evaluating the optical
    /// effects. A zero value disables the visible merge band.
    pub merge_rate: f32,
    /// Whether to include the reference circle in the SDF composition.
    pub show_shape1: bool,
    /// Per-surface cast shadow. `clear()` deliberately disables it for
    /// compositor overlays; regular and interactive presets enable it.
    pub shadow: ShadowStyle,
    /// Gains for environment, ambient spill, shadow adaptation, and size
    /// response. They are kept on the material so custom controls can tune
    /// the response without replacing the renderer.
    pub adaptive: AdaptiveStyle,
    /// Whole-control brightening applied while the pointer engages the control.
    pub interaction: InteractionResponse,
    /// Centre light of a spherical body (traffic-light variants).
    pub core_light: CoreLight,
}

impl GlassMaterial {
    #[must_use]
    pub const fn variant(mut self, variant: GlassVariant) -> Self {
        self.variant = variant;
        self
    }

    #[must_use]
    pub const fn clear() -> Self {
        Self {
            variant: GlassVariant::Clear,
            traffic_light: TrafficLightStyle::physical(),
            blur: BlurStyle { radius: 10.0, edge_blur: true },
            tint: Color::rgba(1.0, 1.0, 1.0, 0.12),
            whiteness: 0.0,
            refraction: RefractionStyle { thickness: 0.18, index: 1.45, strength: 0.35 },
            dispersion: DispersionStyle { strength: 0.08, spread: 0.02 },
            fresnel: FresnelStyle { range: 0.75, hardness: 0.6, strength: 0.35 },
            glare: GlareStyle::system(),
            opacity: 0.92,
            merge_rate: 0.05,
            show_shape1: false,
            shadow: ShadowStyle::none(),
            adaptive: AdaptiveStyle::system(),
            interaction: InteractionResponse::new(),
            core_light: CoreLight::new(),
        }
    }

    #[must_use]
    pub const fn regular() -> Self {
        Self {
            variant: GlassVariant::Regular,
            blur: BlurStyle { radius: 20.0, edge_blur: true },
            shadow: ShadowStyle::subtle(),
            ..Self::clear()
        }
    }

    #[must_use]
    pub const fn thick() -> Self {
        Self {
            blur: BlurStyle { radius: 34.0, edge_blur: true },
            shadow: ShadowStyle::subtle(),
            ..Self::regular()
        }
    }

    #[must_use]
    pub const fn interactive() -> Self {
        Self {
            refraction: RefractionStyle { strength: 0.5, ..Self::regular().refraction },
            shadow: ShadowStyle::elevated(),
            ..Self::regular()
        }
    }
}

impl Default for GlassMaterial {
    fn default() -> Self {
        Self::regular()
    }
}

/// The portion of the already-rendered scene needed by a glass node.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BackdropRegion {
    pub bounds: Rect,
    pub padding: f32,
    pub blur_radius: f32,
}

impl BackdropRegion {
    #[must_use]
    pub fn capture_bounds(self) -> Rect {
        self.bounds.expand(self.padding + self.blur_radius)
    }
}

/// A renderer-facing glass node.
#[derive(Clone, Debug, PartialEq)]
pub struct GlassNode {
    pub id: GlassId,
    pub bounds: Rect,
    pub shape: GlassShape,
    pub corner_curve: CornerCurve,
    pub material: GlassMaterial,
    pub backdrop: BackdropRegion,
    pub z_index: i32,
    pub interaction: GlassInteraction,
    pub fused_shapes: Vec<GlassShapeLayer>,
}

impl GlassNode {
    #[must_use]
    pub fn new(id: GlassId, bounds: Rect) -> Self {
        let material = GlassMaterial::default();
        Self {
            id,
            bounds,
            shape: GlassShape::default(),
            corner_curve: CornerCurve::default(),
            backdrop: BackdropRegion { bounds, padding: 8.0, blur_radius: material.blur.radius },
            material,
            z_index: 0,
            interaction: GlassInteraction::inactive(),
            fused_shapes: Vec::new(),
        }
    }

    #[must_use]
    pub fn shape(mut self, shape: GlassShape) -> Self {
        self.shape = shape;
        self
    }

    #[must_use]
    pub const fn corner_curve(mut self, corner_curve: CornerCurve) -> Self {
        self.corner_curve = corner_curve;
        self
    }

    #[must_use]
    pub fn material(mut self, material: GlassMaterial) -> Self {
        self.backdrop.blur_radius = material.blur.radius;
        self.material = material;
        self
    }

    #[must_use]
    pub const fn interaction(mut self, interaction: GlassInteraction) -> Self {
        self.interaction = interaction;
        self
    }

    /// Adds one secondary shape to this node's pre-optical SDF union.
    #[must_use]
    pub fn fuse_shape(mut self, shape: GlassShapeLayer) -> Self {
        self.backdrop.bounds = union(self.backdrop.bounds, union(self.bounds, shape.bounds));
        self.fused_shapes.push(shape);
        self
    }

    /// Returns the union of the primary and fused geometry bounds.
    #[must_use]
    pub fn visual_bounds(&self) -> Rect {
        self.fused_shapes.iter().fold(self.bounds, |bounds, shape| union(bounds, shape.bounds))
    }

    /// Fuses the other node's geometry into this node while keeping this
    /// node's material and z-order. The caller should render the returned
    /// node instead of the two original nodes.
    #[must_use]
    pub fn fuse_with(self, other: &Self) -> Self {
        self.fuse_shape(
            GlassShapeLayer::new(other.bounds, other.shape.clone())
                .corner_curve(other.corner_curve),
        )
    }
}

/// A reusable container for fusing several glass shapes into one optical
/// surface.
///
/// The reference GPU path evaluates the first four secondary shapes in one
/// uniform block. A caller can keep adding shapes to the data model, while a
/// backend may expose a larger storage-buffer implementation later without
/// changing the public composition model.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GlassEffectContainer {
    node: Option<GlassNode>,
}

impl GlassEffectContainer {
    #[must_use]
    pub fn new(node: GlassNode) -> Self {
        Self { node: Some(node) }
    }

    #[must_use]
    pub fn fuse_shape(mut self, shape: GlassShapeLayer) -> Self {
        if let Some(node) = self.node.take() {
            self.node = Some(node.fuse_shape(shape));
        }
        self
    }

    #[must_use]
    pub fn fuse_node(mut self, other: &GlassNode) -> Self {
        if let Some(node) = self.node.take() {
            self.node = Some(node.fuse_with(other));
        }
        self
    }

    #[must_use]
    pub const fn node(&self) -> Option<&GlassNode> {
        self.node.as_ref()
    }

    #[must_use]
    pub fn into_node(self) -> Option<GlassNode> {
        self.node
    }
}

/// A scene collection consumed by the Liquid Glass compositor.
#[derive(Clone, Debug, Default)]
pub struct GlassScene {
    nodes: Vec<GlassNode>,
    render_options: Option<GlassRenderOptions>,
}

impl GlassScene {
    pub fn push(&mut self, node: GlassNode) {
        self.nodes.push(node);
    }

    #[must_use]
    pub fn nodes(&self) -> &[GlassNode] {
        &self.nodes
    }

    /// Returns mutable nodes for compositor adapters that apply a final
    /// viewport transform before rendering.
    pub fn nodes_mut(&mut self) -> &mut [GlassNode] {
        &mut self.nodes
    }

    /// Sets render options for this scene only.
    pub fn set_render_options(&mut self, options: GlassRenderOptions) {
        self.render_options = Some(options);
    }

    /// Returns scene-local render options, if the scene has an override.
    #[must_use]
    pub const fn render_options(&self) -> Option<GlassRenderOptions> {
        self.render_options
    }

    /// Returns an owned scene with render options attached.
    #[must_use]
    pub fn with_render_options(mut self, options: GlassRenderOptions) -> Self {
        self.set_render_options(options);
        self
    }

    /// Returns nodes in stable back-to-front order for compositor drawing.
    #[must_use]
    pub fn nodes_in_render_order(&self) -> Vec<&GlassNode> {
        let mut nodes: Vec<_> = self.nodes.iter().collect();
        nodes.sort_by_key(|node| node.z_index);
        nodes
    }

    #[must_use]
    pub fn capture_bounds(&self) -> Option<Rect> {
        self.nodes.iter().map(|node| node.backdrop.capture_bounds()).reduce(union)
    }
}

fn union(a: Rect, b: Rect) -> Rect {
    let left = a.x.min(b.x);
    let top = a.y.min(b.y);
    let right = (a.x + a.width).max(b.x + b.width);
    let bottom = (a.y + a.height).max(b.y + b.height);
    Rect::new(left, top, right - left, bottom - top)
}

impl fmt::Display for GlassId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "glass-{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_capture_bounds_include_backdrop_padding_and_blur() {
        let node = GlassNode::new(GlassId(1), Rect::new(10.0, 20.0, 100.0, 50.0))
            .material(GlassMaterial::regular());
        let expected = Rect::new(-18.0, -8.0, 156.0, 106.0);

        let mut scene = GlassScene::default();
        scene.push(node);

        assert_eq!(scene.capture_bounds(), Some(expected));
    }

    #[test]
    fn material_change_updates_backdrop_blur_radius() {
        let node = GlassNode::new(GlassId(1), Rect::new(0.0, 0.0, 10.0, 10.0))
            .material(GlassMaterial::thick());

        assert!(
            (node.backdrop.blur_radius - GlassMaterial::thick().blur.radius).abs() < f32::EPSILON
        );
    }

    #[test]
    fn render_order_is_sorted_by_z_index() {
        let mut front = GlassNode::new(GlassId(2), Rect::new(20.0, 20.0, 10.0, 10.0));
        front.z_index = 10;
        let mut back = GlassNode::new(GlassId(1), Rect::new(0.0, 0.0, 10.0, 10.0));
        back.z_index = -1;

        let mut scene = GlassScene::default();
        scene.push(front);
        scene.push(back);

        let order = scene.nodes_in_render_order();

        assert_eq!(order.iter().map(|node| node.id).collect::<Vec<_>>(), [GlassId(1), GlassId(2)]);
    }

    #[test]
    fn glass_nodes_default_to_continuous_corners() {
        let node = GlassNode::new(GlassId(3), Rect::new(0.0, 0.0, 72.0, 36.0));

        assert_eq!(node.corner_curve, CornerCurve::continuous());
        assert!(
            (node.corner_curve.exponent() - CornerCurve::DEFAULT_CONTINUOUS_EXPONENT).abs()
                < f32::EPSILON
        );
    }

    #[test]
    fn circular_and_invalid_curves_resolve_safely() {
        assert!((CornerCurve::Circular.exponent() - 2.0).abs() < f32::EPSILON);
        assert!(
            (CornerCurve::continuous_with_exponent(f32::NAN).exponent() - 2.0).abs() < f32::EPSILON
        );
    }

    #[test]
    fn material_variants_are_explicit() {
        assert_eq!(GlassMaterial::regular().variant, GlassVariant::Regular);
        assert_eq!(GlassMaterial::clear().variant, GlassVariant::Clear);
        assert_eq!(GlassMaterial::default().variant, GlassVariant::Regular);
    }

    #[test]
    fn environment_estimate_is_bounded_and_scene_override_is_preserved() {
        let environment = GlassEnvironment::from_rgba8(&[
            255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 128, 128, 128, 255,
        ]);

        assert!((0.0..=1.0).contains(&environment.luminance));
        assert!((0.0..=1.0).contains(&environment.contrast));
        assert!((0.0..=1.0).contains(&environment.ambient_color.r));

        let options = GlassRenderOptions {
            accessibility: GlassAccessibility {
                increased_contrast: true,
                ..GlassAccessibility::none()
            },
            environment,
        };
        let scene = GlassScene::default().with_render_options(options);

        assert_eq!(scene.render_options(), Some(options));
    }

    #[test]
    fn environment_luminance_is_estimated_in_linear_light() {
        let environment = GlassEnvironment::from_rgba8(&[128, 128, 128, 255]);

        assert!((environment.luminance - 0.215_860_53).abs() < 0.000_01);
        assert!(environment.contrast.abs() < f32::EPSILON);
        // Public colors remain encoded sRGB values for API consistency.
        assert!((environment.ambient_color.r - 128.0 / 255.0).abs() < f32::EPSILON);
    }

    #[test]
    fn interaction_is_normalized_without_moving_geometry() {
        let interaction = GlassInteraction::normalized([1.4, -0.2], 1.2, -0.1, 0.3);
        assert_eq!(interaction.pointer, [1.0, 0.0]);
        assert_eq!(interaction.hover, 1.0);
        assert_eq!(interaction.press, 0.0);
        assert_eq!(interaction.focus, 0.3);
    }

    #[test]
    fn fused_shape_keeps_one_material_and_backdrop_node() {
        let base = GlassNode::new(GlassId(10), Rect::new(20.0, 20.0, 80.0, 32.0));
        let other = GlassNode::new(GlassId(11), Rect::new(88.0, 20.0, 80.0, 32.0))
            .shape(GlassShape::Capsule);
        let fused = base.clone().fuse_with(&other);

        assert_eq!(fused.id, base.id);
        assert_eq!(fused.material, base.material);
        assert_eq!(fused.fused_shapes.first().map(|shape| shape.bounds), Some(other.bounds));
        assert_eq!(fused.visual_bounds(), Rect::new(20.0, 20.0, 148.0, 32.0));
    }
}
