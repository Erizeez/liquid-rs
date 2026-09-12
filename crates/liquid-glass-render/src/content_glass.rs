//! Fitted content-glass material (macOS 27).
//!
//! This is the clean, measured content-glass path: an affine body grade over a
//! mip-blurred backdrop, two-sided rim refraction, chromatic aberration and an
//! anisotropic sub-pixel contact contour. It deliberately contains no painted
//! bevel, rim glint, edge bleed, key fill or press light — those belong to
//! control chrome and are what made the earlier material read as plastic.
//!
//! The body matrices and edge factors were fitted against real macOS 27
//! `NSGlassEffectView` captures on build 26A5421a (interior MAE 1.5–3.3/255).
//!
//! The whole path works in sRGB-encoded space because the matrices were fitted
//! there: the backdrop is an `Rgba8Unorm` texture (raw sRGB bytes), mips are
//! generated in that same space, and the fragment shader decodes to linear
//! only on the way out when the target is an sRGB format.

use bytemuck::{Pod, Zeroable};

use crate::{GpuError, GpuSize};

/// std140/WGSL size of one [`ContentGlassUniform`].
pub const CONTENT_GLASS_UNIFORM_BYTES: u32 = 464;

/// Maximum number of content-glass elements drawn in one call.
pub const MAX_CONTENT_GLASS_NODES: usize = 64;

/// Appearance blend for the content material: 0 = light, 1 = dark.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContentGlassAppearance(pub f32);

impl ContentGlassAppearance {
    pub const LIGHT: Self = Self(0.0);
    pub const DARK: Self = Self(1.0);
}

/// One content-glass element in canvas pixels (top-left origin).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContentGlassNode {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    /// Corner radius in pixels. The measured macOS 27 glass corner is a
    /// circular arc of this radius; `min(width, height) / 2` gives a capsule.
    pub corner_radius: f32,
    pub appearance: ContentGlassAppearance,
    /// 0 = absent, 1 = fully present. Drives the appear/disappear ramp.
    pub diffusion: f32,
}

impl ContentGlassNode {
    #[must_use]
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
            corner_radius: 16.0,
            appearance: ContentGlassAppearance::LIGHT,
            diffusion: 1.0,
        }
    }

    #[must_use]
    pub const fn corner_radius(mut self, corner_radius: f32) -> Self {
        self.corner_radius = corner_radius;
        self
    }

    /// Rounds the short axis fully, producing a capsule.
    #[must_use]
    pub fn capsule(mut self) -> Self {
        self.corner_radius = self.width.min(self.height) * 0.5;
        self
    }

    #[must_use]
    pub const fn appearance(mut self, appearance: ContentGlassAppearance) -> Self {
        self.appearance = appearance;
        self
    }

    #[must_use]
    pub const fn diffusion(mut self, diffusion: f32) -> Self {
        self.diffusion = diffusion;
        self
    }
}

/// Fitted material constants. Defaults are the measured macOS 27 values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContentGlassMaterial {
    /// (`inner amount`, `inner inv_height`, `outer amount`, `outer inv_height`)
    pub inner_refract: [f32; 4],
    /// (refract opacity, threshold0, threshold1, unused)
    pub refract_params: [f32; 4],
    pub refract_angle: [f32; 2],
    pub displacement: [f32; 4],
    /// (`aberration amount`, `inv_height`, `offset`, unused)
    pub aberration: [f32; 4],
    pub aberration_angle: [f32; 2],
    /// Blur radius in **device pixels**, applied uniformly. The measured
    /// macOS 27 material uses a fixed radius (best-fit 65px at 2x) rather than
    /// one that shrinks with the element. Halve it for a 1x target.
    pub blur_radius: f32,
    /// Reference half-extent (device pixels) for the refraction lobes, which
    /// do scale with the element ("larger glass = thicker glass").
    pub scale_ref: f32,
    pub edge_width: f32,
    pub light_face_cm: [[f32; 4]; 3],
    pub dark_face_cm: [[f32; 4]; 3],
    /// (left/right, top/bottom) contact-contour opacity per appearance.
    pub light_edge_opacity: [f32; 2],
    pub dark_edge_opacity: [f32; 2],
    /// Quadratic colour correction, one `[f32; 3]` per term in the order
    /// `(r2, g2, b2, rg, rb, gb)`. The real material is not a pure affine map
    /// of the blurred backdrop; this term cuts the body error from ~2.5/255
    /// to ~1.9/255.
    pub light_face_quad: [[f32; 3]; 6],
    pub dark_face_quad: [[f32; 3]; 6],
}

/// Traffic light button kind / role for macOS window controls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrafficLightKind {
    /// Red window close button.
    Close,
    /// Yellow window minimize button.
    Minimize,
    /// Green window zoom / fullscreen button.
    Zoom,
}

impl ContentGlassMaterial {
    /// The fitted macOS 27 `clear` *style* (`NSGlassEffectViewStyleClear`):
    /// much more transparent than the regular styles, and more blurred
    /// (measured radius 80px at 2x). This is where the rim lensing is easiest
    /// to see.
    #[must_use]
    pub fn clear() -> Self {
        Self {
            blur_radius: 80.0,
            light_face_cm: [
                [0.9111, -0.1699, 0.0368, 0.2295],
                [-0.0928, 0.7454, -0.0217, 0.3014],
                [-0.0519, -0.1432, 0.9273, 0.2433],
            ],
            dark_face_cm: [
                [0.8638, -0.0236, 0.0096, 0.0425],
                [-0.0466, 0.7531, -0.0254, 0.1144],
                [-0.0443, -0.0713, 0.9331, 0.0473],
            ],
            light_edge_opacity: [0.595, 0.190],
            dark_edge_opacity: [0.526, 0.0],
            ..Self::transparent()
        }
    }

    /// The most-transparent regular appearance (macOS Liquid Glass set to its
    /// most transparent, `NSGlassTintAmount = 0`): lower milkiness
    /// (bias ~0.40) and a 45px blur at 2x. One boundary of the appearance
    /// slider.
    #[must_use]
    pub fn transparent() -> Self {
        let wx = 28.0_f32;
        let hy = 32.0_f32;
        let base_h = (wx * hy).sqrt();
        Self {
            inner_refract: [-56.0, 1.0 / base_h, 1.02, 0.16],
            refract_params: [0.0, 0.0, 0.0, 0.0],
            refract_angle: [1.0, 0.0],
            displacement: [wx / base_h, 0.0, 0.0, hy / base_h],
            aberration: [0.0, 0.0, 0.0, 0.0],
            aberration_angle: [1.0, 0.0],
            blur_radius: 14.0,
            scale_ref: 0.0,
            edge_width: 1.5,
            light_face_cm: [
                [0.9253, -0.4425, -0.0871, 0.4779],
                [-0.1682, 0.6773, -0.0960, 0.4676],
                [-0.1787, -0.4133, 0.9682, 0.4834],
            ],
            dark_face_cm: [
                [1.2918, -0.2243, -0.0226, 0.0800],
                [-0.0667, 1.1342, -0.0226, 0.0800],
                [-0.0667, -0.2243, 1.3359, 0.0800],
            ],
            light_face_quad: [[0.0; 3]; 6],
            dark_face_quad: [[0.0; 3]; 6],
            light_edge_opacity: [0.6284, 0.4581],
            dark_edge_opacity: [0.7300, 0.5200],
        }
    }

    /// The tinted regular appearance: more milky (bias ~0.69), a 65px blur at
    /// 2x and stronger contact contours. The other boundary of the slider.
    #[must_use]
    pub fn tinted() -> Self {
        Self {
            blur_radius: 65.0,
            light_face_cm: [
                [0.5655, 0.0243, -0.1112, 0.4910],
                [0.1239, 0.5192, -0.1474, 0.4858],
                [0.1407, 0.0068, 0.3305, 0.4844],
            ],
            dark_face_cm: [
                [1.1500, -0.1500, -0.0200, 0.1000],
                [-0.0500, 1.0500, -0.0200, 0.1000],
                [-0.0500, -0.1500, 1.1500, 0.1000],
            ],
            light_face_quad: [[0.0; 3]; 6],
            dark_face_quad: [[0.0; 3]; 6],
            light_edge_opacity: [0.6284, 0.4581],
            dark_edge_opacity: [0.7300, 0.5200],
            ..Self::transparent()
        }
    }

    /// Curated physical preset specifically tuned for macOS Dock bars:
    /// - Aligned to concentric 13px icon padding (Width X = 26px, Height Y = 26px @2x Retina)
    /// - 0.74px micro-diffusion (1.5px @2x) for 97% crisp clarity
    /// - 0% neutral milkiness for crystal-clear background transmission
    /// - 80% circular arc + 20% smooth transition (smoothing = 0.20) for distinct circular curvature
    /// - Pure physical symmetric 1-2px specular grazing highlight and lateral subtractive crevice AO
    #[must_use]
    pub fn dock() -> Self {
        Self::transparent()
            .with_amount(-54.0)
            .with_bezier_profile(0.18, 0.00, 0.00)
            .with_bevel_dimensions(26.0, 26.0)
            .with_blur_radius(1.5)
            .with_milkiness(0.0)
            .with_corner_smoothing(0.20)
    }

    /// Adaptive physical glass material preset for desktop window frames / titlebars.
    #[must_use]
    pub fn window_chrome(clarity: f32) -> Self {
        Self::blend(clarity)
            .with_bezier_profile(0.25, 0.05, 0.00)
            .with_bevel_dimensions(24.0, 24.0)
            .with_corner_smoothing(0.20)
    }

    /// Adaptive physical glass material preset for split-view sidebars (System Settings style).
    #[must_use]
    pub fn sidebar(clarity: f32) -> Self {
        Self::blend(clarity)
            .with_blur_radius(48.0)
            .with_milkiness(0.08)
            .with_bevel_dimensions(32.0, 32.0)
            .with_corner_smoothing(0.20)
    }

    /// Adaptive physical glass material preset for search fields and input capsules.
    #[must_use]
    pub fn search_field(clarity: f32) -> Self {
        Self::blend(clarity)
            .with_blur_radius(8.0)
            .with_bevel_dimensions(16.0, 16.0)
            .with_corner_smoothing(0.0)
    }

    /// Adaptive physical glass material preset for context menus and floating popovers.
    #[must_use]
    pub fn popover(clarity: f32) -> Self {
        Self::blend(clarity)
            .with_blur_radius(64.0)
            .with_milkiness(0.10)
            .with_bevel_dimensions(20.0, 20.0)
            .with_corner_smoothing(0.20)
    }

    /// Physical glass material preset tuned for 14pt macOS window controls (traffic lights).
    ///
    /// Features:
    /// - High spherical convexity (P1 = 1.25, P2 = 0.85, P3 = 0.20) for rich droplet bead lensing
    /// - Calibrated grazing highlight intensity (0.85) and dark rim (2.00)
    /// - Tight 1.0px sub-pixel edge width
    /// - Subtractive micro-crevice AO for natural dark perimeter contour without artificial strokes
    /// - Tailored chromatic internal tinting matching Apple's saturated jewel glass buttons
    #[must_use]
    pub fn traffic_light(kind: TrafficLightKind) -> Self {
        let glow = match kind {
            TrafficLightKind::Close | TrafficLightKind::Minimize | TrafficLightKind::Zoom => 0.60,
        };
        let mut m = Self::transparent()
            .with_bezier_profile(0.00, 0.00, 0.00)
            .with_highlight_intensity(0.85)
            .with_dark_rim(2.00)
            .with_internal_glow(glow)
            .with_bevel_dimensions(14.0, 14.0)
            .with_blur_radius(12.0)
            .with_edge_width(1.0);
        match kind {
            TrafficLightKind::Close => {
                m.light_face_cm = [
                    [1.3500, -0.0500, -0.0500, 0.5500],
                    [-0.1000, 0.4000, -0.0500, 0.1800],
                    [-0.1000, -0.0500, 0.4000, 0.1800],
                ];
            }
            TrafficLightKind::Minimize => {
                m.light_face_cm = [
                    [1.2500, 0.1500, -0.1000, 0.5200],
                    [0.0500, 1.1500, -0.1000, 0.4800],
                    [-0.1500, -0.1500, 0.3000, 0.1200],
                ];
            }
            TrafficLightKind::Zoom => {
                m.light_face_cm = [
                    [0.3500, -0.0500, -0.0500, 0.1800],
                    [-0.0500, 1.3000, -0.0500, 0.5200],
                    [-0.0500, -0.0500, 0.4500, 0.2000],
                ];
            }
        }
        m
    }

    /// Opalescent white translucent glass preset (porcelain / jade finish):
    /// Pure neutral white colloidal scattering at 28% without color cast.
    #[must_use]
    pub fn milky_jade() -> Self {
        Self::transparent().with_milkiness(0.28)
    }

    /// Sets the 4th-order Bernstein Bézier lensing curve profile parameters:
    /// - `p1`: Outer shoulder convexity (0.0 .. 2.0)
    /// - `p2`: Mid-belly arch/sag (0.0 .. 2.0)
    /// - `p3`: Inner landing slope (0.0 .. 1.0)
    #[must_use]
    pub fn with_bezier_profile(mut self, p1: f32, p2: f32, p3: f32) -> Self {
        self.inner_refract[2] = p1;
        self.inner_refract[3] = p2;
        self.refract_params[3] = p3;
        self
    }

    /// Sets the refraction displacement amount in device pixels.
    #[must_use]
    pub fn with_amount(mut self, amount: f32) -> Self {
        self.inner_refract[0] = amount;
        self
    }

    /// Sets the physical grazing highlight intensity (0.0 ..= 2.0).
    #[must_use]
    pub fn with_highlight_intensity(mut self, intensity: f32) -> Self {
        self.aberration[3] = intensity;
        self
    }

    /// Sets the subtractive perimeter crevice AO dark rim multiplier (0.0 ..= 2.0).
    #[must_use]
    pub fn with_dark_rim(mut self, rim: f32) -> Self {
        self.refract_params[0] = rim;
        self
    }

    /// Sets the internal translucent center and lower caustic glow brightness (0.0 ..= 2.0).
    #[must_use]
    pub fn with_internal_glow(mut self, glow: f32) -> Self {
        self.aberration_angle[1] = glow;
        self
    }

    /// Sets independent bevel dimensions in device pixels:
    /// - `width_x`: Left/right vertical edge bevel depth
    /// - `height_y`: Top/bottom horizontal edge bevel depth
    #[must_use]
    pub fn with_bevel_dimensions(mut self, width_x: f32, height_y: f32) -> Self {
        let wx = width_x.max(1.0);
        let hy = height_y.max(1.0);
        let base_h = (wx * hy).sqrt();
        self.inner_refract[1] = 1.0 / base_h;
        self.displacement = [wx / base_h, 0.0, 0.0, hy / base_h];
        self
    }

    /// Sets pure neutral white colloidal milkiness (0.0 ..= 1.0) without color cast.
    #[must_use]
    pub fn with_milkiness(mut self, milkiness: f32) -> Self {
        self.refract_params[2] = milkiness.clamp(0.0, 1.0);
        self
    }

    /// Sets continuous curvature smoothing ratio (0.0 for standard circle, 0.6 for Apple G2 squircle).
    #[must_use]
    pub fn with_corner_smoothing(mut self, smoothing: f32) -> Self {
        self.refract_params[1] = smoothing.clamp(0.0, 1.0);
        self
    }

    /// Sets the blur radius in device pixels.
    #[must_use]
    pub fn with_blur_radius(mut self, blur_radius: f32) -> Self {
        self.blur_radius = blur_radius.max(0.0);
        self
    }

    /// Sets the edge line transition width in device pixels.
    #[must_use]
    pub fn with_edge_width(mut self, edge_width: f32) -> Self {
        self.edge_width = edge_width.max(0.0);
        self
    }

    /// Interpolates the two measured regular appearances, matching a
    /// clarity slider where **left is clearest and right is blurriest**:
    /// `t = 0` is [`Self::transparent`] (most clear) and `t = 1` is
    /// [`Self::tinted`] (most blurred).
    #[must_use]
    pub fn blend(t: f32) -> Self {
        Self::lerp(&Self::transparent(), &Self::tinted(), t)
    }

    /// Per-field linear interpolation between two materials.
    #[must_use]
    pub fn lerp(a: &Self, b: &Self, amount: f32) -> Self {
        let amount = amount.clamp(0.0, 1.0);
        let mix = |x: f32, y: f32| {
            if amount <= 0.0 {
                x
            } else if amount >= 1.0 {
                y
            } else {
                x + (y - x) * amount
            }
        };
        let mix2 = |x: [f32; 2], y: [f32; 2]| [mix(x[0], y[0]), mix(x[1], y[1])];
        let mix4 = |x: [f32; 4], y: [f32; 4]| {
            [mix(x[0], y[0]), mix(x[1], y[1]), mix(x[2], y[2]), mix(x[3], y[3])]
        };
        let mix_matrix = |x: [[f32; 4]; 3], y: [[f32; 4]; 3]| {
            [mix4(x[0], y[0]), mix4(x[1], y[1]), mix4(x[2], y[2])]
        };
        let mix3 = |x: [f32; 3], y: [f32; 3]| [mix(x[0], y[0]), mix(x[1], y[1]), mix(x[2], y[2])];
        let mix_quad = |x: [[f32; 3]; 6], y: [[f32; 3]; 6]| {
            [
                mix3(x[0], y[0]),
                mix3(x[1], y[1]),
                mix3(x[2], y[2]),
                mix3(x[3], y[3]),
                mix3(x[4], y[4]),
                mix3(x[5], y[5]),
            ]
        };
        Self {
            inner_refract: mix4(a.inner_refract, b.inner_refract),
            refract_params: mix4(a.refract_params, b.refract_params),
            refract_angle: mix2(a.refract_angle, b.refract_angle),
            displacement: mix4(a.displacement, b.displacement),
            aberration: mix4(a.aberration, b.aberration),
            aberration_angle: mix2(a.aberration_angle, b.aberration_angle),
            blur_radius: mix(a.blur_radius, b.blur_radius),
            scale_ref: mix(a.scale_ref, b.scale_ref),
            edge_width: mix(a.edge_width, b.edge_width),
            light_face_cm: mix_matrix(a.light_face_cm, b.light_face_cm),
            dark_face_cm: mix_matrix(a.dark_face_cm, b.dark_face_cm),
            light_edge_opacity: mix2(a.light_edge_opacity, b.light_edge_opacity),
            dark_edge_opacity: mix2(a.dark_edge_opacity, b.dark_edge_opacity),
            light_face_quad: mix_quad(a.light_face_quad, b.light_face_quad),
            dark_face_quad: mix_quad(a.dark_face_quad, b.dark_face_quad),
        }
    }
}

impl Default for ContentGlassMaterial {
    fn default() -> Self {
        Self::transparent()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ContentGlassUniform {
    rect: [f32; 4],
    canvas: [f32; 2],
    half_size: [f32; 2],
    corner_radius: f32,
    dark: f32,
    _pad_frame: [f32; 2],
    inner_refract: [f32; 4],
    refract_params: [f32; 4],
    refract_angle: [f32; 2],
    _pad0: [f32; 2],
    displacement: [f32; 4],
    aberration: [f32; 4],
    aberration_angle: [f32; 2],
    _pad1: [f32; 2],
    blur_params: [f32; 4],
    edge_params: [f32; 4],
    face_cm0: [f32; 4],
    face_cm1: [f32; 4],
    face_cm2: [f32; 4],
    face_cm_dark0: [f32; 4],
    face_cm_dark1: [f32; 4],
    face_cm_dark2: [f32; 4],
    quad_light: [[f32; 4]; 6],
    quad_dark: [[f32; 4]; 6],
}

impl ContentGlassUniform {
    fn for_node(
        size: GpuSize,
        node: &ContentGlassNode,
        material: &ContentGlassMaterial,
        output_linear: f32,
    ) -> Self {
        let dark = node.appearance.0.clamp(0.0, 1.0);
        let (edge_lr, edge_tb) = if dark > 0.5 {
            (material.dark_edge_opacity[0], material.dark_edge_opacity[1])
        } else {
            (material.light_edge_opacity[0], material.light_edge_opacity[1])
        };
        Self {
            rect: [node.x, node.y, node.width, node.height],
            canvas: [size.width as f32, size.height as f32],
            half_size: [node.width * 0.5, node.height * 0.5],
            corner_radius: node.corner_radius,
            dark,
            _pad_frame: [0.0; 2],
            inner_refract: material.inner_refract,
            refract_params: material.refract_params,
            refract_angle: material.refract_angle,
            _pad0: [0.0; 2],
            displacement: material.displacement,
            aberration: material.aberration,
            aberration_angle: material.aberration_angle,
            _pad1: [0.0; 2],
            blur_params: [
                material.blur_radius,
                material.scale_ref,
                material.edge_width,
                output_linear,
            ],
            edge_params: [edge_lr, edge_tb, node.diffusion.clamp(0.0, 1.0), 0.0],
            face_cm0: material.light_face_cm[0],
            face_cm1: material.light_face_cm[1],
            face_cm2: material.light_face_cm[2],
            face_cm_dark0: material.dark_face_cm[0],
            face_cm_dark1: material.dark_face_cm[1],
            face_cm_dark2: material.dark_face_cm[2],
            quad_light: material.light_face_quad.map(|q| [q[0], q[1], q[2], 0.0]),
            quad_dark: material.dark_face_quad.map(|q| [q[0], q[1], q[2], 0.0]),
        }
    }
}

const BLUR_SHADER: &str = include_str!("../../../shaders/glass/content-glass-blur.wgsl");

/// GPU renderer for the fitted content-glass material.
#[derive(Debug)]
pub struct ContentGlassRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    size: GpuSize,
    material: ContentGlassMaterial,
    output_linear: f32,
    backdrop: Option<wgpu::Texture>,
    backdrop_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    blur_pipeline: wgpu::RenderPipeline,
    blur_bind_group_layout: wgpu::BindGroupLayout,
    blur_uniform: wgpu::Buffer,
    blur_temp: Option<wgpu::Texture>,
    blur_temp_view: wgpu::TextureView,
    blurred: Option<wgpu::Texture>,
    blurred_view: wgpu::TextureView,
    blurred_sigma: f32,
    uniform: wgpu::Buffer,
    uniform_stride: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct BlurUniform {
    dir: [f32; 2],
    sigma: f32,
    radius: f32,
}

impl ContentGlassRenderer {
    /// Creates a headless renderer.
    ///
    /// # Errors
    /// Returns an error when no adapter/device is available.
    pub async fn new(size: GpuSize, output_format: wgpu::TextureFormat) -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| GpuError::AdapterUnavailable(e.to_string()))?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("content-glass device"),
                ..Default::default()
            })
            .await
            .map_err(|e| GpuError::DeviceUnavailable(e.to_string()))?;
        Ok(Self::from_device(device, queue, size, output_format))
    }

    /// Builds a renderer from an existing device.
    ///
    /// # Panics
    /// Panics when `size` has a zero dimension.
    #[must_use]
    pub fn from_device(
        device: wgpu::Device,
        queue: wgpu::Queue,
        size: GpuSize,
        output_format: wgpu::TextureFormat,
    ) -> Self {
        assert!(size.is_valid(), "content-glass size must be non-zero");
        let alignment = device.limits().min_uniform_buffer_offset_alignment.max(1);
        let uniform_stride = CONTENT_GLASS_UNIFORM_BYTES.div_ceil(alignment) * alignment;
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("content-glass uniforms"),
            size: u64::from(uniform_stride) * MAX_CONTENT_GLASS_NODES as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("content-glass backdrop sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("content-glass layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(u64::from(
                            CONTENT_GLASS_UNIFORM_BYTES,
                        )),
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("content-glass pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let vertex = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("content-glass vertex"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/glass/content-glass-vertex.wgsl").into(),
            ),
        });
        let fragment = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("content-glass fragment"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/glass/content-glass.wgsl").into(),
            ),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("content-glass pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &vertex,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fragment,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: output_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let blur_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("content-glass blur layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                std::mem::size_of::<BlurUniform>() as u64,
                            ),
                        },
                        count: None,
                    },
                ],
            });
        let blur_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("content-glass blur pipeline layout"),
            bind_group_layouts: &[&blur_bind_group_layout],
            push_constant_ranges: &[],
        });
        let blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("content-glass blur shader"),
            source: wgpu::ShaderSource::Wgsl(BLUR_SHADER.into()),
        });
        let blur_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("content-glass blur pipeline"),
            layout: Some(&blur_layout),
            vertex: wgpu::VertexState {
                module: &blur_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blur_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let blur_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("content-glass blur uniform"),
            size: std::mem::size_of::<BlurUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let placeholder = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("content-glass placeholder backdrop"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let backdrop_view = placeholder.create_view(&wgpu::TextureViewDescriptor::default());
        let blur_temp_view = placeholder.create_view(&wgpu::TextureViewDescriptor::default());
        let blurred_view = placeholder.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            device,
            queue,
            size,
            material: ContentGlassMaterial::default(),
            output_linear: if output_format.is_srgb() { 1.0 } else { 0.0 },
            backdrop: None,
            backdrop_view,
            sampler,
            bind_group_layout,
            pipeline,
            blur_pipeline,
            blur_bind_group_layout,
            blur_uniform,
            blur_temp: None,
            blur_temp_view,
            blurred: None,
            blurred_view,
            blurred_sigma: f32::NAN,
            uniform,
            uniform_stride,
        }
    }

    #[must_use]
    pub const fn material(&self) -> &ContentGlassMaterial {
        &self.material
    }

    pub fn set_material(&mut self, material: ContentGlassMaterial) {
        self.material = material;
        self.apply_blur();
    }

    /// Drives the appearance slider: `0.0` is the most clear (transparent)
    /// appearance and `1.0` is the most blurred (tinted) one. Values in
    /// between interpolate every material parameter.
    pub fn set_appearance_blend(&mut self, t: f32) {
        self.material = ContentGlassMaterial::blend(t);
    }

    /// Uploads the composited backdrop behind the glass and builds its mip
    /// chain. `rgba8` is tightly packed sRGB bytes.
    ///
    /// # Errors
    /// Returns [`GpuError::InvalidBackgroundFrame`] for malformed metadata.
    pub fn set_backdrop_rgba8(
        &mut self,
        width: u32,
        height: u32,
        rgba8: &[u8],
    ) -> Result<(), GpuError> {
        let stride = width.checked_mul(4).ok_or(GpuError::InvalidBackgroundFrame {
            width,
            height,
            stride: 0,
            byte_len: rgba8.len(),
        })?;
        let expected = (stride as usize).checked_mul(height as usize).ok_or(
            GpuError::InvalidBackgroundFrame { width, height, stride, byte_len: rgba8.len() },
        )?;
        if width == 0 || height == 0 || rgba8.len() != expected {
            return Err(GpuError::InvalidBackgroundFrame {
                width,
                height,
                stride,
                byte_len: rgba8.len(),
            });
        }
        self.size = GpuSize::new(width, height);
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("content-glass backdrop"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba8,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(stride),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
        let make = |label: &str| {
            self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
        };
        let blur_temp = make("content-glass blur temp");
        let blurred = make("content-glass blurred backdrop");
        self.backdrop_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.blur_temp_view = blur_temp.create_view(&wgpu::TextureViewDescriptor::default());
        self.blurred_view = blurred.create_view(&wgpu::TextureViewDescriptor::default());
        self.backdrop = Some(texture);
        self.blur_temp = Some(blur_temp);
        self.blurred = Some(blurred);
        self.blurred_sigma = f32::NAN;
        self.apply_blur();
        Ok(())
    }

    /// Runs the separable Gaussian blur that produces the texture the glass
    /// samples. Re-runs only when the material's blur radius changed.
    fn apply_blur(&mut self) {
        let (Some(backdrop), Some(blur_temp), Some(blurred)) =
            (&self.backdrop, &self.blur_temp, &self.blurred)
        else {
            return;
        };
        let sigma = self.material.blur_radius.max(0.5);
        if (self.blurred_sigma - sigma).abs() < f32::EPSILON {
            return;
        }
        self.blurred_sigma = sigma;
        let radius = (sigma * 2.0).ceil();
        for (dir, src, dst) in
            [([1.0f32, 0.0], backdrop, blur_temp), ([0.0f32, 1.0], blur_temp, blurred)]
        {
            self.queue.write_buffer(
                &self.blur_uniform,
                0,
                bytemuck::bytes_of(&BlurUniform { dir, sigma, radius }),
            );
            let src_view = src.create_view(&wgpu::TextureViewDescriptor::default());
            let dst_view = dst.create_view(&wgpu::TextureViewDescriptor::default());
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("content-glass blur bind group"),
                layout: &self.blur_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&src_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.blur_uniform.as_entire_binding(),
                    },
                ],
            });
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("content-glass blur pass"),
            });
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("content-glass blur pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &dst_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(&self.blur_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
            self.queue.submit([encoder.finish()]);
        }
    }

    /// Draws the elements into `output_view` with the given load operation.
    ///
    /// # Panics
    /// Panics when `nodes.len()` exceeds [`MAX_CONTENT_GLASS_NODES`].
    pub fn render_to_view(
        &self,
        output_view: &wgpu::TextureView,
        nodes: &[ContentGlassNode],
        load: wgpu::LoadOp<wgpu::Color>,
    ) {
        assert!(
            nodes.len() <= MAX_CONTENT_GLASS_NODES,
            "content-glass supports at most {MAX_CONTENT_GLASS_NODES} nodes"
        );
        for (index, node) in nodes.iter().enumerate() {
            let uniform =
                ContentGlassUniform::for_node(self.size, node, &self.material, self.output_linear);
            let offset = u64::from(self.uniform_stride) * index as u64;
            self.queue.write_buffer(&self.uniform, offset, bytemuck::bytes_of(&uniform));
        }
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("content-glass bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.blurred_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.uniform,
                        offset: 0,
                        size: wgpu::BufferSize::new(u64::from(CONTENT_GLASS_UNIFORM_BYTES)),
                    }),
                },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("content-glass encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("content-glass pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load, store: wgpu::StoreOp::Store },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.pipeline);
            for index in 0..nodes.len() {
                let offset = self.uniform_stride * index as u32;
                pass.set_bind_group(0, &bind_group, &[offset]);
                pass.draw(0..4, 0..1);
            }
        }
        self.queue.submit([encoder.finish()]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_matches_wgsl_layout_size() {
        assert_eq!(
            std::mem::size_of::<ContentGlassUniform>(),
            CONTENT_GLASS_UNIFORM_BYTES as usize,
        );
    }

    #[test]
    fn default_material_is_the_fitted_macos_27_set() {
        let m = ContentGlassMaterial::default();
        // Default is the analytical macOS 27 physics model (White=0.96, Black=0.40, Sat=1.20, MaxLumaSDR=0.94).
        assert!((m.light_face_cm[0][3] - 0.4779).abs() < 1.0e-4);
        assert!((m.dark_face_cm[0][3] - 0.0800).abs() < 1.0e-4);
        // Anisotropic contact contour: left/right darker than top/bottom.
        assert!(m.light_edge_opacity[0] > m.light_edge_opacity[1]);
        assert!(m.dark_edge_opacity[0] > m.dark_edge_opacity[1]);
        assert!((m.light_edge_opacity[0] - 0.6284).abs() < 1.0e-4);
        assert!((m.dark_edge_opacity[0] - 0.7300).abs() < 1.0e-4);
        assert!((m.dark_edge_opacity[1] - 0.5200).abs() < 1.0e-4);
        // The tinted boundary keeps the original frosted fit.
        let t = ContentGlassMaterial::tinted();
        assert!((t.light_face_cm[0][3] - 0.4910).abs() < 1.0e-4);
        assert!((t.blur_radius - 65.0).abs() < f32::EPSILON);
    }

    #[test]
    fn appearance_blend_hits_both_boundaries_and_interpolates() {
        // Left (0) is clear/transparent, right (1) is blurred/tinted.
        assert_eq!(ContentGlassMaterial::blend(0.0), ContentGlassMaterial::transparent());
        assert_eq!(ContentGlassMaterial::blend(1.0), ContentGlassMaterial::tinted());
        let mid = ContentGlassMaterial::blend(0.5);
        let blurred = ContentGlassMaterial::tinted();
        let clear = ContentGlassMaterial::transparent();
        assert!(mid.blur_radius > clear.blur_radius && mid.blur_radius < blurred.blur_radius);
        assert!(mid.light_face_cm[0][3] < blurred.light_face_cm[0][3]);
        assert!(mid.light_face_cm[0][3] > clear.light_face_cm[0][3]);
        // Clamped outside [0, 1].
        assert_eq!(ContentGlassMaterial::blend(-1.0), clear);
        assert_eq!(ContentGlassMaterial::blend(2.0), blurred);
    }

    #[test]
    fn node_uniform_packs_geometry_and_appearance() {
        let size = GpuSize::new(1280, 800);
        let node = ContentGlassNode::new(100.0, 50.0, 320.0, 120.0)
            .corner_radius(60.0)
            .appearance(ContentGlassAppearance::DARK)
            .diffusion(0.5);
        let u = ContentGlassUniform::for_node(size, &node, &ContentGlassMaterial::default(), 1.0);
        assert_eq!(u.rect, [100.0, 50.0, 320.0, 120.0]);
        assert_eq!(u.canvas, [1280.0, 800.0]);
        assert_eq!(u.half_size, [160.0, 60.0]);
        assert!((u.dark - 1.0).abs() < f32::EPSILON);
        assert!((u.blur_params[3] - 1.0).abs() < f32::EPSILON);
        // Dark appearance selects the dark edge opacities.
        assert!((u.edge_params[0] - 0.7300).abs() < 1.0e-4);
        assert!((u.edge_params[2] - 0.5).abs() < f32::EPSILON);
        assert!((u.corner_radius - 60.0).abs() < f32::EPSILON);
        assert!((u.inner_refract[0] - (-56.0)).abs() < 1.0e-4);
    }
}
