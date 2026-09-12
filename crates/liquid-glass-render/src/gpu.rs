use std::{borrow::Cow, fmt};

use crate::ScrollEdgeStyle;
use bytemuck::{Pod, Zeroable};
use liquid_glass_scene::{
    GlassAccessibility, GlassEnvironment, GlassNode, GlassRenderOptions, GlassScene, GlassShape,
    GlassVariant, Rect,
};

const DEFAULT_OUTPUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
#[allow(clippy::cast_possible_truncation)]
const GLASS_UNIFORM_SIZE: u32 = std::mem::size_of::<GlassUniform>() as u32;
const MAX_KAWASE_RADIUS: usize = 200;
const MAX_GRADIENT_COMPOSITES: usize = 2;

const FEATURE_EDGE_BLUR: i32 = 1;
const FEATURE_REDUCED_TRANSPARENCY: i32 = 1 << 1;
const FEATURE_INCREASED_CONTRAST: i32 = 1 << 2;
const FEATURE_REDUCED_MOTION: i32 = 1 << 3;
const FEATURE_CLEAR_VARIANT: i32 = 1 << 4;
const FEATURE_TRAFFIC_LIGHT: i32 = 1 << 5;
const FEATURE_TRAFFIC_LIGHT_PHYSICAL: i32 = 1 << 6;
const FEATURE_TRAFFIC_LIGHT_BEAD: i32 = 1 << 8;

const FULLSCREEN_VERTEX_ATTRIBUTES: &[wgpu::VertexAttribute] = &[wgpu::VertexAttribute {
    format: wgpu::VertexFormat::Float32x2,
    offset: 0,
    shader_location: 0,
}];

/// Maximum number of glass nodes that can be drawn in one frame.
pub const MAX_GLASS_NODES: usize = 64;

/// Pixel dimensions for an offscreen render target.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuSize {
    pub width: u32,
    pub height: u32,
}

impl GpuSize {
    #[must_use]
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    #[must_use]
    pub const fn is_valid(self) -> bool {
        self.width > 0 && self.height > 0
    }
}

/// Errors returned while creating the headless GPU backend.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GpuError {
    InvalidSize,
    InvalidBackgroundFrame { width: u32, height: u32, stride: u32, byte_len: usize },
    AdapterUnavailable(String),
    DeviceUnavailable(String),
    SceneNodeLimitExceeded { limit: usize },
    MultipleScrollEdgesInFrameBatch,
    GradientCompositeLimitExceeded { limit: usize },
}

impl fmt::Display for GpuError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSize => write!(formatter, "offscreen target dimensions must be non-zero"),
            Self::InvalidBackgroundFrame { width, height, stride, byte_len } => write!(
                formatter,
                "invalid RGBA8 backdrop frame ({width}x{height}, stride {stride}, {byte_len} bytes)",
            ),
            Self::AdapterUnavailable(error) => {
                write!(formatter, "no suitable GPU adapter: {error}")
            }
            Self::DeviceUnavailable(error) => {
                write!(formatter, "failed to create GPU device: {error}")
            }
            Self::SceneNodeLimitExceeded { limit } => {
                write!(formatter, "scene contains more than {limit} glass nodes")
            }
            Self::MultipleScrollEdgesInFrameBatch => {
                write!(formatter, "a GPU frame batch supports one scroll-edge blur")
            }
            Self::GradientCompositeLimitExceeded { limit } => {
                write!(formatter, "a GPU frame batch supports {limit} gradient composites")
            }
        }
    }
}

impl std::error::Error for GpuError {}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GlassUniform {
    // [width, height, uniform dpr (always 1: geometry is already physical),
    // device pixels per logical point].
    resolution_dpr_pad: [f32; 4],
    mouse_and_spring: [f32; 4],
    shape: [f32; 4],
    capsule_bezier_x: [f32; 4],
    capsule_bezier_y: [f32; 4],
    merge_glare_shadow: [f32; 4],
    shadow_position_bg_ratio: [f32; 3],
    bg_type: i32,
    flags: [i32; 4],
    tint: [f32; 4],
    refraction_and_fresnel: [f32; 6],
    glare: [f32; 5],
    // [refraction strength, opacity, interaction strength, environment
    // luminance, environment contrast].
    _pad: [f32; 5],
    // [tint response, ambient spill response, shadow response, size response].
    adaptive: [f32; 4],
    // [center x, center y, width, height] for up to four fused shapes in
    // bottom-origin logical pixels.
    fused_bounds: [[f32; 4]; 4],
    // [corner radius, corner exponent, enabled, unused] for each shape.
    fused_geometry: [[f32; 4]; 4],
    // [spring pointer x, spring pointer y, parallax, focus].
    interaction_state: [f32; 4],
    // [substrate coverage, lower substrate coverage, lower tint coverage,
    // angular lower light] for TrafficLightPhysical.
    traffic_light: [f32; 4],
    // [normalized lower-light angle, body thickness, internal scattering,
    // side-edge darkness].
    traffic_light_light: [f32; 4],
    // [side-edge width multiplier, lower-light softness, side bias,
    // side-angle in degrees].
    traffic_light_edge: [f32; 4],
    interaction_response: [f32; 4],
    core_light: [f32; 4],
    core_light_gradient: [f32; 4],
    rim_profile: [f32; 4],
    bead_a: [f32; 4],
    bead_b: [f32; 4],
    bead_c: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct KawaseUniform {
    halfpixel: [f32; 2],
    offset: f32,
    alpha: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct VibrancyTintUniform {
    tint: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GradientCompositeUniform {
    fallback: [f32; 4],
    output_size: [f32; 2],
    _pad: [f32; 2],
    // [start_y, end_y, minimum blur mix, unused] in physical top-origin
    // pixels. The minimum mix keeps a toolbar softly blurred at its lower
    // edge instead of fading all the way to a sharp copy.
    gradient: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct WindowMaskUniform {
    size: [f32; 2],
    radius: f32,
    exponent: f32,
}

struct FrameTargets {
    scene: wgpu::Texture,
    scene_view: wgpu::TextureView,
    _kawase_half: wgpu::Texture,
    kawase_half_view: wgpu::TextureView,
    _kawase_quarter: wgpu::Texture,
    kawase_quarter_view: wgpu::TextureView,
    _kawase_eighth: wgpu::Texture,
    kawase_eighth_view: wgpu::TextureView,
    _kawase_up_quarter: wgpu::Texture,
    kawase_up_quarter_view: wgpu::TextureView,
    _kawase_up_half: wgpu::Texture,
    kawase_up_half_view: wgpu::TextureView,
    _kawase_full: wgpu::Texture,
    kawase_full_view: wgpu::TextureView,
    output: wgpu::Texture,
    output_view: wgpu::TextureView,
}

impl FrameTargets {
    fn new(device: &wgpu::Device, size: GpuSize, format: wgpu::TextureFormat) -> Self {
        let descriptor = |label, target_size: GpuSize| wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: target_size.width,
                height: target_size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        };
        let scene = device.create_texture(&descriptor("liquid-glass scene texture", size));
        let kawase_half_size = downsampled_size(size, 2);
        let kawase_quarter_size = downsampled_size(size, 4);
        let kawase_eighth_size = downsampled_size(size, 8);
        let kawase_half = device.create_texture(&descriptor(
            "liquid-glass vibrancy half-resolution texture",
            kawase_half_size,
        ));
        let kawase_quarter = device.create_texture(&descriptor(
            "liquid-glass vibrancy quarter-resolution texture",
            kawase_quarter_size,
        ));
        let kawase_eighth = device.create_texture(&descriptor(
            "liquid-glass vibrancy eighth-resolution texture",
            kawase_eighth_size,
        ));
        let kawase_up_quarter = device.create_texture(&descriptor(
            "liquid-glass vibrancy upsampled quarter-resolution texture",
            kawase_quarter_size,
        ));
        let kawase_up_half = device.create_texture(&descriptor(
            "liquid-glass vibrancy upsampled half-resolution texture",
            kawase_half_size,
        ));
        let kawase_full = device
            .create_texture(&descriptor("liquid-glass vibrancy full-resolution texture", size));
        let output = device.create_texture(&descriptor("liquid-glass output texture", size));
        let scene_view = scene.create_view(&wgpu::TextureViewDescriptor::default());
        let kawase_half_view = kawase_half.create_view(&wgpu::TextureViewDescriptor::default());
        let kawase_quarter_view =
            kawase_quarter.create_view(&wgpu::TextureViewDescriptor::default());
        let kawase_eighth_view = kawase_eighth.create_view(&wgpu::TextureViewDescriptor::default());
        let kawase_up_quarter_view =
            kawase_up_quarter.create_view(&wgpu::TextureViewDescriptor::default());
        let kawase_up_half_view =
            kawase_up_half.create_view(&wgpu::TextureViewDescriptor::default());
        let kawase_full_view = kawase_full.create_view(&wgpu::TextureViewDescriptor::default());
        let output_view = output.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            scene,
            scene_view,
            _kawase_half: kawase_half,
            kawase_half_view,
            _kawase_quarter: kawase_quarter,
            kawase_quarter_view,
            _kawase_eighth: kawase_eighth,
            kawase_eighth_view,
            _kawase_up_quarter: kawase_up_quarter,
            kawase_up_quarter_view,
            _kawase_up_half: kawase_up_half,
            kawase_up_half_view,
            _kawase_full: kawase_full,
            kawase_full_view,
            output,
            output_view,
        }
    }
}

/// A real `wgpu` offscreen compositor for a scene of SDF Glass nodes.
///
/// This backend intentionally has no window or Iced dependency. It renders a
/// reference background into an offscreen texture, applies a shared
/// `vibrancy-rs` Dual Kawase pyramid, and samples the result while
/// evaluating the reference SDF, refraction, dispersion, Fresnel, glare, and
/// tint composition. Multiple nodes are composed in z-order through ping-pong
/// targets, so each later node samples the already-composited lower layers.
pub struct GpuRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    size: GpuSize,
    output_format: wgpu::TextureFormat,
    targets: FrameTargets,
    sampler: wgpu::Sampler,
    fullscreen_vertex_buffer: wgpu::Buffer,
    background_bind_group: wgpu::BindGroup,
    kawase_bind_group_layout: wgpu::BindGroupLayout,
    kawase_down_from_scene_bind_group: wgpu::BindGroup,
    kawase_down_from_output_bind_group: wgpu::BindGroup,
    kawase_down_half_bind_group: wgpu::BindGroup,
    kawase_down_quarter_bind_group: wgpu::BindGroup,
    kawase_up_eighth_bind_group: wgpu::BindGroup,
    kawase_up_quarter_bind_group: wgpu::BindGroup,
    kawase_up_quarter_source_bind_group: wgpu::BindGroup,
    kawase_up_half_bind_group: wgpu::BindGroup,
    glass_bind_group_layout: wgpu::BindGroupLayout,
    glass_from_scene_bind_group: wgpu::BindGroup,
    glass_from_output_bind_group: wgpu::BindGroup,
    glass_uniform: wgpu::Buffer,
    kawase_uniform: wgpu::Buffer,
    kawase_uniform_stride: u32,
    glass_uniform_stride: u32,
    placeholder_texture: wgpu::Texture,
    background_texture: Option<wgpu::Texture>,
    background_texture_ratio: f32,
    background_pipeline: wgpu::RenderPipeline,
    kawase_downsample_pipeline: wgpu::RenderPipeline,
    kawase_upsample_pipeline: wgpu::RenderPipeline,
    shadow_pipeline: wgpu::RenderPipeline,
    glass_pipeline: wgpu::RenderPipeline,
    copy_bind_group_layout: wgpu::BindGroupLayout,
    copy_bind_group: wgpu::BindGroup,
    copy_pipeline: wgpu::RenderPipeline,
    foreground_copy_pipeline: wgpu::RenderPipeline,
    window_mask_bind_group_layout: wgpu::BindGroupLayout,
    window_mask_bind_group: wgpu::BindGroup,
    window_mask_uniform: wgpu::Buffer,
    window_mask_pipeline: wgpu::RenderPipeline,
    window_corner_radius: f32,
    vibrancy_tint_bind_group_layout: wgpu::BindGroupLayout,
    vibrancy_tint_bind_group: wgpu::BindGroup,
    vibrancy_tint_uniform: wgpu::Buffer,
    vibrancy_tint_pipeline: wgpu::RenderPipeline,
    gradient_composite_bind_group_layout: wgpu::BindGroupLayout,
    gradient_composite_bind_group: wgpu::BindGroup,
    gradient_composite_uniform: wgpu::Buffer,
    gradient_composite_uniform_stride: u32,
    gradient_composite_pipeline: wgpu::RenderPipeline,
    transparent_background: bool,
    options: GlassRenderOptions,
}

/// A group of Liquid Glass operations encoded into one GPU command buffer.
///
/// The batch owns a monotonically increasing uniform slot cursor so separate
/// glass groups can safely share one command buffer. Call [`Self::submit`] only
/// after all operations for the phase have been encoded.
#[derive(Debug)]
pub struct GpuFrameBatch<'a> {
    renderer: &'a GpuRenderer,
    encoder: wgpu::CommandEncoder,
    next_uniform_slot: usize,
    gradient_slots_used: usize,
    scroll_edge_encoded: bool,
}

impl GpuFrameBatch<'_> {
    fn reserve_uniform_slots(&mut self, count: usize) -> Result<usize, GpuError> {
        let Some(next) = self.next_uniform_slot.checked_add(count) else {
            return Err(GpuError::SceneNodeLimitExceeded { limit: MAX_GLASS_NODES });
        };
        if next > MAX_GLASS_NODES {
            return Err(GpuError::SceneNodeLimitExceeded { limit: MAX_GLASS_NODES });
        }
        let first = self.next_uniform_slot;
        self.next_uniform_slot = next;
        Ok(first)
    }

    fn reserve_gradient_slot(&mut self) -> Result<usize, GpuError> {
        if self.gradient_slots_used >= MAX_GRADIENT_COMPOSITES {
            return Err(GpuError::GradientCompositeLimitExceeded {
                limit: MAX_GRADIENT_COMPOSITES,
            });
        }
        let slot = self.gradient_slots_used;
        self.gradient_slots_used += 1;
        Ok(slot)
    }

    /// Encodes the renderer's current backdrop into an external target.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::SceneNodeLimitExceeded`] when the batch has no
    /// dynamic-uniform slots left.
    pub fn render_background_to_view(
        &mut self,
        output_view: &wgpu::TextureView,
    ) -> Result<(), GpuError> {
        let slot = self.reserve_uniform_slots(1)?;
        self.renderer.encode_background_to_view(&mut self.encoder, output_view, slot);
        Ok(())
    }

    /// Clears an external render target without sampling a backdrop texture.
    ///
    /// Transparent-window integrations use this when the platform compositor
    /// already owns the desktop backdrop. It is equivalent to the initial
    /// transparent background pass, but avoids a full-screen fragment shader.
    pub fn clear_view(&mut self, output_view: &wgpu::TextureView, color: wgpu::Color) {
        let _pass = self.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("liquid-glass external clear pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(color),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
    }

    /// Encodes one flat-color region without submitting the command buffer.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::SceneNodeLimitExceeded`] when the batch has no
    /// dynamic-uniform slots left.
    pub fn render_solid_region_to_view(
        &mut self,
        output_view: &wgpu::TextureView,
        region: (u32, u32, u32, u32),
        color: [f32; 4],
    ) -> Result<(), GpuError> {
        let slot = self.reserve_uniform_slots(1)?;
        self.renderer.encode_solid_region_to_view(
            &mut self.encoder,
            output_view,
            region,
            color,
            slot,
        );
        Ok(())
    }

    /// Encodes a flat-color region into the internal composition target.
    ///
    /// This is used for measured, platform-specific separator pixels that
    /// must remain above the glass material but below framework foreground
    /// content.
    pub fn render_solid_region_to_output(
        &mut self,
        region: (u32, u32, u32, u32),
        color: [f32; 4],
    ) -> Result<(), GpuError> {
        let slot = self.reserve_uniform_slots(1)?;
        self.renderer.encode_solid_region_to_view(
            &mut self.encoder,
            &self.renderer.targets.output_view,
            region,
            color,
            slot,
        );
        Ok(())
    }

    /// Seeds the internal compositor from application content and evaluates a
    /// glass scene. The result remains internal for subsequent batch stages.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::SceneNodeLimitExceeded`] when all scenes in the
    /// batch exceed the renderer's dynamic-uniform capacity.
    pub fn render_scene_with_source(
        &mut self,
        source_texture: &wgpu::Texture,
        scene: &GlassScene,
        time_seconds: f32,
    ) -> Result<(), GpuError> {
        let nodes = scene.nodes_in_render_order();
        let first_slot = self.reserve_uniform_slots(nodes.len())?;
        self.renderer.encode_nodes_to_view(
            &mut self.encoder,
            &self.renderer.targets.output_view,
            &nodes,
            time_seconds,
            false,
            Some(source_texture),
            None,
            scene.render_options().unwrap_or(self.renderer.options),
            first_slot,
        )
    }

    /// Evaluates a glass scene over the batch's current internal output.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::SceneNodeLimitExceeded`] when all scenes in the
    /// batch exceed the renderer's dynamic-uniform capacity.
    pub fn render_scene_over_output(
        &mut self,
        scene: &GlassScene,
        time_seconds: f32,
    ) -> Result<(), GpuError> {
        let nodes = scene.nodes_in_render_order();
        let first_slot = self.reserve_uniform_slots(nodes.len())?;
        self.renderer.encode_nodes_to_view(
            &mut self.encoder,
            &self.renderer.targets.output_view,
            &nodes,
            time_seconds,
            false,
            Some(&self.renderer.targets.output),
            None,
            scene.render_options().unwrap_or(self.renderer.options),
            first_slot,
        )
    }

    /// Alpha-composites an application-owned foreground into the current
    /// internal output.
    pub fn composite_texture_to_output(&mut self, source_texture: &wgpu::Texture) {
        self.renderer.encode_composite_texture_to_output(&mut self.encoder, source_texture);
    }

    /// Encodes a scroll-edge blur over the current internal output.
    ///
    /// # Errors
    ///
    /// The scroll-edge treatment is a vertical gradient blur with a clear
    /// endpoint. Other gradient composites can be encoded in the same batch;
    /// each one receives its own blur and composition uniform slot.
    pub fn render_scroll_edge_to_output(
        &mut self,
        region: (u32, u32, u32, u32),
        fade_start_y: u32,
        maximum_radius: u32,
        fallback_color: [f32; 4],
        style: ScrollEdgeStyle,
    ) -> Result<(), GpuError> {
        if self.scroll_edge_encoded {
            return Err(GpuError::MultipleScrollEdgesInFrameBatch);
        }
        self.scroll_edge_encoded = true;
        let fade_start_y = match style {
            ScrollEdgeStyle::Soft => fade_start_y,
            ScrollEdgeStyle::Hard => region.1.saturating_add(region.3),
        };
        self.render_vertical_gradient_blur_to_output(
            region,
            fade_start_y,
            maximum_radius,
            fallback_color,
            0.0,
        )
    }

    /// Encodes a vertical blur gradient over the current internal output.
    ///
    /// `fade_start_y` is the physical y coordinate where the blur starts to
    /// reduce. `minimum_blur_mix` controls how much of the blurred result is
    /// retained at the bottom of the region; a non-zero value is useful for a
    /// fused titlebar whose lower edge remains soft rather than clear.
    pub fn render_vertical_gradient_blur_to_output(
        &mut self,
        region: (u32, u32, u32, u32),
        fade_start_y: u32,
        maximum_radius: u32,
        fallback_color: [f32; 4],
        minimum_blur_mix: f32,
    ) -> Result<(), GpuError> {
        let gradient_slot = self.reserve_gradient_slot()?;
        self.renderer.encode_vertical_gradient_blur_to_output_with_source(
            &mut self.encoder,
            &self.renderer.targets.output,
            region,
            fade_start_y,
            0,
            maximum_radius,
            fallback_color,
            minimum_blur_mix,
            gradient_slot,
        );
        Ok(())
    }

    /// Encodes the current internal output into an external target.
    pub fn copy_output_to_view(&mut self, output_view: &wgpu::TextureView) {
        self.renderer.encode_copy_output_to_view(&mut self.encoder, output_view);
    }

    /// Submits every operation in this batch as one command buffer.
    pub fn submit(self) {
        self.renderer.queue.submit([self.encoder.finish()]);
    }
}

impl fmt::Debug for GpuRenderer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("GpuRenderer").field("size", &self.size).finish_non_exhaustive()
    }
}

impl GpuRenderer {
    /// Creates a headless renderer using the best available local adapter.
    ///
    /// # Errors
    ///
    /// Returns an error when no compatible adapter or logical device can be
    /// created for the requested target.
    pub async fn new_headless(size: GpuSize) -> Result<Self, GpuError> {
        if !size.is_valid() {
            return Err(GpuError::InvalidSize);
        }

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|error| GpuError::AdapterUnavailable(error.to_string()))?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("liquid-glass device"),
                ..Default::default()
            })
            .await
            .map_err(|error| GpuError::DeviceUnavailable(error.to_string()))?;

        Ok(Self::from_device(device, queue, size))
    }

    /// Builds a renderer from an existing device. Useful for tests and for
    /// applications that own the window/surface lifecycle.
    ///
    /// # Panics
    ///
    /// Panics when either target dimension is zero.
    #[must_use]
    pub fn from_device(device: wgpu::Device, queue: wgpu::Queue, size: GpuSize) -> Self {
        Self::from_device_with_format(device, queue, size, DEFAULT_OUTPUT_FORMAT)
    }

    /// Builds a renderer whose output pipelines target the supplied texture format.
    ///
    /// This is used by window integrations because a platform Surface may
    /// prefer BGRA sRGB instead of RGBA sRGB.
    ///
    /// # Panics
    ///
    /// Panics when either target dimension is zero.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn from_device_with_format(
        device: wgpu::Device,
        queue: wgpu::Queue,
        size: GpuSize,
        output_format: wgpu::TextureFormat,
    ) -> Self {
        assert!(size.is_valid(), "GPU renderer size must be non-zero");
        let targets = FrameTargets::new(&device, size, output_format);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("liquid-glass linear sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let fullscreen_vertex_buffer = create_fullscreen_vertex_buffer(&device, &queue);
        let glass_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("liquid-glass backdrop layout"),
                entries: &[
                    texture_binding(0),
                    texture_binding(1),
                    sampler_binding(2),
                    uniform_binding(3, std::mem::size_of::<GlassUniform>()),
                ],
            });
        let glass_uniform_stride = uniform_stride(&device);
        let glass_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("liquid-glass material uniform"),
            size: u64::from(glass_uniform_stride) * MAX_GLASS_NODES as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let kawase_bind_group_layout = create_kawase_bind_group_layout(&device);
        let kawase_uniform_stride =
            uniform_stride_for_size(&device, std::mem::size_of::<KawaseUniform>());
        let kawase_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("liquid-glass vibrancy Kawase uniform"),
            size: u64::from(kawase_uniform_stride)
                * (MAX_GLASS_NODES as u64 + MAX_GRADIENT_COMPOSITES as u64)
                * 8,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let placeholder_texture = create_placeholder_texture(&device);
        let glass_from_scene_bind_group = create_glass_bind_group(
            &device,
            &glass_bind_group_layout,
            &targets.scene_view,
            &targets.kawase_full_view,
            &sampler,
            &glass_uniform,
        );
        let glass_from_output_bind_group = create_glass_bind_group(
            &device,
            &glass_bind_group_layout,
            &targets.output_view,
            &targets.kawase_full_view,
            &sampler,
            &glass_uniform,
        );
        let background_bind_group = create_background_bind_group(
            &device,
            &glass_bind_group_layout,
            &placeholder_texture,
            &sampler,
            &glass_uniform,
        );
        let kawase_down_from_scene_bind_group = create_kawase_bind_group(
            &device,
            &kawase_bind_group_layout,
            &targets.scene_view,
            &sampler,
            &kawase_uniform,
        );
        let kawase_down_from_output_bind_group = create_kawase_bind_group(
            &device,
            &kawase_bind_group_layout,
            &targets.output_view,
            &sampler,
            &kawase_uniform,
        );
        let kawase_down_half_bind_group = create_kawase_bind_group(
            &device,
            &kawase_bind_group_layout,
            &targets.kawase_half_view,
            &sampler,
            &kawase_uniform,
        );
        let kawase_down_quarter_bind_group = create_kawase_bind_group(
            &device,
            &kawase_bind_group_layout,
            &targets.kawase_quarter_view,
            &sampler,
            &kawase_uniform,
        );
        let kawase_up_eighth_bind_group = create_kawase_bind_group(
            &device,
            &kawase_bind_group_layout,
            &targets.kawase_eighth_view,
            &sampler,
            &kawase_uniform,
        );
        let kawase_up_quarter_bind_group = create_kawase_bind_group(
            &device,
            &kawase_bind_group_layout,
            &targets.kawase_up_quarter_view,
            &sampler,
            &kawase_uniform,
        );
        let kawase_up_quarter_source_bind_group = create_kawase_bind_group(
            &device,
            &kawase_bind_group_layout,
            &targets.kawase_quarter_view,
            &sampler,
            &kawase_uniform,
        );
        let kawase_up_half_bind_group = create_kawase_bind_group(
            &device,
            &kawase_bind_group_layout,
            &targets.kawase_up_half_view,
            &sampler,
            &kawase_uniform,
        );

        let (
            background_pipeline,
            kawase_downsample_pipeline,
            kawase_upsample_pipeline,
            shadow_pipeline,
            glass_pipeline,
        ) = create_pipelines(
            &device,
            &glass_bind_group_layout,
            &kawase_bind_group_layout,
            output_format,
        );
        let copy_bind_group_layout = create_copy_bind_group_layout(&device);
        let copy_bind_group = create_copy_bind_group(
            &device,
            &copy_bind_group_layout,
            &targets.output_view,
            &sampler,
        );
        let copy_pipeline =
            create_copy_pipeline(&device, &copy_bind_group_layout, output_format, None);
        let foreground_copy_pipeline = create_copy_pipeline(
            &device,
            &copy_bind_group_layout,
            output_format,
            Some(wgpu::BlendState::ALPHA_BLENDING),
        );
        let window_mask_bind_group_layout = create_window_mask_bind_group_layout(&device);
        let window_mask_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("liquid-glass window mask uniform"),
            size: u64::try_from(std::mem::size_of::<WindowMaskUniform>())
                .expect("window mask uniform size fits in u64"),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let window_mask_bind_group = create_window_mask_bind_group(
            &device,
            &window_mask_bind_group_layout,
            &targets.output_view,
            &sampler,
            &window_mask_uniform,
        );
        let window_mask_pipeline =
            create_window_mask_pipeline(&device, &window_mask_bind_group_layout, output_format);
        let vibrancy_tint_bind_group_layout = create_vibrancy_tint_bind_group_layout(&device);
        let vibrancy_tint_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("liquid-glass vibrancy tint uniform"),
            size: u64::try_from(std::mem::size_of::<VibrancyTintUniform>())
                .expect("vibrancy tint uniform size fits in u64"),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let vibrancy_tint_bind_group = create_vibrancy_tint_bind_group(
            &device,
            &vibrancy_tint_bind_group_layout,
            &targets.kawase_full_view,
            &sampler,
            &vibrancy_tint_uniform,
        );
        let vibrancy_tint_pipeline =
            create_vibrancy_tint_pipeline(&device, &vibrancy_tint_bind_group_layout, output_format);
        let gradient_composite_bind_group_layout =
            create_gradient_composite_bind_group_layout(&device);
        let gradient_composite_uniform_stride =
            uniform_stride_for_size(&device, std::mem::size_of::<GradientCompositeUniform>());
        let gradient_composite_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("liquid-glass gradient composite uniform"),
            size: u64::from(gradient_composite_uniform_stride) * MAX_GRADIENT_COMPOSITES as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let gradient_composite_bind_group = create_gradient_composite_bind_group(
            &device,
            &gradient_composite_bind_group_layout,
            &targets.kawase_full_view,
            &targets.scene_view,
            &sampler,
            &gradient_composite_uniform,
        );
        let gradient_composite_pipeline = create_gradient_composite_pipeline(
            &device,
            &gradient_composite_bind_group_layout,
            output_format,
        );

        Self {
            device,
            queue,
            size,
            output_format,
            targets,
            sampler,
            fullscreen_vertex_buffer,
            background_bind_group,
            kawase_bind_group_layout,
            kawase_down_from_scene_bind_group,
            kawase_down_from_output_bind_group,
            kawase_down_half_bind_group,
            kawase_down_quarter_bind_group,
            kawase_up_eighth_bind_group,
            kawase_up_quarter_bind_group,
            kawase_up_quarter_source_bind_group,
            kawase_up_half_bind_group,
            glass_bind_group_layout,
            glass_from_scene_bind_group,
            glass_from_output_bind_group,
            glass_uniform,
            kawase_uniform,
            kawase_uniform_stride,
            glass_uniform_stride,
            placeholder_texture,
            background_texture: None,
            background_texture_ratio: 1.0,
            background_pipeline,
            kawase_downsample_pipeline,
            kawase_upsample_pipeline,
            shadow_pipeline,
            glass_pipeline,
            copy_bind_group_layout,
            copy_bind_group,
            copy_pipeline,
            foreground_copy_pipeline,
            window_mask_bind_group_layout,
            window_mask_bind_group,
            window_mask_uniform,
            window_mask_pipeline,
            window_corner_radius: 0.0,
            vibrancy_tint_bind_group_layout,
            vibrancy_tint_bind_group,
            vibrancy_tint_uniform,
            vibrancy_tint_pipeline,
            gradient_composite_bind_group_layout,
            gradient_composite_bind_group,
            gradient_composite_uniform,
            gradient_composite_uniform_stride,
            gradient_composite_pipeline,
            transparent_background: false,
            options: GlassRenderOptions::default(),
        }
    }

    /// Recreates offscreen targets after a logical size change.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::InvalidSize`] when either target dimension is zero.
    pub fn resize(&mut self, size: GpuSize) -> Result<(), GpuError> {
        if !size.is_valid() {
            return Err(GpuError::InvalidSize);
        }
        self.size = size;
        self.targets = FrameTargets::new(&self.device, size, self.output_format);
        self.glass_from_scene_bind_group = create_glass_bind_group(
            &self.device,
            &self.glass_bind_group_layout,
            &self.targets.scene_view,
            &self.targets.kawase_full_view,
            &self.sampler,
            &self.glass_uniform,
        );
        self.glass_from_output_bind_group = create_glass_bind_group(
            &self.device,
            &self.glass_bind_group_layout,
            &self.targets.output_view,
            &self.targets.kawase_full_view,
            &self.sampler,
            &self.glass_uniform,
        );
        self.background_bind_group = create_background_bind_group(
            &self.device,
            &self.glass_bind_group_layout,
            self.background_texture.as_ref().unwrap_or(&self.placeholder_texture),
            &self.sampler,
            &self.glass_uniform,
        );
        self.kawase_down_from_scene_bind_group = create_kawase_bind_group(
            &self.device,
            &self.kawase_bind_group_layout,
            &self.targets.scene_view,
            &self.sampler,
            &self.kawase_uniform,
        );
        self.kawase_down_from_output_bind_group = create_kawase_bind_group(
            &self.device,
            &self.kawase_bind_group_layout,
            &self.targets.output_view,
            &self.sampler,
            &self.kawase_uniform,
        );
        self.copy_bind_group = create_copy_bind_group(
            &self.device,
            &self.copy_bind_group_layout,
            &self.targets.output_view,
            &self.sampler,
        );
        self.window_mask_bind_group = create_window_mask_bind_group(
            &self.device,
            &self.window_mask_bind_group_layout,
            &self.targets.output_view,
            &self.sampler,
            &self.window_mask_uniform,
        );
        self.update_window_mask_uniform();
        self.vibrancy_tint_bind_group = create_vibrancy_tint_bind_group(
            &self.device,
            &self.vibrancy_tint_bind_group_layout,
            &self.targets.kawase_full_view,
            &self.sampler,
            &self.vibrancy_tint_uniform,
        );
        self.kawase_down_half_bind_group = create_kawase_bind_group(
            &self.device,
            &self.kawase_bind_group_layout,
            &self.targets.kawase_half_view,
            &self.sampler,
            &self.kawase_uniform,
        );
        self.kawase_down_quarter_bind_group = create_kawase_bind_group(
            &self.device,
            &self.kawase_bind_group_layout,
            &self.targets.kawase_quarter_view,
            &self.sampler,
            &self.kawase_uniform,
        );
        self.kawase_up_eighth_bind_group = create_kawase_bind_group(
            &self.device,
            &self.kawase_bind_group_layout,
            &self.targets.kawase_eighth_view,
            &self.sampler,
            &self.kawase_uniform,
        );
        self.kawase_up_quarter_bind_group = create_kawase_bind_group(
            &self.device,
            &self.kawase_bind_group_layout,
            &self.targets.kawase_up_quarter_view,
            &self.sampler,
            &self.kawase_uniform,
        );
        self.kawase_up_quarter_source_bind_group = create_kawase_bind_group(
            &self.device,
            &self.kawase_bind_group_layout,
            &self.targets.kawase_quarter_view,
            &self.sampler,
            &self.kawase_uniform,
        );
        self.kawase_up_half_bind_group = create_kawase_bind_group(
            &self.device,
            &self.kawase_bind_group_layout,
            &self.targets.kawase_up_half_view,
            &self.sampler,
            &self.kawase_uniform,
        );
        self.gradient_composite_bind_group = create_gradient_composite_bind_group(
            &self.device,
            &self.gradient_composite_bind_group_layout,
            &self.targets.kawase_full_view,
            &self.targets.scene_view,
            &self.sampler,
            &self.gradient_composite_uniform,
        );
        Ok(())
    }

    /// Uses an externally decoded image as the reference project's backdrop.
    ///
    /// The texture must be created with [`wgpu::TextureUsages::TEXTURE_BINDING`]
    /// and use a filterable RGBA format.
    pub fn set_background_texture(&mut self, texture: wgpu::Texture, aspect_ratio: f32) {
        self.background_texture_ratio = aspect_ratio.max(f32::EPSILON);
        let bind_group = create_background_bind_group(
            &self.device,
            &self.glass_bind_group_layout,
            &texture,
            &self.sampler,
            &self.glass_uniform,
        );
        self.background_texture = Some(texture);
        self.background_bind_group = bind_group;
    }

    /// Replaces the renderer-wide accessibility policy.
    pub fn set_accessibility(&mut self, accessibility: GlassAccessibility) {
        self.options.accessibility = accessibility;
    }

    /// Replaces the renderer-wide backdrop environment estimate.
    pub fn set_environment(&mut self, environment: GlassEnvironment) {
        self.options.environment = environment;
    }

    /// Replaces all renderer-wide Liquid Glass options at once.
    pub fn set_render_options(&mut self, options: GlassRenderOptions) {
        self.options = options;
    }

    /// Returns the options used by subsequent composition calls.
    #[must_use]
    pub const fn render_options(&self) -> GlassRenderOptions {
        self.options
    }

    /// Starts a command-buffer batch for one composition phase.
    ///
    /// A platform integration can keep framework-owned rendering submissions
    /// between batches while collapsing all adjacent Liquid Glass operations
    /// into a single queue submission.
    #[must_use]
    pub fn begin_frame_batch(&self, label: Option<&str>) -> GpuFrameBatch<'_> {
        GpuFrameBatch {
            renderer: self,
            encoder: self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label }),
            next_uniform_slot: 0,
            gradient_slots_used: 0,
            scroll_edge_encoded: false,
        }
    }

    /// Uploads a platform-captured RGBA8 backdrop frame.
    ///
    /// The frame can contain padded rows (`stride > width * 4`). Padding is
    /// removed while preparing the GPU upload. The frame is sampled by the
    /// full reference refraction, dispersion, Fresnel, and vibrancy blur pipeline.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::InvalidBackgroundFrame`] when the dimensions,
    /// stride, or byte length do not describe a complete RGBA8 frame.
    ///
    /// This method does not panic for malformed frame metadata.
    #[allow(clippy::cast_precision_loss)]
    pub fn set_background_rgba8(
        &mut self,
        width: u32,
        height: u32,
        stride: u32,
        rgba8: &[u8],
    ) -> Result<(), GpuError> {
        let Some(tight_stride) = width.checked_mul(4) else {
            return Err(GpuError::InvalidBackgroundFrame {
                width,
                height,
                stride,
                byte_len: rgba8.len(),
            });
        };
        let Some(expected) = usize::try_from(stride).ok().and_then(|stride| {
            usize::try_from(height).ok().and_then(|height| stride.checked_mul(height))
        }) else {
            return Err(GpuError::InvalidBackgroundFrame {
                width,
                height,
                stride,
                byte_len: rgba8.len(),
            });
        };
        if width == 0 || height == 0 || stride < tight_stride || rgba8.len() != expected {
            return Err(GpuError::InvalidBackgroundFrame {
                width,
                height,
                stride,
                byte_len: rgba8.len(),
            });
        }

        let frame_height = height;
        let packed = if stride == tight_stride {
            None
        } else {
            let tight_stride = usize::try_from(tight_stride).map_err(|_| {
                GpuError::InvalidBackgroundFrame { width, height, stride, byte_len: rgba8.len() }
            })?;
            let height = usize::try_from(height).map_err(|_| GpuError::InvalidBackgroundFrame {
                width,
                height,
                stride,
                byte_len: rgba8.len(),
            })?;
            let tight_len =
                tight_stride.checked_mul(height).ok_or(GpuError::InvalidBackgroundFrame {
                    width,
                    height: u32::MAX,
                    stride,
                    byte_len: rgba8.len(),
                })?;
            let mut packed = Vec::with_capacity(tight_len);
            let stride = usize::try_from(stride).map_err(|_| GpuError::InvalidBackgroundFrame {
                width,
                height: frame_height,
                stride,
                byte_len: rgba8.len(),
            })?;
            for row in rgba8.chunks_exact(stride).take(height) {
                packed.extend_from_slice(&row[..tight_stride]);
            }
            Some(packed)
        };
        let pixels = packed.as_deref().unwrap_or(rgba8);
        self.options.environment = GlassEnvironment::from_rgba8(pixels);
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("liquid-glass platform backdrop texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(tight_stride),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
        self.set_background_texture(texture, width as f32 / height as f32);
        Ok(())
    }

    /// Controls whether the base pass writes an opaque background or leaves
    /// it transparent for the operating-system window compositor.
    ///
    /// A transparent frame still needs a backdrop source for local glass
    /// optics. Applications that want refraction of the real desktop should
    /// provide a platform-captured texture with [`Self::set_background_texture`]
    /// or [`Self::set_background_rgba8`]; window alpha alone can reveal the
    /// desktop, but it cannot make desktop pixels available to a custom shader.
    pub fn set_transparent_background(&mut self, transparent: bool) {
        self.transparent_background = transparent;
    }

    /// Sets the continuous-curvature mask applied to the final transparent
    /// window surface. A zero radius keeps the historical rectangular output.
    pub fn set_window_corner_radius(&mut self, radius: f32) {
        self.window_corner_radius = radius.max(0.0);
        self.update_window_mask_uniform();
    }

    fn update_window_mask_uniform(&self) {
        let uniform = WindowMaskUniform {
            size: [self.size.width as f32, self.size.height as f32],
            radius: self.window_corner_radius,
            exponent: liquid_glass_scene::CornerCurve::DEFAULT_CONTINUOUS_EXPONENT,
        };
        self.queue.write_buffer(&self.window_mask_uniform, 0, bytemuck::bytes_of(&uniform));
    }

    /// Encodes and submits one background, blur, and glass frame.
    ///
    /// # Panics
    ///
    /// This panics only if the renderer's compile-time scene node capacity is
    /// smaller than one node.
    pub fn render_panel(&self, node: &GlassNode, time_seconds: f32) {
        self.render_nodes_to_view(
            &self.targets.output_view,
            &[node],
            time_seconds,
            false,
            None,
            None,
            self.options,
        )
        .expect("a single glass node must fit in the scene uniform buffer");
    }

    /// Encodes a scene containing multiple glass nodes into the offscreen target.
    ///
    /// Nodes are drawn in ascending `z_index` order.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::SceneNodeLimitExceeded`] when the scene is larger
    /// than the renderer's dynamic uniform capacity.
    pub fn render_scene(&self, scene: &GlassScene, time_seconds: f32) -> Result<(), GpuError> {
        let nodes = scene.nodes_in_render_order();
        self.render_nodes_to_view(
            &self.targets.output_view,
            &nodes,
            time_seconds,
            false,
            None,
            None,
            scene.render_options().unwrap_or(self.options),
        )
    }

    /// Encodes a scene into an externally owned texture view.
    ///
    /// The external view must use the format passed to
    /// [`Self::from_device_with_format`].
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::SceneNodeLimitExceeded`] when the scene is larger
    /// than the renderer's dynamic uniform capacity.
    pub fn render_scene_to_view(
        &self,
        output_view: &wgpu::TextureView,
        scene: &GlassScene,
        time_seconds: f32,
    ) -> Result<(), GpuError> {
        let nodes = scene.nodes_in_render_order();
        self.render_nodes_to_view(
            output_view,
            &nodes,
            time_seconds,
            true,
            None,
            None,
            scene.render_options().unwrap_or(self.options),
        )
    }

    /// Encodes a scene into an externally owned view while using an external
    /// texture as the first compositing layer.
    ///
    /// The source texture must use the renderer's output format and include
    /// [`wgpu::TextureUsages::COPY_SRC`]. It is copied into the renderer's
    /// scene target before any glass node is evaluated, so every pixel drawn by
    /// the caller—including text and icons—can be refracted by the nodes.
    /// Nodes are still evaluated in ascending `z_index` order, and each later
    /// node samples the previous glass composite.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::SceneNodeLimitExceeded`] when the scene is larger
    /// than the renderer's dynamic uniform capacity.
    pub fn render_scene_to_view_with_source(
        &self,
        output_view: &wgpu::TextureView,
        source_texture: &wgpu::Texture,
        scene: &GlassScene,
        time_seconds: f32,
    ) -> Result<(), GpuError> {
        let nodes = scene.nodes_in_render_order();
        self.render_nodes_to_view(
            output_view,
            &nodes,
            time_seconds,
            true,
            Some(source_texture),
            None,
            scene.render_options().unwrap_or(self.options),
        )
    }

    /// Renders a scene over the renderer's current output instead of
    /// rebuilding it from an external source. This is used for glass controls
    /// that must sit above a post-composition layer such as a sidebar fade.
    pub fn render_scene_over_output(
        &self,
        output_view: &wgpu::TextureView,
        scene: &GlassScene,
        time_seconds: f32,
    ) -> Result<(), GpuError> {
        let nodes = scene.nodes_in_render_order();
        self.render_nodes_to_view(
            output_view,
            &nodes,
            time_seconds,
            true,
            Some(&self.targets.output),
            None,
            scene.render_options().unwrap_or(self.options),
        )
    }

    /// Encodes a scene using a plain rectangular Dual Kawase blur surface below
    /// the liquid-glass nodes.
    ///
    /// The blur region is deliberately not a [`GlassNode`]: it has no SDF,
    /// refraction, dispersion, Fresnel, glare, or liquid edge treatment. The
    /// tint alpha controls the amount of the uniform medium mixed over the
    /// blurred source.
    pub fn render_scene_to_view_with_source_and_blur_region(
        &self,
        output_view: &wgpu::TextureView,
        source_texture: &wgpu::Texture,
        blur_region: (u32, u32, u32, u32),
        blur_radius: u32,
        tint: [f32; 4],
        scene: &GlassScene,
        time_seconds: f32,
    ) -> Result<(), GpuError> {
        let nodes = scene.nodes_in_render_order();
        self.render_nodes_to_view(
            output_view,
            &nodes,
            time_seconds,
            true,
            Some(source_texture),
            Some(SimpleBlurRegion { bounds: blur_region, radius: blur_radius, tint }),
            scene.render_options().unwrap_or(self.options),
        )
    }

    /// Encodes and submits one frame into an externally owned texture view.
    ///
    /// # Panics
    ///
    /// This panics only if the renderer's compile-time scene node capacity is
    /// smaller than one node.
    pub fn render_panel_to_view(
        &self,
        output_view: &wgpu::TextureView,
        node: &GlassNode,
        time_seconds: f32,
    ) {
        self.render_nodes_to_view(
            output_view,
            &[node],
            time_seconds,
            true,
            None,
            None,
            self.options,
        )
        .expect("a single glass node must fit in the scene uniform buffer");
    }

    /// Renders the current backdrop texture into an external target view.
    ///
    /// This is used to seed an application-owned offscreen UI surface before
    /// the UI renderer draws its content over it. The resulting texture can
    /// then be passed to [`Self::render_scene_to_view_with_source`].
    pub fn render_background_to_view(&self, output_view: &wgpu::TextureView) {
        let mut batch = self.begin_frame_batch(Some("liquid-glass source background batch"));
        self.encode_background_to_view(&mut batch.encoder, output_view, 0);
        batch.submit();
    }

    fn encode_background_to_view(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        output_view: &wgpu::TextureView,
        uniform_slot: usize,
    ) {
        let node = GlassNode::new(
            liquid_glass_scene::GlassId(0),
            liquid_glass_scene::Rect::new(
                0.0,
                0.0,
                self.size.width as f32,
                self.size.height as f32,
            ),
        );
        let mut uniform = uniform_for_node(
            self.size,
            &node,
            0.0,
            self.background_texture.is_some(),
            self.background_texture_ratio,
            true,
            false,
            self.options,
        );
        uniform.bg_type = if self.background_texture.is_some() { 11 } else { 12 };
        let uniform_offset = self.glass_uniform_stride
            * u32::try_from(uniform_slot).expect("uniform slot fits in u32");
        self.queue.write_buffer(
            &self.glass_uniform,
            u64::from(uniform_offset),
            bytemuck::bytes_of(&uniform),
        );
        encode_fullscreen_pass(
            encoder,
            "liquid-glass external source background pass",
            output_view,
            &self.background_pipeline,
            &self.fullscreen_vertex_buffer,
            Some(&self.background_bind_group),
            Some(uniform_offset),
            wgpu::Color::BLACK,
        );
    }

    /// Fills a physical-pixel region of an external view with a flat color.
    ///
    /// This is useful for transparent-window demos that have an opaque app
    /// surface beside a translucent native desktop-glass surface. The color's
    /// alpha is preserved so the platform compositor can remain visible. The
    /// fill happens before the caller draws its source UI, so that source
    /// pixels remain available to the glass compositor while the final
    /// foreground pass stays clear.
    pub fn render_solid_region_to_view(
        &self,
        output_view: &wgpu::TextureView,
        region: (u32, u32, u32, u32),
        color: [f32; 4],
    ) {
        let mut batch = self.begin_frame_batch(Some("liquid-glass solid region batch"));
        self.encode_solid_region_to_view(&mut batch.encoder, output_view, region, color, 0);
        batch.submit();
    }

    fn encode_solid_region_to_view(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        output_view: &wgpu::TextureView,
        region: (u32, u32, u32, u32),
        color: [f32; 4],
        uniform_slot: usize,
    ) {
        let node = GlassNode::new(
            liquid_glass_scene::GlassId(0),
            liquid_glass_scene::Rect::new(
                0.0,
                0.0,
                self.size.width as f32,
                self.size.height as f32,
            ),
        );
        let mut uniform =
            uniform_for_node(self.size, &node, 0.0, false, 1.0, false, false, self.options);
        uniform.bg_type = 3;
        uniform.tint = srgb_to_linear_rgba(color);
        let uniform_offset = self.glass_uniform_stride
            * u32::try_from(uniform_slot).expect("uniform slot fits in u32");
        self.queue.write_buffer(
            &self.glass_uniform,
            u64::from(uniform_offset),
            bytemuck::bytes_of(&uniform),
        );
        encode_scissored_fullscreen_pass(
            encoder,
            "liquid-glass external solid region pass",
            output_view,
            &self.background_pipeline,
            &self.fullscreen_vertex_buffer,
            Some(&self.background_bind_group),
            Some(uniform_offset),
            wgpu::Color::TRANSPARENT,
            region,
        );
    }

    /// Alpha-composites an application-owned transparent layer over an
    /// already-composited glass view.
    ///
    /// The source texture must use the renderer's output format and have the
    /// same dimensions as the renderer. Pixels with zero alpha leave the
    /// existing glass result untouched.
    pub fn composite_texture_to_view(
        &self,
        output_view: &wgpu::TextureView,
        source_texture: &wgpu::Texture,
    ) {
        let source_view = source_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = create_copy_bind_group(
            &self.device,
            &self.copy_bind_group_layout,
            &source_view,
            &self.sampler,
        );
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("liquid-glass foreground composite encoder"),
        });
        encode_fullscreen_load_pass(
            &mut encoder,
            "liquid-glass foreground composite pass",
            output_view,
            &self.foreground_copy_pipeline,
            &self.fullscreen_vertex_buffer,
            &bind_group,
        );
        self.queue.submit([encoder.finish()]);
    }

    /// Alpha-composites an application-owned transparent layer into the
    /// renderer's ping-pong output before a later post-composition pass.
    pub fn composite_texture_to_output(&self, source_texture: &wgpu::Texture) {
        let mut batch = self.begin_frame_batch(Some("liquid-glass foreground composite batch"));
        batch.composite_texture_to_output(source_texture);
        batch.submit();
    }

    fn encode_composite_texture_to_output(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source_texture: &wgpu::Texture,
    ) {
        let source_view = source_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = create_copy_bind_group(
            &self.device,
            &self.copy_bind_group_layout,
            &source_view,
            &self.sampler,
        );
        encode_fullscreen_load_pass(
            encoder,
            "liquid-glass internal foreground composite pass",
            &self.targets.output_view,
            &self.foreground_copy_pipeline,
            &self.fullscreen_vertex_buffer,
            &bind_group,
        );
    }

    /// Copies the renderer's final internal output into an external target.
    pub fn copy_output_to_view(&self, output_view: &wgpu::TextureView) {
        let mut batch = self.begin_frame_batch(Some("liquid-glass output copy batch"));
        batch.copy_output_to_view(output_view);
        batch.submit();
    }

    fn encode_copy_output_to_view(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        output_view: &wgpu::TextureView,
    ) {
        if self.window_corner_radius > 0.0 {
            encode_fullscreen_pass(
                encoder,
                "liquid-glass final rounded window mask pass",
                output_view,
                &self.window_mask_pipeline,
                &self.fullscreen_vertex_buffer,
                Some(&self.window_mask_bind_group),
                None,
                wgpu::Color::TRANSPARENT,
            );
        } else {
            let source_view =
                self.targets.output.create_view(&wgpu::TextureViewDescriptor::default());
            let bind_group = create_copy_bind_group(
                &self.device,
                &self.copy_bind_group_layout,
                &source_view,
                &self.sampler,
            );
            encode_fullscreen_load_pass(
                encoder,
                "liquid-glass final output copy pass",
                output_view,
                &self.copy_pipeline,
                &self.fullscreen_vertex_buffer,
                &bind_group,
            );
        }
    }

    /// Applies a uniformly blurred overlay whose opacity fades from top to
    /// bottom. The blur uses `maximum_radius`; `minimum_radius` is retained
    /// for compatibility with the earlier radius-gradient implementation.
    pub fn render_vertical_gradient_blur_to_output(
        &self,
        region: (u32, u32, u32, u32),
        minimum_radius: u32,
        maximum_radius: u32,
        fallback_color: [f32; 4],
    ) {
        self.render_vertical_gradient_blur_to_output_with_source(
            &self.targets.output,
            region,
            region.1,
            minimum_radius,
            maximum_radius,
            fallback_color,
            0.0,
            0,
        );
    }

    /// Applies a uniformly blurred overlay with a flat opaque top, followed
    /// by a top-to-bottom alpha fade. This models a floating search field:
    /// content above its upper edge stays strongly blurred, while the area
    /// between its upper and lower edges fades to the clear list below.
    pub fn render_vertical_blur_with_flat_top_to_output(
        &self,
        region: (u32, u32, u32, u32),
        fade_start_y: u32,
        maximum_radius: u32,
        fallback_color: [f32; 4],
    ) {
        self.render_vertical_gradient_blur_to_output_with_source(
            &self.targets.output,
            region,
            fade_start_y,
            0,
            maximum_radius,
            fallback_color,
            0.0,
            0,
        );
    }

    /// Applies a reusable scroll-edge treatment to the current output.
    ///
    /// The top-edge form is intentionally expressed in physical pixels so a
    /// platform adapter can keep the fade aligned with a clipped scroll
    /// viewport. `Soft` fades the blur across the region; `Hard` keeps the
    /// entire region blurred and ends it at the region boundary.
    pub fn render_scroll_edge_to_output(
        &self,
        region: (u32, u32, u32, u32),
        fade_start_y: u32,
        maximum_radius: u32,
        fallback_color: [f32; 4],
        style: ScrollEdgeStyle,
    ) {
        let fade_start_y = match style {
            ScrollEdgeStyle::Soft => fade_start_y,
            ScrollEdgeStyle::Hard => region.1.saturating_add(region.3),
        };
        self.render_vertical_gradient_blur_to_output_with_source(
            &self.targets.output,
            region,
            fade_start_y,
            0,
            maximum_radius,
            fallback_color,
            0.0,
            0,
        );
    }

    /// Encodes the Dual Kawase pyramid supplied by `vibrancy-rs`.
    ///
    /// The pyramid is shared by glass nodes and the sidebar edge treatment.
    /// Each logical blur slot owns eight uniform slots, so all passes in one
    /// command buffer observe the parameters that were present when they were
    /// encoded, even when several glass layers are composed back-to-front.
    fn encode_kawase_blur(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: PingPongSource,
        radius: u32,
        logical_slot: usize,
    ) {
        let plan = vibrancy_rs::KawasePassPlan::new(
            self.size.width as i32,
            self.size.height as i32,
            radius as f32,
        );
        let source_bind_group = match source {
            PingPongSource::Scene => &self.kawase_down_from_scene_bind_group,
            PingPongSource::Output => &self.kawase_down_from_output_bind_group,
        };
        let half_size = plan.src_size;
        let quarter_size = plan.down1_size;
        write_kawase_uniform(
            &self.queue,
            &self.kawase_uniform,
            self.kawase_uniform_stride,
            logical_slot,
            0,
            vibrancy_rs::compute_halfpixel(half_size.0, half_size.1),
            plan.offset,
        );
        encode_kawase_pass(
            encoder,
            "liquid-glass vibrancy downsample source",
            &self.targets.kawase_half_view,
            &self.kawase_downsample_pipeline,
            &self.fullscreen_vertex_buffer,
            source_bind_group,
            kawase_dynamic_offset(self.kawase_uniform_stride, logical_slot, 0),
        );

        write_kawase_uniform(
            &self.queue,
            &self.kawase_uniform,
            self.kawase_uniform_stride,
            logical_slot,
            1,
            plan.down1_halfpixel,
            plan.offset,
        );
        encode_kawase_pass(
            encoder,
            "liquid-glass vibrancy downsample half",
            &self.targets.kawase_quarter_view,
            &self.kawase_downsample_pipeline,
            &self.fullscreen_vertex_buffer,
            &self.kawase_down_half_bind_group,
            kawase_dynamic_offset(self.kawase_uniform_stride, logical_slot, 1),
        );

        if let (Some(down2_size), Some(down2_halfpixel)) = (plan.down2_size, plan.down2_halfpixel) {
            write_kawase_uniform(
                &self.queue,
                &self.kawase_uniform,
                self.kawase_uniform_stride,
                logical_slot,
                2,
                down2_halfpixel,
                plan.offset,
            );
            encode_kawase_pass(
                encoder,
                "liquid-glass vibrancy downsample quarter",
                &self.targets.kawase_eighth_view,
                &self.kawase_downsample_pipeline,
                &self.fullscreen_vertex_buffer,
                &self.kawase_down_quarter_bind_group,
                kawase_dynamic_offset(self.kawase_uniform_stride, logical_slot, 2),
            );

            write_kawase_uniform(
                &self.queue,
                &self.kawase_uniform,
                self.kawase_uniform_stride,
                logical_slot,
                3,
                vibrancy_rs::compute_halfpixel(down2_size.0, down2_size.1),
                plan.offset,
            );
            encode_kawase_pass(
                encoder,
                "liquid-glass vibrancy upsample eighth",
                &self.targets.kawase_up_quarter_view,
                &self.kawase_upsample_pipeline,
                &self.fullscreen_vertex_buffer,
                &self.kawase_up_eighth_bind_group,
                kawase_dynamic_offset(self.kawase_uniform_stride, logical_slot, 3),
            );
        }

        write_kawase_uniform(
            &self.queue,
            &self.kawase_uniform,
            self.kawase_uniform_stride,
            logical_slot,
            4,
            vibrancy_rs::compute_halfpixel(quarter_size.0, quarter_size.1),
            plan.offset,
        );
        encode_kawase_pass(
            encoder,
            "liquid-glass vibrancy upsample quarter",
            &self.targets.kawase_up_half_view,
            &self.kawase_upsample_pipeline,
            &self.fullscreen_vertex_buffer,
            if plan.use_deep_blur {
                &self.kawase_up_quarter_bind_group
            } else {
                &self.kawase_up_quarter_source_bind_group
            },
            kawase_dynamic_offset(self.kawase_uniform_stride, logical_slot, 4),
        );

        write_kawase_uniform(
            &self.queue,
            &self.kawase_uniform,
            self.kawase_uniform_stride,
            logical_slot,
            5,
            vibrancy_rs::compute_halfpixel(half_size.0, half_size.1),
            plan.offset,
        );
        encode_kawase_pass(
            encoder,
            "liquid-glass vibrancy upsample half",
            &self.targets.kawase_full_view,
            &self.kawase_upsample_pipeline,
            &self.fullscreen_vertex_buffer,
            &self.kawase_up_half_bind_group,
            kawase_dynamic_offset(self.kawase_uniform_stride, logical_slot, 5),
        );
    }

    fn render_vertical_gradient_blur_to_output_with_source(
        &self,
        source_texture: &wgpu::Texture,
        region: (u32, u32, u32, u32),
        fade_start_y: u32,
        minimum_radius: u32,
        maximum_radius: u32,
        fallback_color: [f32; 4],
        minimum_blur_mix: f32,
        gradient_slot: usize,
    ) {
        let mut batch = self.begin_frame_batch(Some("liquid-glass scroll edge batch"));
        self.encode_vertical_gradient_blur_to_output_with_source(
            &mut batch.encoder,
            source_texture,
            region,
            fade_start_y,
            minimum_radius,
            maximum_radius,
            fallback_color,
            minimum_blur_mix,
            gradient_slot,
        );
        batch.submit();
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_vertical_gradient_blur_to_output_with_source(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source_texture: &wgpu::Texture,
        region: (u32, u32, u32, u32),
        fade_start_y: u32,
        _minimum_radius: u32,
        maximum_radius: u32,
        fallback_color: [f32; 4],
        minimum_blur_mix: f32,
        gradient_slot: usize,
    ) {
        let x = region.0.min(self.size.width);
        let y = region.1.min(self.size.height);
        let width = region.2.min(self.size.width.saturating_sub(x));
        let height = region.3.min(self.size.height.saturating_sub(y));
        if width == 0 || height == 0 {
            return;
        }

        let maximum_radius = maximum_radius.min(128);
        let fade_start_y = fade_start_y.clamp(y, y + height);
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: source_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &self.targets.scene,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.size.width,
                height: self.size.height,
                depth_or_array_layers: 1,
            },
        );
        self.encode_kawase_blur(
            encoder,
            PingPongSource::Scene,
            maximum_radius,
            MAX_GLASS_NODES + gradient_slot,
        );

        self.queue.write_buffer(
            &self.gradient_composite_uniform,
            u64::from(self.gradient_composite_uniform_stride)
                * u64::try_from(gradient_slot).expect("gradient slot fits in u64"),
            bytemuck::bytes_of(&GradientCompositeUniform {
                fallback: srgb_to_linear_rgba(fallback_color),
                output_size: [self.size.width as f32, self.size.height as f32],
                _pad: [0.0; 2],
                gradient: [
                    fade_start_y as f32,
                    (y + height) as f32,
                    minimum_blur_mix.clamp(0.0, 1.0),
                    0.0,
                ],
            }),
        );
        let scissor = (x, y, width, height);
        // Preserve the unblurred output for the final mix. This prevents
        // transparent or not-yet-captured desktop pixels from turning into a
        // black rectangle when the gradient is applied.
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.targets.output,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &self.targets.scene,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
        encode_scissored_fullscreen_pass(
            encoder,
            "liquid-glass vertical gradient composite pass",
            &self.targets.output_view,
            &self.gradient_composite_pipeline,
            &self.fullscreen_vertex_buffer,
            Some(&self.gradient_composite_bind_group),
            Some(
                self.gradient_composite_uniform_stride
                    * u32::try_from(gradient_slot).expect("gradient slot fits in u32"),
            ),
            wgpu::Color::TRANSPARENT,
            scissor,
        );
    }

    /// Renders a scene directly to a configured `wgpu` `SurfaceTexture`.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::SceneNodeLimitExceeded`] when the scene is larger
    /// than the renderer's dynamic uniform capacity.
    pub fn render_scene_to_surface_texture(
        &self,
        surface_texture: wgpu::SurfaceTexture,
        scene: &GlassScene,
        time_seconds: f32,
    ) -> Result<(), GpuError> {
        let view = surface_texture.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let result = self.render_scene_to_view(&view, scene, time_seconds);
        if result.is_ok() {
            surface_texture.present();
        }
        result
    }

    #[allow(clippy::too_many_lines)]
    fn render_nodes_to_view(
        &self,
        output_view: &wgpu::TextureView,
        nodes: &[&GlassNode],
        time_seconds: f32,
        copy_to_external_view: bool,
        source_texture: Option<&wgpu::Texture>,
        simple_blur: Option<SimpleBlurRegion>,
        options: GlassRenderOptions,
    ) -> Result<(), GpuError> {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("liquid-glass frame encoder"),
        });
        self.encode_nodes_to_view(
            &mut encoder,
            output_view,
            nodes,
            time_seconds,
            copy_to_external_view,
            source_texture,
            simple_blur,
            options,
            0,
        )?;
        self.queue.submit([encoder.finish()]);
        Ok(())
    }

    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn encode_nodes_to_view(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        output_view: &wgpu::TextureView,
        nodes: &[&GlassNode],
        time_seconds: f32,
        copy_to_external_view: bool,
        source_texture: Option<&wgpu::Texture>,
        simple_blur: Option<SimpleBlurRegion>,
        options: GlassRenderOptions,
        first_uniform_slot: usize,
    ) -> Result<(), GpuError> {
        let Some(uniform_slot_end) = first_uniform_slot.checked_add(nodes.len()) else {
            return Err(GpuError::SceneNodeLimitExceeded { limit: MAX_GLASS_NODES });
        };
        if uniform_slot_end > MAX_GLASS_NODES {
            return Err(GpuError::SceneNodeLimitExceeded { limit: MAX_GLASS_NODES });
        }

        for (index, node) in nodes.iter().enumerate() {
            let uniform_slot = first_uniform_slot + index;
            let offset = u64::from(self.glass_uniform_stride)
                * u64::try_from(uniform_slot).expect("scene node index fits in u64");
            let uniform = uniform_for_node(
                self.size,
                node,
                time_seconds,
                self.background_texture.is_some(),
                self.background_texture_ratio,
                self.transparent_background,
                source_texture.is_some() || index > 0,
                options,
            );
            self.queue.write_buffer(&self.glass_uniform, offset, bytemuck::bytes_of(&uniform));
        }
        if let Some(source_texture) = source_texture {
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: source_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &self.targets.scene,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: self.size.width,
                    height: self.size.height,
                    depth_or_array_layers: 1,
                },
            );
        } else {
            encode_fullscreen_pass(
                encoder,
                "liquid-glass scene pass",
                &self.targets.scene_view,
                &self.background_pipeline,
                &self.fullscreen_vertex_buffer,
                Some(&self.background_bind_group),
                Some(0),
                wgpu::Color { r: 0.04, g: 0.06, b: 0.12, a: 1.0 },
            );
        }

        let mut source = PingPongSource::Scene;
        if let Some(simple_blur) = simple_blur {
            let x = simple_blur.bounds.0.min(self.size.width);
            let y = simple_blur.bounds.1.min(self.size.height);
            let width = simple_blur.bounds.2.min(self.size.width.saturating_sub(x));
            let height = simple_blur.bounds.3.min(self.size.height.saturating_sub(y));
            if width > 0 && height > 0 {
                self.encode_kawase_blur(
                    encoder,
                    PingPongSource::Scene,
                    simple_blur.radius,
                    MAX_GLASS_NODES,
                );
                self.queue.write_buffer(
                    &self.vibrancy_tint_uniform,
                    0,
                    bytemuck::bytes_of(&VibrancyTintUniform {
                        tint: srgb_to_linear_rgba(simple_blur.tint),
                    }),
                );
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &self.targets.scene,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: &self.targets.output,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: self.size.width,
                        height: self.size.height,
                        depth_or_array_layers: 1,
                    },
                );
                encode_scissored_fullscreen_pass(
                    encoder,
                    "liquid-glass vibrancy tint pass",
                    &self.targets.output_view,
                    &self.vibrancy_tint_pipeline,
                    &self.fullscreen_vertex_buffer,
                    Some(&self.vibrancy_tint_bind_group),
                    None,
                    wgpu::Color::TRANSPARENT,
                    (x, y, width, height),
                );
                source = PingPongSource::Output;
            }
        }
        for (index, _) in nodes.iter().enumerate() {
            let uniform_slot = first_uniform_slot + index;
            let blur_radius = blur_radius_for_node(nodes[index], options);
            // The shape remains bounded by the node's SDF, but the draw pass
            // must extend beyond it so cast shadows and edge light are not
            // clipped at the interactive control rectangle.
            let Some(effect_region) = node_effect_region(nodes[index], self.size) else {
                continue;
            };
            let uniform_offset = self.glass_uniform_stride
                * u32::try_from(uniform_slot).expect("node index fits in u32");
            let (
                source_texture,
                source_view,
                destination_texture,
                destination_view,
                shadow_bind_group,
                glass_bind_group,
            ) = match source {
                PingPongSource::Scene => (
                    &self.targets.scene,
                    &self.targets.scene_view,
                    &self.targets.output,
                    &self.targets.output_view,
                    &self.glass_from_scene_bind_group,
                    &self.glass_from_output_bind_group,
                ),
                PingPongSource::Output => (
                    &self.targets.output,
                    &self.targets.output_view,
                    &self.targets.scene,
                    &self.targets.scene_view,
                    &self.glass_from_output_bind_group,
                    &self.glass_from_scene_bind_group,
                ),
            };
            self.encode_kawase_blur(encoder, source, blur_radius as u32, uniform_slot);
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: source_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: destination_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: self.size.width,
                    height: self.size.height,
                    depth_or_array_layers: 1,
                },
            );
            // Build the glass from the clean backdrop first. Applying the
            // shadow before this point would make the blur sample the shadow
            // and bleed it back into the material as an artificial inner
            // dark band.
            encode_glass_node_pass(
                encoder,
                source_view,
                &self.glass_pipeline,
                glass_bind_group,
                &self.fullscreen_vertex_buffer,
                uniform_offset,
                effect_region,
            );
            // The remaining shadow is a faint SDF-derived elevation tail on
            // the finished composite. It is deliberately kept out of the
            // backdrop blur and refraction inputs, so it cannot look like a
            // second translucent surface inside the control.
            encode_glass_node_pass(
                encoder,
                destination_view,
                &self.shadow_pipeline,
                shadow_bind_group,
                &self.fullscreen_vertex_buffer,
                uniform_offset,
                effect_region,
            );
            source = match source {
                PingPongSource::Scene => PingPongSource::Output,
                PingPongSource::Output => PingPongSource::Scene,
            };
        }

        if source == PingPongSource::Scene {
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.targets.scene,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &self.targets.output,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: self.size.width,
                    height: self.size.height,
                    depth_or_array_layers: 1,
                },
            );
        }
        if copy_to_external_view {
            encode_fullscreen_pass(
                encoder,
                "liquid-glass present copy pass",
                output_view,
                &self.copy_pipeline,
                &self.fullscreen_vertex_buffer,
                Some(&self.copy_bind_group),
                None,
                wgpu::Color::BLACK,
            );
        }
        Ok(())
    }

    /// Renders one frame directly to a configured `wgpu` `SurfaceTexture` and presents it.
    pub fn render_panel_to_surface_texture(
        &self,
        surface_texture: wgpu::SurfaceTexture,
        node: &GlassNode,
        time_seconds: f32,
    ) {
        let view = surface_texture.texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.render_panel_to_view(&view, node, time_seconds);
        surface_texture.present();
    }

    /// Returns the final texture for a future surface/present pass.
    #[must_use]
    pub const fn output_texture(&self) -> &wgpu::Texture {
        &self.targets.output
    }

    /// Returns the dimensions of the current offscreen target.
    #[must_use]
    pub const fn size(&self) -> GpuSize {
        self.size
    }

    /// Returns the device used to create the renderer's resources.
    #[must_use]
    pub const fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Returns the texture format targeted by the render pipelines.
    #[must_use]
    pub const fn output_format(&self) -> wgpu::TextureFormat {
        self.output_format
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_fullscreen_pass(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    target: &wgpu::TextureView,
    pipeline: &wgpu::RenderPipeline,
    vertex_buffer: &wgpu::Buffer,
    bind_group: Option<&wgpu::BindGroup>,
    dynamic_offset: Option<u32>,
    clear: wgpu::Color,
) {
    encode_fullscreen_pass_with_load(
        encoder,
        label,
        target,
        pipeline,
        vertex_buffer,
        bind_group,
        dynamic_offset,
        wgpu::LoadOp::Clear(clear),
    );
}

fn encode_fullscreen_load_pass(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    target: &wgpu::TextureView,
    pipeline: &wgpu::RenderPipeline,
    vertex_buffer: &wgpu::Buffer,
    bind_group: &wgpu::BindGroup,
) {
    encode_fullscreen_pass_with_load(
        encoder,
        label,
        target,
        pipeline,
        vertex_buffer,
        Some(bind_group),
        None,
        wgpu::LoadOp::Load,
    );
}

#[allow(clippy::too_many_arguments)]
fn encode_fullscreen_pass_with_load(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    target: &wgpu::TextureView,
    pipeline: &wgpu::RenderPipeline,
    vertex_buffer: &wgpu::Buffer,
    bind_group: Option<&wgpu::BindGroup>,
    dynamic_offset: Option<u32>,
    load: wgpu::LoadOp<wgpu::Color>,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations { load, store: wgpu::StoreOp::Store },
        })],
        ..Default::default()
    });
    pass.set_pipeline(pipeline);
    pass.set_vertex_buffer(0, vertex_buffer.slice(..));
    if let Some(bind_group) = bind_group {
        if let Some(dynamic_offset) = dynamic_offset {
            pass.set_bind_group(0, bind_group, &[dynamic_offset]);
        } else {
            pass.set_bind_group(0, bind_group, &[]);
        }
    }
    pass.draw(0..4, 0..1);
}

fn encode_kawase_pass(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    target: &wgpu::TextureView,
    pipeline: &wgpu::RenderPipeline,
    vertex_buffer: &wgpu::Buffer,
    bind_group: &wgpu::BindGroup,
    dynamic_offset: u32,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
        })],
        ..Default::default()
    });
    pass.set_pipeline(pipeline);
    pass.set_vertex_buffer(0, vertex_buffer.slice(..));
    pass.set_bind_group(0, bind_group, &[dynamic_offset]);
    pass.draw(0..4, 0..1);
}

#[allow(clippy::too_many_arguments)]
fn encode_scissored_fullscreen_pass(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    target: &wgpu::TextureView,
    pipeline: &wgpu::RenderPipeline,
    vertex_buffer: &wgpu::Buffer,
    bind_group: Option<&wgpu::BindGroup>,
    dynamic_offset: Option<u32>,
    clear_color: wgpu::Color,
    region: (u32, u32, u32, u32),
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
        })],
        ..Default::default()
    });
    pass.set_pipeline(pipeline);
    pass.set_vertex_buffer(0, vertex_buffer.slice(..));
    pass.set_scissor_rect(region.0, region.1, region.2.max(1), region.3.max(1));
    if let Some(bind_group) = bind_group {
        if let Some(dynamic_offset) = dynamic_offset {
            pass.set_bind_group(0, bind_group, &[dynamic_offset]);
        } else {
            pass.set_bind_group(0, bind_group, &[]);
        }
    }
    let _ = clear_color;
    pass.draw(0..4, 0..1);
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct SimpleBlurRegion {
    bounds: (u32, u32, u32, u32),
    radius: u32,
    tint: [f32; 4],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PingPongSource {
    Scene,
    Output,
}

fn encode_glass_node_pass(
    encoder: &mut wgpu::CommandEncoder,
    target: &wgpu::TextureView,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    vertex_buffer: &wgpu::Buffer,
    uniform_offset: u32,
    region: (u32, u32, u32, u32),
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("liquid-glass pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
        })],
        ..Default::default()
    });
    pass.set_pipeline(pipeline);
    pass.set_vertex_buffer(0, vertex_buffer.slice(..));
    pass.set_scissor_rect(region.0, region.1, region.2.max(1), region.3.max(1));
    pass.set_bind_group(0, bind_group, &[uniform_offset]);
    pass.draw(0..4, 0..1);
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn node_effect_region(node: &GlassNode, size: GpuSize) -> Option<(u32, u32, u32, u32)> {
    let shadow = node.material.shadow;
    let padding = shadow.expand.max(0.0) + shadow.offset[0].abs().max(shadow.offset[1].abs()) + 4.0;
    let bounds = node_optical_bounds(node, size);
    let left = (bounds.x - padding).max(0.0).floor() as u32;
    let top = (bounds.y - padding).max(0.0).floor() as u32;
    let right = (bounds.x + bounds.width + padding).max(0.0).ceil() as u32;
    let bottom = (bounds.y + bounds.height + padding).max(0.0).ceil() as u32;
    let left = left.min(size.width);
    let top = top.min(size.height);
    let right = right.min(size.width);
    let bottom = bottom.min(size.height);
    let width = right.saturating_sub(left);
    let height = bottom.saturating_sub(top);
    (width > 0 && height > 0).then_some((left, top, width, height))
}

/// Bounds every shape evaluated by `mainSDF`, including the fixed 200px
/// reference circle used by the source fusion demo. Scissoring only to the
/// primary node used to clip most of that circle and leave a dark sliver at
/// the merge neck, which made the enhanced path appear to have lost fusion.
#[allow(clippy::cast_precision_loss)]
fn node_optical_bounds(node: &GlassNode, size: GpuSize) -> Rect {
    let bounds = node.visual_bounds();
    if !node.material.show_shape1 {
        return bounds;
    }

    let reference_circle =
        Rect::new(size.width as f32 * 0.5 - 100.0, size.height as f32 * 0.5 - 100.0, 200.0, 200.0);
    let left = bounds.x.min(reference_circle.x);
    let top = bounds.y.min(reference_circle.y);
    let right = (bounds.x + bounds.width).max(reference_circle.x + reference_circle.width);
    let bottom = (bounds.y + bounds.height).max(reference_circle.y + reference_circle.height);
    Rect::new(left, top, right - left, bottom - top)
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, clippy::cast_sign_loss)]
fn uniform_for_node(
    size: GpuSize,
    node: &GlassNode,
    _time_seconds: f32,
    background_texture_ready: bool,
    background_texture_ratio: f32,
    transparent_background: bool,
    has_composited_source: bool,
    options: GlassRenderOptions,
) -> GlassUniform {
    let center_x = node.bounds.x + node.bounds.width * 0.5;
    let center_y = size.height as f32 - node.bounds.y - node.bounds.height * 0.5;
    let material = node.material;
    let accessibility = options.accessibility;
    let size_gain = size_response(node, material.adaptive.size);
    let mut feature_flags = 0;
    if material.blur.edge_blur {
        feature_flags |= FEATURE_EDGE_BLUR;
    }
    if accessibility.reduced_transparency {
        feature_flags |= FEATURE_REDUCED_TRANSPARENCY;
    }
    if accessibility.increased_contrast {
        feature_flags |= FEATURE_INCREASED_CONTRAST;
    }
    if accessibility.reduced_motion {
        feature_flags |= FEATURE_REDUCED_MOTION;
    }
    if material.variant == GlassVariant::Clear {
        feature_flags |= FEATURE_CLEAR_VARIANT;
    }
    if material.variant == GlassVariant::TrafficLightBead {
        // Shares the traffic-light antialiasing, glyph treatment, and the exact
        // circular SDF path the physical variant used -- that path is what makes
        // the silhouette a whole circle rather than a slice of the generic
        // capsule. The shader still returns early instead of composing glass.
        feature_flags |=
            FEATURE_TRAFFIC_LIGHT | FEATURE_TRAFFIC_LIGHT_PHYSICAL | FEATURE_TRAFFIC_LIGHT_BEAD;
    }
    let interaction = if accessibility.reduced_motion { 0.0 } else { node.interaction.strength() };
    let pointer_x = node.bounds.x + node.bounds.width * node.interaction.pointer[0];
    let pointer_y =
        size.height as f32 - node.bounds.y - node.bounds.height * node.interaction.pointer[1];
    let spring_pointer = if accessibility.reduced_motion {
        node.interaction.pointer
    } else {
        node.interaction.spring
    };
    let spring_x = node.bounds.x + node.bounds.width * spring_pointer[0];
    let spring_y = size.height as f32 - node.bounds.y - node.bounds.height * spring_pointer[1];
    let variant_factor = if material.variant == GlassVariant::Clear { 0.78 } else { 1.0 };
    let refraction_strength = if accessibility.reduced_transparency {
        0.0
    } else {
        material.refraction.strength * size_gain * variant_factor
    };
    let dispersion_strength = if accessibility.reduced_transparency {
        0.0
    } else {
        material.dispersion.strength * variant_factor
    };
    let fresnel_strength = if accessibility.reduced_transparency {
        material.fresnel.strength * 0.35
    } else {
        material.fresnel.strength * variant_factor
    };
    let blur_radius = blur_radius_for_node(node, options);
    let tint = tint_with_whiteness(material);
    let shape_radius = shape_radius(node);
    let shape_roundness = shape_roundness(node);
    let (fused_bounds, fused_geometry) = fused_shape_uniforms(size, node);
    GlassUniform {
        resolution_dpr_pad: [
            size.width as f32,
            size.height as f32,
            1.0,
            options.scale_factor.max(1.0),
        ],
        mouse_and_spring: [pointer_x, pointer_y, center_x, center_y],
        shape: [node.bounds.width, node.bounds.height, shape_radius, shape_roundness],
        capsule_bezier_x: [0.0; 4],
        capsule_bezier_y: [0.0; 4],
        merge_glare_shadow: [
            material.merge_rate.max(f32::EPSILON),
            // This legacy glare-angle slot is unused by the fixed lighting
            // model. Reuse it for press strength without changing the
            // uniform buffer layout or shifting every following field.
            node.interaction.press.clamp(0.0, 1.0),
            material.shadow.expand.max(1.0),
            (material.shadow.factor * material.adaptive.shadow * size_gain * variant_factor)
                .clamp(0.0, 0.6),
        ],
        shadow_position_bg_ratio: [
            material.shadow.offset[0],
            material.shadow.offset[1],
            background_texture_ratio,
        ],
        bg_type: if transparent_background {
            if has_composited_source { 13 } else { 12 }
        } else if background_texture_ready {
            11
        } else {
            0
        },
        flags: [
            i32::from(background_texture_ready),
            i32::from(material.show_shape1),
            blur_radius,
            feature_flags,
        ],
        tint,
        refraction_and_fresnel: [
            (material.refraction.thickness * 100.0).max(1.0),
            material.refraction.index,
            (dispersion_strength * 100.0).max(0.0),
            (material.fresnel.range * 40.0).max(1.0),
            material.fresnel.hardness,
            fresnel_strength,
        ],
        glare: [
            material.glare.range,
            material.glare.hardness,
            material.glare.convergence,
            material.glare.opposite_factor,
            material.glare.factor,
        ],
        // The final five floats are kept as a 16-byte-aligned tail in the
        // uniform. Use the first two instead of dropping refraction.strength:
        // the reference shader's offset is otherwise effectively hard-coded
        // and small controls look like plain translucent pills.
        _pad: [
            refraction_strength,
            if accessibility.reduced_transparency {
                material.opacity.max(0.92)
            } else {
                material.opacity
            },
            interaction,
            options.environment.luminance.clamp(0.0, 1.0),
            options.environment.contrast.clamp(0.0, 1.0),
        ],
        adaptive: [
            material.adaptive.tint.clamp(0.0, 2.0),
            material.adaptive.ambient.clamp(0.0, 2.0),
            material.adaptive.shadow.clamp(0.0, 2.0),
            material.adaptive.size.clamp(0.0, 2.0),
        ],
        fused_bounds,
        fused_geometry,
        interaction_state: [spring_x, spring_y, node.interaction.parallax, node.interaction.focus],
        traffic_light: [
            material.traffic_light.substrate_coverage.clamp(0.0, 1.0),
            material.traffic_light.lower_substrate_coverage.clamp(0.0, 1.0),
            material.traffic_light.lower_tint_coverage.clamp(0.0, 1.0),
            material.traffic_light.angular_light.clamp(0.0, 2.0),
        ],
        traffic_light_light: [
            material.traffic_light.light_angle.clamp(0.0, 1.0),
            material.traffic_light.body_thickness.clamp(0.25, 3.0),
            material.traffic_light.internal_scattering.clamp(0.0, 1.0),
            material.traffic_light.side_edge_darkness.clamp(0.0, 4.0),
        ],
        traffic_light_edge: [
            material.traffic_light.side_edge_width.clamp(0.25, 4.0),
            material.traffic_light.light_softness.clamp(0.0, 1.0),
            material.traffic_light.edge_side_bias.clamp(0.0, 1.0),
            material.traffic_light.edge_side_angle.clamp(10.0, 80.0),
        ],
        interaction_response: [
            material.interaction.hover_gain.clamp(0.0, 1.0),
            material.interaction.press_gain.clamp(0.0, 1.0),
            material.interaction.press_lift.clamp(0.0, 1.0),
            0.0,
        ],
        core_light: [
            material.core_light.uniform_light.clamp(0.0, 1.0),
            material.core_light.thin_light_gain.clamp(0.0, 1.0),
            material.core_light.core_power.clamp(1.0, 8.0),
            material.core_light.vertical_power.clamp(0.25, 4.0),
        ],
        bead_a: [
            material.bead.b1, material.bead.b2, material.bead.b3, material.bead.center_glow,
        ],
        bead_b: [
            material.bead.saturation_lift,
            material.bead.highlight_intensity,
            material.bead.dark_rim_intensity,
            material.bead.core_span_factor,
        ],
        bead_c: [
            material.bead.rim_span_factor,
            material.bead.caustic_light,
            material.bead.caustic_dark,
            material.bead.mode_dark,
        ],
        rim_profile: [
            material.rim_profile.lateral_power.clamp(0.25, 16.0),
            material.rim_profile.vertical_floor.clamp(0.0, 1.0),
            material.rim_profile.grazing_power.clamp(0.1, 4.0),
            0.0,
        ],
        core_light_gradient: [
            material.core_light.core_lift.clamp(0.0, 1.0),
            material.core_light.axial_glow.clamp(0.0, 1.0),
            material.core_light.horizontal_power.clamp(0.05, 2.0),
            0.0,
        ],
    }
}

#[allow(clippy::cast_precision_loss)]
fn fused_shape_uniforms(size: GpuSize, node: &GlassNode) -> ([[f32; 4]; 4], [[f32; 4]; 4]) {
    let mut bounds = [[0.0; 4]; 4];
    let mut geometry = [[0.0; 4]; 4];
    for (index, fused_shape) in node.fused_shapes.iter().take(4).enumerate() {
        let center_x = fused_shape.bounds.x + fused_shape.bounds.width * 0.5;
        let center_y = size.height as f32 - fused_shape.bounds.y - fused_shape.bounds.height * 0.5;
        bounds[index] = [center_x, center_y, fused_shape.bounds.width, fused_shape.bounds.height];
        geometry[index] =
            [shape_radius_for_layer(fused_shape), shape_roundness_for_layer(fused_shape), 1.0, 0.0];
    }
    (bounds, geometry)
}

fn shape_radius_for_layer(layer: &liquid_glass_scene::GlassShapeLayer) -> f32 {
    match layer.shape {
        GlassShape::RoundedRect { radius } => radius,
        GlassShape::Superellipse { .. } => layer.bounds.width.min(layer.bounds.height) * 0.4,
        GlassShape::Capsule => layer.bounds.height * 0.5,
        GlassShape::Circle => layer.bounds.width.min(layer.bounds.height) * 0.5,
        GlassShape::Ellipse => layer.bounds.width.min(layer.bounds.height) * 0.25,
    }
}

fn shape_roundness_for_layer(layer: &liquid_glass_scene::GlassShapeLayer) -> f32 {
    match layer.shape {
        GlassShape::Superellipse { exponent } => exponent,
        GlassShape::RoundedRect { .. } => layer.corner_curve.exponent(),
        // The fused path currently shares the rounded-rectangle evaluator;
        // use a continuous circular curve for capsule-like secondary shapes.
        GlassShape::Capsule | GlassShape::Circle | GlassShape::Ellipse => 2.0,
    }
}

fn tint_with_whiteness(material: liquid_glass_scene::GlassMaterial) -> [f32; 4] {
    let tint_alpha = material.tint.a.clamp(0.0, 1.0);
    let white_alpha = material.whiteness.clamp(0.0, 1.0);
    let combined_alpha = 1.0 - (1.0 - tint_alpha) * (1.0 - white_alpha);
    let tint =
        srgb_to_linear_rgba([material.tint.r, material.tint.g, material.tint.b, material.tint.a]);
    if combined_alpha <= f32::EPSILON {
        return [tint[0], tint[1], tint[2], 0.0];
    }

    let tint_weight = tint_alpha * (1.0 - white_alpha);
    [
        (tint[0] * tint_weight + white_alpha) / combined_alpha,
        (tint[1] * tint_weight + white_alpha) / combined_alpha,
        (tint[2] * tint_weight + white_alpha) / combined_alpha,
        combined_alpha,
    ]
}

fn srgb_to_linear_rgba(color: [f32; 4]) -> [f32; 4] {
    [
        srgb_channel_to_linear(color[0]),
        srgb_channel_to_linear(color[1]),
        srgb_channel_to_linear(color[2]),
        color[3].clamp(0.0, 1.0),
    ]
}

fn srgb_channel_to_linear(value: f32) -> f32 {
    let value = value.clamp(0.0, 1.0);
    if value <= 0.04045 { value / 12.92 } else { ((value + 0.055) / 1.055).powf(2.4) }
}

#[must_use]
const fn div_ceil_u32(value: u32, divisor: u32) -> u32 {
    value.saturating_add(divisor.saturating_sub(1)) / divisor
}

#[must_use]
const fn downsampled_size(size: GpuSize, factor: u32) -> GpuSize {
    let factor = if factor == 0 { 1 } else { factor };
    let width = div_ceil_u32(size.width, factor);
    let height = div_ceil_u32(size.height, factor);
    GpuSize::new(if width == 0 { 1 } else { width }, if height == 0 { 1 } else { height })
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, clippy::cast_sign_loss)]
fn blur_radius_for_node(node: &GlassNode, options: GlassRenderOptions) -> i32 {
    let size_gain = size_response(node, node.material.adaptive.size);
    let accessibility_gain = if options.accessibility.reduced_transparency { 1.35 } else { 1.0 };
    let variant_gain = if node.material.variant == GlassVariant::Clear { 0.82 } else { 1.0 };
    (node.material.blur.radius * size_gain * accessibility_gain * variant_gain)
        .round()
        .clamp(0.0, MAX_KAWASE_RADIUS as f32) as i32
}

#[allow(clippy::cast_precision_loss)]
fn size_response(node: &GlassNode, gain: f32) -> f32 {
    let minimum_dimension = node.bounds.width.min(node.bounds.height).max(1.0);
    let normalized = (minimum_dimension / 48.0).sqrt().clamp(0.65, 1.45);
    1.0 + (normalized - 1.0) * gain.clamp(0.0, 2.0)
}

fn create_placeholder_texture(device: &wgpu::Device) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("liquid-glass placeholder texture"),
        size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

fn create_fullscreen_vertex_buffer(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Buffer {
    let vertices = [-1.0_f32, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("liquid-glass fullscreen vertices"),
        size: u64::try_from(std::mem::size_of_val(&vertices)).expect("vertex buffer size fits"),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&vertices));
    buffer
}

fn create_glass_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    scene_view: &wgpu::TextureView,
    blur_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    uniform: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("liquid-glass backdrop bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(scene_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(blur_view),
            },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(sampler) },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: uniform,
                    offset: 0,
                    size: Some(
                        wgpu::BufferSize::new(u64::from(GLASS_UNIFORM_SIZE))
                            .expect("glass uniform size is non-zero"),
                    ),
                }),
            },
        ],
    })
}

fn create_copy_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("liquid-glass copy layout"),
        entries: &[texture_binding(0), sampler_binding(1)],
    })
}

fn create_vibrancy_tint_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("liquid-glass vibrancy tint layout"),
        entries: &[
            texture_binding(0),
            sampler_binding(1),
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        u64::try_from(std::mem::size_of::<VibrancyTintUniform>())
                            .expect("vibrancy tint uniform size fits in u64"),
                    ),
                },
                count: None,
            },
        ],
    })
}

fn create_vibrancy_tint_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    blurred_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    uniform: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("liquid-glass vibrancy tint bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(blurred_view),
            },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: uniform,
                    offset: 0,
                    size: Some(
                        wgpu::BufferSize::new(
                            u64::try_from(std::mem::size_of::<VibrancyTintUniform>())
                                .expect("vibrancy tint uniform size fits in u64"),
                        )
                        .expect("vibrancy tint uniform size is non-zero"),
                    ),
                }),
            },
        ],
    })
}

fn create_gradient_composite_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("liquid-glass gradient composite layout"),
        entries: &[
            texture_binding(0),
            texture_binding(1),
            sampler_binding(2),
            uniform_binding_dynamic(3, std::mem::size_of::<GradientCompositeUniform>()),
        ],
    })
}

fn create_gradient_composite_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    blurred_view: &wgpu::TextureView,
    original_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    uniform: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("liquid-glass gradient composite bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(blurred_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(original_view),
            },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(sampler) },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: uniform,
                    offset: 0,
                    size: Some(
                        wgpu::BufferSize::new(
                            u64::try_from(std::mem::size_of::<GradientCompositeUniform>())
                                .expect("gradient composite uniform size fits in u64"),
                        )
                        .expect("gradient composite uniform size is non-zero"),
                    ),
                }),
            },
        ],
    })
}

fn create_copy_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    source_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("liquid-glass copy bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(source_view),
            },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
        ],
    })
}

fn create_background_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    placeholder_texture: &wgpu::Texture,
    sampler: &wgpu::Sampler,
    uniform: &wgpu::Buffer,
) -> wgpu::BindGroup {
    let placeholder_view = placeholder_texture.create_view(&wgpu::TextureViewDescriptor::default());
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("liquid-glass background bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&placeholder_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&placeholder_view),
            },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(sampler) },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: uniform,
                    offset: 0,
                    size: Some(
                        wgpu::BufferSize::new(u64::from(GLASS_UNIFORM_SIZE))
                            .expect("glass uniform size is non-zero"),
                    ),
                }),
            },
        ],
    })
}

fn texture_binding(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn sampler_binding(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

fn uniform_binding(binding: u32, size: usize) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: wgpu::BufferSize::new(size as u64),
        },
        count: None,
    }
}

fn uniform_stride(device: &wgpu::Device) -> u32 {
    uniform_stride_for_size(device, std::mem::size_of::<GlassUniform>())
}

fn uniform_stride_for_size(device: &wgpu::Device, size: usize) -> u32 {
    let alignment = device.limits().min_uniform_buffer_offset_alignment.max(1);
    u32::try_from(size).expect("uniform size fits in u32").div_ceil(alignment) * alignment
}

fn create_kawase_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("liquid-glass vibrancy Kawase layout"),
        entries: &[
            uniform_binding_dynamic(0, std::mem::size_of::<KawaseUniform>()),
            texture_binding(1),
            sampler_binding(2),
        ],
    })
}

fn create_kawase_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    source_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    uniform: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("liquid-glass vibrancy Kawase bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: uniform,
                    offset: 0,
                    size: Some(
                        wgpu::BufferSize::new(
                            u64::try_from(std::mem::size_of::<KawaseUniform>())
                                .expect("Kawase uniform size fits in u64"),
                        )
                        .expect("Kawase uniform size is non-zero"),
                    ),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(source_view),
            },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(sampler) },
        ],
    })
}

fn uniform_binding_dynamic(binding: u32, size: usize) -> wgpu::BindGroupLayoutEntry {
    uniform_binding(binding, size)
}

fn kawase_dynamic_offset(stride: u32, logical_slot: usize, pass: usize) -> u32 {
    let slot = logical_slot
        .checked_mul(8)
        .and_then(|base| base.checked_add(pass))
        .expect("Kawase uniform slot fits in usize");
    stride
        .checked_mul(u32::try_from(slot).expect("Kawase uniform slot fits in u32"))
        .expect("Kawase dynamic offset fits in u32")
}

fn write_kawase_uniform(
    queue: &wgpu::Queue,
    uniform: &wgpu::Buffer,
    stride: u32,
    logical_slot: usize,
    pass: usize,
    halfpixel: (f32, f32),
    offset: f32,
) {
    let value = KawaseUniform { halfpixel: [halfpixel.0, halfpixel.1], offset, alpha: 1.0 };
    queue.write_buffer(
        uniform,
        u64::from(kawase_dynamic_offset(stride, logical_slot, pass)),
        bytemuck::bytes_of(&value),
    );
}

fn create_pipeline(
    device: &wgpu::Device,
    label: &str,
    shader: &wgpu::ShaderModule,
    bind_group_layout: Option<&wgpu::BindGroupLayout>,
    fragment_entry: &str,
    blend: Option<wgpu::BlendState>,
    output_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[bind_group_layout.expect("pipeline requires a bind group layout")],
        push_constant_ranges: &[],
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: 8,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: FULLSCREEN_VERTEX_ATTRIBUTES,
            }],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(fragment_entry),
            targets: &[Some(wgpu::ColorTargetState {
                format: output_format,
                blend,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        multiview: None,
        cache: None,
    })
}

fn create_copy_pipeline(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    output_format: wgpu::TextureFormat,
    blend: Option<wgpu::BlendState>,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("liquid-glass hierarchical copy shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Owned(format!(
            "{}\n{}",
            include_str!("../../../shaders/glass/reference/vertex.wgsl"),
            include_str!("../../../shaders/glass/copy.wgsl"),
        ))),
    });
    create_pipeline(
        device,
        "liquid-glass hierarchical copy pipeline",
        &shader,
        Some(bind_group_layout),
        "fs_main",
        blend,
        output_format,
    )
}

fn create_window_mask_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("liquid-glass window mask layout"),
        entries: &[
            texture_binding(0),
            sampler_binding(1),
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        u64::try_from(std::mem::size_of::<WindowMaskUniform>())
                            .expect("window mask uniform size fits in u64"),
                    ),
                },
                count: None,
            },
        ],
    })
}

fn create_window_mask_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    source_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    uniform: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("liquid-glass window mask bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(source_view),
            },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
            wgpu::BindGroupEntry { binding: 2, resource: uniform.as_entire_binding() },
        ],
    })
}

fn create_window_mask_pipeline(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    output_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("liquid-glass rounded window mask shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Owned(format!(
            "{}\n{}\n{}",
            include_str!("../../../shaders/glass/reference/vertex.wgsl"),
            squircle_rs::wgsl_squircle_sdf_source(),
            include_str!("../../../shaders/glass/window-mask.wgsl"),
        ))),
    });
    create_pipeline(
        device,
        "liquid-glass rounded window mask pipeline",
        &shader,
        Some(bind_group_layout),
        "fs_main",
        None,
        output_format,
    )
}

fn create_vibrancy_tint_pipeline(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    output_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("liquid-glass vibrancy tint shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Owned(format!(
            "{}\n{}",
            include_str!("../../../shaders/glass/reference/vertex.wgsl"),
            include_str!("../../../shaders/glass/vibrancy-tint.wgsl"),
        ))),
    });
    create_pipeline(
        device,
        "liquid-glass vibrancy tint pipeline",
        &shader,
        Some(bind_group_layout),
        "fs_main",
        None,
        output_format,
    )
}

fn create_gradient_composite_pipeline(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    output_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("liquid-glass gradient composite shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Owned(format!(
            "{}\n{}",
            include_str!("../../../shaders/glass/reference/vertex.wgsl"),
            include_str!("../../../shaders/glass/gradient-composite.wgsl"),
        ))),
    });
    create_pipeline(
        device,
        "liquid-glass gradient composite pipeline",
        &shader,
        Some(bind_group_layout),
        "fs_main",
        None,
        output_format,
    )
}

fn create_pipelines(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    kawase_bind_group_layout: &wgpu::BindGroupLayout,
    output_format: wgpu::TextureFormat,
) -> (
    wgpu::RenderPipeline,
    wgpu::RenderPipeline,
    wgpu::RenderPipeline,
    wgpu::RenderPipeline,
    wgpu::RenderPipeline,
) {
    let background_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("liquid-glass reference background shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Owned(reference_shader(
            include_str!("../../../shaders/glass/reference/fragment-bg.wgsl"),
            ReferencePass::Background,
        ))),
    });
    let downsample_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("liquid-glass vibrancy Dual Kawase downsample shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Owned(format!(
            "{}\n{}",
            KAWASE_VERTEX_SHADER,
            vibrancy_rs::WGSL_DOWNSAMPLE_SHADER,
        ))),
    });
    let upsample_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("liquid-glass vibrancy Dual Kawase upsample shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Owned(format!(
            "{}\n{}",
            KAWASE_VERTEX_SHADER,
            vibrancy_rs::WGSL_UPSAMPLE_SHADER,
        ))),
    });
    let glass_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("liquid-glass reference main shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Owned(reference_shader(
            include_str!("../../../shaders/glass/reference/fragment-main.wgsl"),
            ReferencePass::Main,
        ))),
    });
    let background = create_pipeline(
        device,
        "liquid-glass background pipeline",
        &background_shader,
        Some(bind_group_layout),
        "fs_main",
        None,
        output_format,
    );
    let downsample = create_pipeline(
        device,
        "liquid-glass vibrancy Dual Kawase downsample pipeline",
        &downsample_shader,
        Some(kawase_bind_group_layout),
        "fs_main",
        None,
        output_format,
    );
    let upsample = create_pipeline(
        device,
        "liquid-glass vibrancy Dual Kawase upsample pipeline",
        &upsample_shader,
        Some(kawase_bind_group_layout),
        "fs_main",
        None,
        output_format,
    );
    let glass = create_pipeline(
        device,
        "liquid-glass SDF pipeline",
        &glass_shader,
        Some(bind_group_layout),
        "fs_main",
        None,
        output_format,
    );
    let shadow = create_pipeline(
        device,
        "liquid-glass analytic shadow pipeline",
        &glass_shader,
        Some(bind_group_layout),
        "fs_shadow",
        None,
        output_format,
    );

    (background, downsample, upsample, shadow, glass)
}

const KAWASE_VERTEX_SHADER: &str = r#"
@vertex
fn vs_main(@location(0) a_position: vec2<f32>) -> VertexOutput {
    var out: VertexOutput;
    let uv = (a_position + 1.0) * 0.5;
    out.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    out.position = vec4<f32>(a_position, 0.0, 1.0);
    return out;
}
"#;

#[derive(Clone, Copy)]
enum ReferencePass {
    Background,
    Main,
}

fn reference_shader(fragment: &str, pass: ReferencePass) -> String {
    let vertex = include_str!("../../../shaders/glass/reference/vertex.wgsl");
    let mut fragment = fragment.to_owned();
    match pass {
        ReferencePass::Background => {
            fragment = fragment
                .replace(
                    "@binding(0) var<uniform> u: Uniforms",
                    "@binding(3) var<uniform> u: Uniforms",
                )
                .replace("@binding(1) var u_bgTexture", "@binding(0) var u_bgTexture");
        }
        ReferencePass::Main => {
            fragment = fragment
                .replace(
                    "@binding(0) var<uniform> u: Uniforms",
                    "@binding(3) var<uniform> u: Uniforms",
                )
                .replace("@binding(2) var u_bg", "@binding(0) var u_bg")
                .replace("@binding(3) var u_sampler", "@binding(2) var u_sampler");
        }
    }
    format!("{vertex}\n{fragment}")
}

fn shape_radius(node: &GlassNode) -> f32 {
    match node.shape {
        GlassShape::RoundedRect { radius } => radius,
        GlassShape::Superellipse { .. } => node.bounds.width.min(node.bounds.height) * 0.4,
        GlassShape::Capsule => node.bounds.height * 0.5,
        GlassShape::Circle => node.bounds.width.min(node.bounds.height) * 0.5,
        GlassShape::Ellipse => node.bounds.width.min(node.bounds.height) * 0.25,
    }
}

fn shape_roundness(node: &GlassNode) -> f32 {
    match node.shape {
        GlassShape::Superellipse { exponent } => exponent,
        GlassShape::RoundedRect { .. } => node.corner_curve.exponent(),
        // The upstream squircle geometry intentionally resolves a capsule
        // whose radius reaches the short axis to circular end caps. Reuse
        // the standard rounded-rectangle SDF for that exact case.
        GlassShape::Capsule => 2.0,
        GlassShape::Circle | GlassShape::Ellipse => 2.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use liquid_glass_scene::{CornerCurve, GlassId, GlassMaterial, Rect};

    #[test]
    fn noop_device_can_build_and_submit_glass_frame() {
        let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
        let mut renderer = GpuRenderer::from_device(device, queue, GpuSize::new(128, 128));
        let padded_backdrop = vec![0_u8; 12 * 2];
        renderer
            .set_background_rgba8(2, 2, 12, &padded_backdrop)
            .expect("padded RGBA8 backdrop upload");
        let node = GlassNode::new(GlassId(1), Rect::new(16.0, 16.0, 96.0, 64.0))
            .material(GlassMaterial::regular());

        renderer.render_panel(&node, 0.0);
        assert_eq!(renderer.size(), GpuSize::new(128, 128));

        let mut scene = GlassScene::default();
        scene.push(node.clone());
        let mut front = GlassNode::new(GlassId(2), Rect::new(48.0, 40.0, 64.0, 56.0))
            .shape(GlassShape::Capsule)
            .material(GlassMaterial::interactive());
        front.z_index = 1;
        scene.push(front);
        renderer.render_scene(&scene, 0.0).expect("render multiple glass nodes");

        renderer.resize(GpuSize::new(65, 33)).expect("resize blur targets");
        renderer.render_scene(&scene, 0.0).expect("render multiple glass nodes after resize");
        assert_eq!(renderer.size(), GpuSize::new(65, 33));
    }

    #[test]
    fn frame_batch_encodes_multiple_glass_groups_with_one_submission() {
        let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
        let renderer = GpuRenderer::from_device(device, queue, GpuSize::new(96, 64));
        let source = renderer.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("batched renderer test source"),
            size: wgpu::Extent3d { width: 96, height: 64, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEFAULT_OUTPUT_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let source_view = source.create_view(&wgpu::TextureViewDescriptor::default());
        let mut source_batch = renderer.begin_frame_batch(Some("batched source test"));
        source_batch.render_background_to_view(&source_view).expect("background slot fits");
        source_batch
            .render_solid_region_to_view(&source_view, (0, 0, 48, 64), [0.8, 0.8, 0.8, 0.8])
            .expect("solid slot fits");
        source_batch.submit();

        let mut back = GlassScene::default();
        back.push(
            GlassNode::new(GlassId(20), Rect::new(8.0, 8.0, 40.0, 28.0))
                .material(GlassMaterial::regular()),
        );
        let mut front = GlassScene::default();
        front.push(
            GlassNode::new(GlassId(21), Rect::new(48.0, 24.0, 40.0, 28.0))
                .material(GlassMaterial::interactive()),
        );
        let mut glass_batch = renderer.begin_frame_batch(Some("batched glass test"));
        glass_batch.render_scene_with_source(&source, &back, 0.0).expect("first glass group fits");
        glass_batch.render_scene_over_output(&front, 0.0).expect("second glass group fits");
        glass_batch
            .render_scroll_edge_to_output(
                (0, 0, 48, 16),
                8,
                64,
                [0.8, 0.8, 0.8, 0.8],
                ScrollEdgeStyle::Soft,
            )
            .expect("one scroll edge fits");
        assert_eq!(
            glass_batch.render_scroll_edge_to_output(
                (0, 16, 48, 16),
                24,
                4,
                [0.8, 0.8, 0.8, 0.8],
                ScrollEdgeStyle::Soft,
            ),
            Err(GpuError::MultipleScrollEdgesInFrameBatch)
        );
        glass_batch.copy_output_to_view(&source_view);
        glass_batch.submit();
    }

    #[test]
    fn rounded_rects_and_capsules_use_continuous_curves() {
        let continuous = GlassNode::new(GlassId(3), Rect::new(0.0, 0.0, 72.0, 36.0))
            .shape(GlassShape::RoundedRect { radius: 12.0 });
        let circular = continuous.clone().corner_curve(CornerCurve::Circular);
        let capsule = continuous.clone().shape(GlassShape::Capsule);
        let circle = continuous.clone().shape(GlassShape::Circle);

        assert!(
            (shape_roundness(&continuous)
                - liquid_glass_scene::CornerCurve::DEFAULT_CONTINUOUS_EXPONENT)
                .abs()
                < f32::EPSILON
        );
        assert!((shape_roundness(&circular) - 2.0).abs() < f32::EPSILON);
        assert!((shape_roundness(&capsule) - 2.0).abs() < f32::EPSILON);
        assert!((shape_roundness(&circle) - 2.0).abs() < f32::EPSILON);
        // Keeps the Rust uniform in lockstep with the WGSL `Uniforms` struct:
        // adding a field on one side only would silently shift every following
        // slot. 33 x 16 bytes, uniform address space.
        assert_eq!(std::mem::size_of::<GlassUniform>(), 528);
    }

    #[test]
    fn the_scale_factor_reaches_the_uniform_without_moving_geometry() {
        let node = GlassNode::new(GlassId(7), Rect::new(4.0, 6.0, 20.0, 20.0));
        let uniform = |scale_factor: f32| {
            uniform_for_node(
                GpuSize::new(100, 100),
                &node,
                0.0,
                false,
                1.0,
                true,
                false,
                GlassRenderOptions { scale_factor, ..GlassRenderOptions::default() },
            )
        };

        let base = uniform(1.0);
        let retina = uniform(2.0);
        assert_eq!(base.resolution_dpr_pad[3], 1.0);
        assert_eq!(retina.resolution_dpr_pad[3], 2.0);
        // The scale factor only feeds point-authored material responses. Node
        // geometry is already physical, so no shape slot may follow it.
        assert_eq!(base.shape, retina.shape);
        assert_eq!(base.mouse_and_spring, retina.mouse_and_spring);
        assert_eq!(base.fused_bounds, retina.fused_bounds);
    }

    #[test]
    fn reference_fusion_circle_is_included_in_scissor_bounds() {
        let mut material = GlassMaterial::clear();
        material.show_shape1 = true;
        let node =
            GlassNode::new(GlassId(4), Rect::new(390.0, 220.0, 200.0, 200.0)).material(material);

        let bounds = node_optical_bounds(&node, GpuSize::new(640, 640));

        assert_eq!(bounds, Rect::new(220.0, 220.0, 370.0, 200.0));
    }

    #[test]
    fn whiteness_adds_a_neutral_layer_above_tint() {
        let mut material = GlassMaterial::clear();
        material.tint = liquid_glass_scene::Color::rgba(0.2, 0.4, 0.8, 0.10);
        material.whiteness = 0.20;

        let effective = tint_with_whiteness(material);

        assert!((effective[3] - 0.28).abs() < f32::EPSILON);
        let tint = srgb_to_linear_rgba([
            material.tint.r,
            material.tint.g,
            material.tint.b,
            material.tint.a,
        ]);
        assert!(effective[0] > tint[0]);
        assert!(effective[1] > tint[1]);
        assert!(effective[2] > tint[2]);
    }

    #[test]
    fn srgb_uniform_colors_are_decoded_to_linear_light() {
        let linear = srgb_to_linear_rgba([0.5, 0.5, 0.5, 0.4]);

        assert!((linear[0] - 0.214_041_14).abs() < 0.000_01);
        assert_eq!(linear[3], 0.4);
    }

    #[test]
    fn downsampled_targets_cover_odd_full_resolution_edges() {
        let full_size = GpuSize::new(9, 7);

        assert_eq!(downsampled_size(full_size, 2), GpuSize::new(5, 4));
    }
}
