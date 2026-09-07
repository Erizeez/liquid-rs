//! Renderer contracts and render-graph foundation.
//!
//! The scene and render-graph contracts stay backend-neutral while the crate
//! also exposes the first `wgpu` compositor implementation.

#![deny(unsafe_code)]

use liquid_glass_scene::{GlassScene, Rect};

mod gpu;

pub use gpu::{GpuError, GpuFrameBatch, GpuRenderer, GpuSize};

/// Passes in the Liquid Glass composition pipeline.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RenderPass {
    Scene,
    Capture,
    Downsample,
    BlurHorizontal,
    BlurVertical,
    Shadow,
    Glass,
    ScrollEdge,
    Foreground,
    Overlay,
    Present,
}

/// How a scroll edge transitions from blurred content to the sharp list.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScrollEdgeStyle {
    /// Interpolates the blur over the edge region.
    Soft,
    /// Keeps the edge region fully blurred and ends it at a defined boundary.
    Hard,
}

/// A declarative render graph for one frame.
#[derive(Clone, Debug, Default)]
pub struct RenderGraph {
    passes: Vec<RenderPass>,
}

impl RenderGraph {
    #[must_use]
    pub fn foundation() -> Self {
        Self {
            passes: vec![
                RenderPass::Scene,
                RenderPass::Capture,
                RenderPass::Downsample,
                RenderPass::BlurHorizontal,
                RenderPass::BlurVertical,
                RenderPass::Shadow,
                RenderPass::Glass,
                RenderPass::ScrollEdge,
                RenderPass::Foreground,
                RenderPass::Overlay,
                RenderPass::Present,
            ],
        }
    }

    #[must_use]
    pub fn passes(&self) -> &[RenderPass] {
        &self.passes
    }
}

/// The properties used to reuse a GPU texture allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TextureKey {
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
}

/// Initial texture formats supported by the pool contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextureFormat {
    Rgba8Unorm,
    Rgba16Float,
}

/// Allocation accounting for the future `wgpu::Texture` pool.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TexturePool {
    retained: Vec<TextureKey>,
}

impl TexturePool {
    pub fn retain(&mut self, key: TextureKey) {
        if !self.retained.contains(&key) {
            self.retained.push(key);
        }
    }

    #[must_use]
    pub fn retained(&self) -> &[TextureKey] {
        &self.retained
    }
}

/// Backend-independent renderer state for the foundation milestone.
#[derive(Clone, Debug)]
pub struct LiquidRenderer {
    graph: RenderGraph,
    pub texture_pool: TexturePool,
}

impl Default for LiquidRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl LiquidRenderer {
    #[must_use]
    pub fn new() -> Self {
        Self { graph: RenderGraph::foundation(), texture_pool: TexturePool::default() }
    }

    #[must_use]
    pub fn graph(&self) -> &RenderGraph {
        &self.graph
    }

    #[must_use]
    pub fn capture_region(&self, scene: &GlassScene) -> Option<Rect> {
        scene.capture_bounds()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn foundation_graph_contains_composition_passes() {
        let graph = RenderGraph::foundation();

        assert_eq!(graph.passes().first(), Some(&RenderPass::Scene));
        assert_eq!(graph.passes().last(), Some(&RenderPass::Present));
        assert!(graph.passes().contains(&RenderPass::BlurHorizontal));
        assert!(graph.passes().contains(&RenderPass::Glass));
        assert!(graph.passes().contains(&RenderPass::Shadow));
        assert!(graph.passes().contains(&RenderPass::ScrollEdge));
    }
}
