//! Renders the fitted content-glass material over a backdrop image.
//!
//! ```sh
//! cargo run -p liquid-glass-render --example content_glass -- backdrop.png out.png
//! ```
//!
//! A second optional argument selects the appearance (`light`/`dark`) and a
//! third selects the material (`regular`/`clear`). `clear` is the transparent
//! variant, which is where the rim lensing is easiest to see. The backdrop is
//! resized to 1280x800 logical pixels (2560x1600 device pixels at 2x),
//! matching the captures the material was fitted on.

use liquid_glass_render::{
    ContentGlassAppearance, ContentGlassMaterial, ContentGlassNode, ContentGlassRenderer, GpuSize,
};

const W: u32 = 2560;
const H: u32 = 1600;
const SCALE: f32 = 2.0;

fn main() {
    pollster::block_on(run());
}

async fn run() {
    let mut args = std::env::args().skip(1);
    let backdrop_path = args.next().unwrap_or_else(|| "backdrop.png".into());
    let out_path = args.next().unwrap_or_else(|| "content_glass.png".into());
    let appearance = match args.next().as_deref() {
        Some("dark") => ContentGlassAppearance::DARK,
        _ => ContentGlassAppearance::LIGHT,
    };
    let clear = matches!(args.next().as_deref(), Some("clear"));

    let backdrop = image::open(&backdrop_path)
        .expect("backdrop image")
        .resize_exact(W, H, image::imageops::FilterType::Lanczos3)
        .to_rgba8()
        .into_raw();

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .expect("adapter");
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor { label: Some("example"), ..Default::default() })
        .await
        .expect("device");

    let mut renderer = ContentGlassRenderer::from_device(
        device.clone(),
        queue.clone(),
        GpuSize::new(W, H),
        wgpu::TextureFormat::Rgba8UnormSrgb,
    );
    if clear {
        renderer.set_material(ContentGlassMaterial::clear());
    }
    renderer.set_backdrop_rgba8(W, H, &backdrop).expect("backdrop");

    // (x, y, w, h, corner_radius) in logical pixels, matching the macOS capture.
    let panels = [
        (120.0, 120.0, 320.0, 120.0, 60.0),
        (520.0, 100.0, 260.0, 260.0, 48.0),
        (880.0, 150.0, 300.0, 110.0, 55.0),
        (420.0, 480.0, 300.0, 200.0, 40.0),
        (880.0, 500.0, 260.0, 100.0, 50.0),
    ];
    let nodes: Vec<ContentGlassNode> = panels
        .iter()
        .map(|&(x, y, w, h, r)| {
            ContentGlassNode::new(x * SCALE, y * SCALE, w * SCALE, h * SCALE)
                .corner_radius(r * SCALE)
                .appearance(appearance)
        })
        .collect();

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("example output"),
        size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &backdrop,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(W * 4),
            rows_per_image: Some(H),
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    renderer.render_to_view(&view, &nodes, wgpu::LoadOp::Load);

    let bytes_per_row = (W * 4).div_ceil(256) * 256;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: u64::from(bytes_per_row) * u64::from(H),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(H),
            },
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
    queue.submit(Some(encoder.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).expect("poll");
    rx.recv().expect("map channel").expect("map");

    let mut pixels = vec![0u8; (W * H * 4) as usize];
    {
        let data = slice.get_mapped_range();
        for y in 0..H as usize {
            let src =
                &data[y * bytes_per_row as usize..y * bytes_per_row as usize + (W * 4) as usize];
            let dst = &mut pixels[y * (W * 4) as usize..(y + 1) * (W * 4) as usize];
            dst.copy_from_slice(src);
        }
    }
    readback.unmap();

    image::RgbaImage::from_raw(W, H, pixels).expect("image").save(&out_path).expect("save");
    println!("wrote {out_path}");
}
