//! 3D mesh viewer binary for morsel.
//!
//! Usage: morsel-view <mesh_file> [--texture <texture_file>] [--parameterize] [--curvature <mean|gaussian>]
//!
//! Controls:
//! - Left mouse drag: Rotate camera
//! - Scroll wheel: Zoom in/out
//! - W: Toggle wireframe mode
//! - B: Toggle backface culling
//! - T: Toggle textured mode (requires texture and UVs)
//! - C: Toggle vertex colors
//! - R: Reset camera
//! - Escape: Quit

mod camera;
mod mesh_gpu;
mod renderer;

use std::env;
use std::path::PathBuf;
use std::sync::Arc;

use winit::{
    application::ApplicationHandler,
    dpi::PhysicalPosition,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

use camera::OrbitCamera;
use mesh_gpu::{GpuMesh, VertexColors};
use renderer::Renderer;

use morsel::algo::curvature;
use morsel::algo::parameterize::{cylindrical_projection, UVMap};
use morsel::io::obj;
use morsel::mesh::HalfEdgeMesh;

/// Type of curvature to visualize.
#[derive(Debug, Clone, Copy)]
enum CurvatureType {
    Mean,
    Gaussian,
}

/// Application state.
struct App {
    /// Path to the mesh file to load.
    mesh_path: String,
    /// Optional path to a texture file.
    texture_path: Option<PathBuf>,
    /// Whether to compute UV parameterization.
    parameterize: bool,
    /// Optional curvature type to visualize.
    curvature_type: Option<CurvatureType>,
    /// `(source vertex, use Dijkstra instead of the heat method)`.
    geodesic: Option<(usize, bool)>,
    /// The window (created after resume).
    window: Option<Arc<Window>>,
    /// The renderer (created after window).
    renderer: Option<Renderer>,
    /// The GPU mesh (created after renderer).
    gpu_mesh: Option<GpuMesh>,
    /// The camera.
    camera: OrbitCamera,
    /// Whether wireframe mode is enabled.
    wireframe: bool,
    /// Whether backface culling is enabled.
    backface_culling: bool,
    /// Whether textured mode is enabled.
    textured: bool,
    /// Whether vertex colors are shown (can be toggled with C key).
    show_colors: bool,
    /// Whether the left mouse button is pressed.
    mouse_pressed: bool,
    /// Last mouse position.
    last_mouse_pos: Option<PhysicalPosition<f64>>,
}

impl App {
    fn new(
        mesh_path: String,
        texture_path: Option<PathBuf>,
        parameterize: bool,
        curvature_type: Option<CurvatureType>,
        geodesic: Option<(usize, bool)>,
    ) -> Self {
        // Disable backface culling by default for curvature visualization
        // (meshes often have inconsistent winding at high-curvature areas)
        let backface_culling = curvature_type.is_none();

        Self {
            mesh_path,
            texture_path,
            parameterize,
            curvature_type,
            geodesic,
            window: None,
            renderer: None,
            gpu_mesh: None,
            camera: OrbitCamera::default(),
            wireframe: false,
            backface_culling,
            textured: true,    // Enabled by default when texture is available
            show_colors: true, // Show vertex colors by default
            mouse_pressed: false,
            last_mouse_pos: None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        // Create window
        let window_attrs = Window::default_attributes()
            .with_title("Morsel Viewer")
            .with_inner_size(winit::dpi::LogicalSize::new(1024, 768));

        let window = Arc::new(
            event_loop
                .create_window(window_attrs)
                .expect("Failed to create window"),
        );

        // Initialize renderer
        let mut renderer = pollster::block_on(Renderer::new(window.clone()));

        let Prepared {
            mesh,
            uv_map,
            vertex_colors,
        } = prepare_mesh(
            &self.mesh_path,
            self.parameterize,
            self.curvature_type,
            self.geodesic,
        );

        // Create GPU mesh with optional UVs and vertex colors
        let gpu_mesh = GpuMesh::from_halfedge_mesh(
            renderer.device(),
            &mesh,
            uv_map.as_ref(),
            vertex_colors.as_ref(),
        );

        // Load texture if specified (only used when no vertex colors)
        if vertex_colors.is_none() {
            if let Some(ref texture_path) = self.texture_path {
                if let Err(e) = renderer.load_texture(texture_path) {
                    log::error!("Failed to load texture: {}", e);
                }
            }
        }

        // Set up camera to view the mesh
        self.camera.reset(gpu_mesh.center, gpu_mesh.radius * 2.5);

        self.window = Some(window);
        self.renderer = Some(renderer);
        self.gpu_mesh = Some(gpu_mesh);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            WindowEvent::Resized(new_size) => {
                if let Some(ref mut renderer) = self.renderer {
                    renderer.resize(new_size);
                }
                if let Some(ref window) = self.window {
                    window.request_redraw();
                }
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    match event.logical_key {
                        Key::Named(NamedKey::Escape) => {
                            event_loop.exit();
                        }
                        Key::Character(ref c) if c == "w" || c == "W" => {
                            self.wireframe = !self.wireframe;
                            log::info!(
                                "Wireframe mode: {}",
                                if self.wireframe { "ON" } else { "OFF" }
                            );
                            if let Some(ref window) = self.window {
                                window.request_redraw();
                            }
                        }
                        Key::Character(ref c) if c == "r" || c == "R" => {
                            if let Some(ref gpu_mesh) = self.gpu_mesh {
                                self.camera.reset(gpu_mesh.center, gpu_mesh.radius * 2.5);
                                log::info!("Camera reset");
                                if let Some(ref window) = self.window {
                                    window.request_redraw();
                                }
                            }
                        }
                        Key::Character(ref c) if c == "b" || c == "B" => {
                            self.backface_culling = !self.backface_culling;
                            log::info!(
                                "Backface culling: {}",
                                if self.backface_culling { "ON" } else { "OFF" }
                            );
                            if let Some(ref window) = self.window {
                                window.request_redraw();
                            }
                        }
                        Key::Character(ref c) if c == "t" || c == "T" => {
                            self.textured = !self.textured;
                            log::info!(
                                "Textured mode: {}",
                                if self.textured { "ON" } else { "OFF" }
                            );
                            if let Some(ref window) = self.window {
                                window.request_redraw();
                            }
                        }
                        Key::Character(ref c) if c == "c" || c == "C" => {
                            self.show_colors = !self.show_colors;
                            log::info!(
                                "Vertex colors: {}",
                                if self.show_colors { "ON" } else { "OFF" }
                            );
                            if let Some(ref window) = self.window {
                                window.request_redraw();
                            }
                        }
                        _ => {}
                    }
                }
            }

            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left {
                    self.mouse_pressed = state == ElementState::Pressed;
                    if !self.mouse_pressed {
                        self.last_mouse_pos = None;
                    }
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                if self.mouse_pressed {
                    if let Some(last_pos) = self.last_mouse_pos {
                        let dx = position.x - last_pos.x;
                        let dy = position.y - last_pos.y;

                        // Rotate camera
                        let sensitivity = 0.005;
                        self.camera
                            .rotate(-dx as f32 * sensitivity, dy as f32 * sensitivity);

                        if let Some(ref window) = self.window {
                            window.request_redraw();
                        }
                    }
                    self.last_mouse_pos = Some(position);
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 100.0,
                };

                // Zoom camera
                let zoom_factor = 1.0 - scroll * 0.1;
                self.camera.zoom(zoom_factor);

                if let Some(ref window) = self.window {
                    window.request_redraw();
                }
            }

            WindowEvent::RedrawRequested => {
                if let (Some(ref mut renderer), Some(ref gpu_mesh)) =
                    (&mut self.renderer, &self.gpu_mesh)
                {
                    match renderer.render(
                        gpu_mesh,
                        &self.camera,
                        self.wireframe,
                        self.backface_culling,
                        self.textured,
                        self.show_colors,
                    ) {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost) => {
                            if let Some(ref window) = self.window {
                                renderer.resize(window.inner_size());
                            }
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => {
                            log::error!("Out of memory");
                            event_loop.exit();
                        }
                        Err(e) => {
                            log::error!("Render error: {:?}", e);
                        }
                    }
                }
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // Request continuous redraws for smooth interaction
        if let Some(ref window) = self.window {
            window.request_redraw();
        }
    }
}

fn main() {
    // Initialize logging
    env_logger::init();

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <mesh_file> [OPTIONS]", args[0]);
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --texture <file>       Load a texture image (PNG, JPG, etc.)");
        eprintln!("  --parameterize         Compute UV coordinates (cylindrical projection)");
        eprintln!(
            "  --curvature <type>     Visualize curvature as vertex colors (mean or gaussian)"
        );
        eprintln!("  --geodesic <vertex>    Colour by geodesic distance from a source vertex,");
        eprintln!("                         with isolines (heat method)");
        eprintln!("  --geodesic-dijkstra    Use Dijkstra graph distance instead");
        eprintln!("  --screenshot <file>    Render one frame to a PNG and exit (no window)");
        eprintln!("  --size WxH             Screenshot size, default 1200x900");
        eprintln!("  --azimuth <radians>    Camera azimuth for the screenshot, default 0.6");
        eprintln!("  --elevation <radians>  Camera elevation for the screenshot, default 0.35");
        eprintln!("  --wireframe            Draw the wireframe instead of solid shading");
        eprintln!("  --distance <d>         Fixed camera distance; use the same value across a");
        eprintln!("                         before/after pair so sizes stay comparable");
        eprintln!();
        eprintln!("Supported mesh formats: .obj, .stl, .ply, .gltf, .glb");
        eprintln!();
        eprintln!("Controls:");
        eprintln!("  Left mouse drag: Rotate camera");
        eprintln!("  Scroll wheel:    Zoom in/out");
        eprintln!("  W:               Toggle wireframe");
        eprintln!("  B:               Toggle backface culling");
        eprintln!("  T:               Toggle textured mode");
        eprintln!("  C:               Toggle vertex colors");
        eprintln!("  R:               Reset camera");
        eprintln!("  Escape:          Quit");
        std::process::exit(1);
    }

    let mesh_path = args[1].clone();

    // Parse optional arguments
    let mut texture_path: Option<PathBuf> = None;
    let mut parameterize = false;
    let mut curvature_type: Option<CurvatureType> = None;
    let mut geodesic: Option<(usize, bool)> = None;
    let mut geodesic_dijkstra = false;
    let mut screenshot: Option<PathBuf> = None;
    let mut size = (1200u32, 900u32);
    let mut azimuth = 0.6f32;
    let mut elevation = 0.35f32;
    let mut wireframe = false;
    let mut distance: Option<f32> = None;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--texture" => {
                if i + 1 < args.len() {
                    texture_path = Some(PathBuf::from(&args[i + 1]));
                    i += 2;
                } else {
                    eprintln!("Error: --texture requires a file path");
                    std::process::exit(1);
                }
            }
            "--parameterize" => {
                parameterize = true;
                i += 1;
            }
            "--curvature" => {
                if i + 1 < args.len() {
                    curvature_type = match args[i + 1].as_str() {
                        "mean" => Some(CurvatureType::Mean),
                        "gaussian" => Some(CurvatureType::Gaussian),
                        other => {
                            eprintln!(
                                "Error: --curvature requires 'mean' or 'gaussian', got '{}'",
                                other
                            );
                            std::process::exit(1);
                        }
                    };
                    i += 2;
                } else {
                    eprintln!("Error: --curvature requires 'mean' or 'gaussian'");
                    std::process::exit(1);
                }
            }
            "--geodesic" => match args.get(i + 1).and_then(|v| v.parse::<usize>().ok()) {
                Some(v) => {
                    geodesic = Some((v, false));
                    i += 2;
                }
                None => {
                    eprintln!("Error: --geodesic requires a source vertex index");
                    std::process::exit(1);
                }
            },
            "--geodesic-dijkstra" => {
                geodesic_dijkstra = true;
                i += 1;
            }
            "--screenshot" => {
                if i + 1 < args.len() {
                    screenshot = Some(PathBuf::from(&args[i + 1]));
                    i += 2;
                } else {
                    eprintln!("Error: --screenshot requires an output path");
                    std::process::exit(1);
                }
            }
            "--size" => {
                let parsed = args.get(i + 1).and_then(|v| {
                    let (w, h) = v.split_once('x')?;
                    Some((w.parse().ok()?, h.parse().ok()?))
                });
                match parsed {
                    Some(wh) => {
                        size = wh;
                        i += 2;
                    }
                    None => {
                        eprintln!("Error: --size expects WIDTHxHEIGHT, e.g. 1200x900");
                        std::process::exit(1);
                    }
                }
            }
            "--azimuth" | "--elevation" => {
                let value = args.get(i + 1).and_then(|v| v.parse::<f32>().ok());
                match value {
                    Some(v) => {
                        if args[i] == "--azimuth" {
                            azimuth = v;
                        } else {
                            elevation = v;
                        }
                        i += 2;
                    }
                    None => {
                        eprintln!("Error: {} expects an angle in radians", args[i]);
                        std::process::exit(1);
                    }
                }
            }
            "--wireframe" => {
                wireframe = true;
                i += 1;
            }
            "--distance" => match args.get(i + 1).and_then(|v| v.parse::<f32>().ok()) {
                Some(v) => {
                    distance = Some(v);
                    i += 2;
                }
                None => {
                    eprintln!("Error: --distance expects a camera distance");
                    std::process::exit(1);
                }
            },
            _ => {
                eprintln!("Unknown option: {}", args[i]);
                std::process::exit(1);
            }
        }
    }

    // `--geodesic-dijkstra` is a modifier on `--geodesic`, so fold it in once both
    // have been seen regardless of the order they appeared.
    if let Some((source, _)) = geodesic {
        geodesic = Some((source, geodesic_dijkstra));
    } else if geodesic_dijkstra {
        eprintln!("Error: --geodesic-dijkstra requires --geodesic <vertex>");
        std::process::exit(1);
    }

    // Offscreen path: render one frame to a file and exit without opening a window.
    if let Some(path) = screenshot {
        render_to_file(
            &mesh_path,
            texture_path.as_deref(),
            parameterize,
            curvature_type,
            geodesic,
            &path,
            size,
            azimuth,
            elevation,
            wireframe,
            distance,
        );
        return;
    }

    // Create event loop and run app
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = App::new(
        mesh_path,
        texture_path,
        parameterize,
        curvature_type,
        geodesic,
    );
    event_loop.run_app(&mut app).expect("Event loop error");
}

/// Render a single frame offscreen and write it out as a PNG.
///
/// Uses the same `Renderer` as the interactive viewer, so the image matches what
/// the window would show. No event loop and no surface, which also means this runs
/// without a display.
#[allow(clippy::too_many_arguments)]
fn render_to_file(
    mesh_path: &str,
    texture_path: Option<&std::path::Path>,
    parameterize: bool,
    curvature_type: Option<CurvatureType>,
    geodesic: Option<(usize, bool)>,
    output: &std::path::Path,
    size: (u32, u32),
    azimuth: f32,
    elevation: f32,
    wireframe: bool,
    distance: Option<f32>,
) {
    let (width, height) = size;
    let mut renderer = pollster::block_on(Renderer::new_offscreen(width, height));

    let Prepared {
        mesh,
        uv_map,
        vertex_colors,
    } = prepare_mesh(mesh_path, parameterize, curvature_type, geodesic);

    let gpu_mesh = GpuMesh::from_halfedge_mesh(
        renderer.device(),
        &mesh,
        uv_map.as_ref(),
        vertex_colors.as_ref(),
    );

    let has_colors = vertex_colors.is_some();
    if !has_colors {
        if let Some(path) = texture_path {
            if let Err(e) = renderer.load_texture(path) {
                eprintln!("Failed to load texture: {e}");
            }
        }
    }

    // Frame the mesh, then swing the camera to the requested angle so successive
    // figures can be given a consistent look.
    //
    // `--distance` overrides the automatic framing, which is essential for a
    // before/after pair: auto-framing scales each mesh to fill the frame, so a
    // shrunken result renders at the same apparent size as the original and the
    // shrinkage becomes invisible. Pass the same distance to both to compare them.
    let mut camera = OrbitCamera::default();
    let dist = distance.unwrap_or(gpu_mesh.radius * 2.5);
    camera.reset(gpu_mesh.center, dist);
    camera.rotate(azimuth, elevation);
    if distance.is_none() {
        println!("auto camera distance: {dist:.4}");
    }

    renderer
        .render(&gpu_mesh, &camera, wireframe, true, true, true)
        .expect("offscreen render failed");

    let pixels = renderer
        .read_pixels()
        .expect("an offscreen renderer should have pixels to read");

    let image = image::RgbaImage::from_raw(width, height, pixels)
        .expect("pixel buffer should match the requested size");
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    image
        .save(output)
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", output.display()));

    println!(
        "Wrote {} ({}x{}, {} vertices, {} faces)",
        output.display(),
        width,
        height,
        mesh.num_vertices(),
        mesh.num_faces()
    );
}

/// A mesh plus whatever per-vertex data was requested alongside it.
struct Prepared {
    mesh: HalfEdgeMesh,
    uv_map: Option<UVMap>,
    vertex_colors: Option<VertexColors>,
}

/// Load a mesh and derive UVs and vertex colours from the requested options.
///
/// Shared by the interactive path and `--screenshot`, so a rendered image shows
/// exactly what the window would.
fn prepare_mesh(
    mesh_path: &str,
    parameterize: bool,
    curvature_type: Option<CurvatureType>,
    geodesic: Option<(usize, bool)>,
) -> Prepared {
    log::info!("Loading mesh from: {}", mesh_path);
    let (mesh, file_uvs): (HalfEdgeMesh, Option<UVMap>) =
        obj::load_with_uvs(mesh_path).expect("Failed to load mesh");
    log::info!(
        "Loaded mesh: {} vertices, {} faces",
        mesh.num_vertices(),
        mesh.num_faces()
    );

    let uv_map: Option<UVMap> = if let Some(uvs) = file_uvs {
        log::info!("Loaded UV coordinates from file");
        Some(uvs)
    } else if parameterize {
        log::info!("Computing UV parameterization (cylindrical projection)...");
        let uvs = cylindrical_projection(&mesh);
        log::info!("UV parameterization complete");
        Some(uvs)
    } else {
        None
    };

    let vertex_colors: Option<VertexColors> = if let Some((source, use_dijkstra)) = geodesic {
        use morsel::algo::geodesic;
        use morsel::mesh::VertexId;

        if source >= mesh.num_vertices() {
            eprintln!(
                "Error: --geodesic source {source} is out of range (mesh has {} vertices)",
                mesh.num_vertices()
            );
            std::process::exit(1);
        }
        let src = VertexId::new(source);

        let distances = if use_dijkstra {
            log::info!("Computing graph distances by Dijkstra from vertex {source}...");
            geodesic::dijkstra(&mesh, src, &geodesic::DijkstraOptions::default())
                .distances()
                .to_vec()
        } else {
            log::info!("Computing geodesic distances by the heat method from vertex {source}...");
            match geodesic::heat_method(&mesh, src, &geodesic::HeatMethodOptions::default()) {
                Ok(r) => r.distances().to_vec(),
                Err(e) => {
                    eprintln!("Error: heat method failed: {e}");
                    std::process::exit(1);
                }
            }
        };

        let unreachable = distances.iter().filter(|d| !d.is_finite()).count();
        if unreachable > 0 {
            eprintln!(
                "warning: {unreachable} vertices are unreachable from vertex {source} \
                 and are drawn grey; the mesh has more than one connected component."
            );
        }
        Some(geodesic_to_vertex_colors(&distances, 12.0))
    } else if let Some(curv_type) = curvature_type {
        log::info!("Computing curvature...");
        let curv_result = curvature::compute_curvature(&mesh);

        let curvature_values: Vec<f64> = match curv_type {
            CurvatureType::Mean => {
                log::info!("Using mean curvature");
                curv_result.mean_values().to_vec()
            }
            CurvatureType::Gaussian => {
                log::info!("Using Gaussian curvature");
                curv_result.gaussian_values().to_vec()
            }
        };

        let nan_count = curvature_values.iter().filter(|v| !v.is_finite()).count();
        if nan_count > 0 {
            eprintln!("WARNING: {} vertices have NaN/Inf curvature!", nan_count);
        }

        log::info!("Smoothing curvature values...");
        let smoothed = smooth_vertex_values(&mesh, &curvature_values, 2);

        log::info!("Computing vertex colors...");
        let colors = curvature_to_vertex_colors(&smoothed);
        log::info!("Vertex colors computed");
        Some(colors)
    } else {
        None
    };

    Prepared {
        mesh,
        uv_map,
        vertex_colors,
    }
}

/// Convert geodesic distances to vertex colours: a sequential ramp with periodic
/// shading that reads as isolines.
///
/// Isolines are the point of a distance figure — evenly spaced rings mean the field
/// really is a distance, and rings that bunch or kink show where it is not. The
/// banding is sinusoidal in the distance rather than a hard threshold, because
/// colours are per-vertex and interpolated across faces, so a sharp band would
/// alias into whatever the triangulation happens to be.
///
/// Unreachable vertices — a different connected component — are flat grey rather
/// than clamped to the far end of the ramp, so "no path" does not masquerade as
/// "very far".
fn geodesic_to_vertex_colors(distances: &[f64], bands: f64) -> VertexColors {
    let max = distances
        .iter()
        .copied()
        .filter(|d| d.is_finite())
        .fold(0.0_f64, f64::max);
    if max <= 0.0 {
        return vec![[0.8, 0.8, 0.8]; distances.len()];
    }

    distances
        .iter()
        .map(|&d| {
            if !d.is_finite() {
                return [0.34, 0.34, 0.38];
            }
            let t = (d / max).clamp(0.0, 1.0);
            let base = sequential_color(t);
            let phase = (t * bands * std::f64::consts::TAU).cos();
            let shade = (0.78 + 0.22 * phase) as f32;
            [base[0] * shade, base[1] * shade, base[2] * shade]
        })
        .collect()
}

/// A sequential blue-to-pale-yellow ramp, ordered so that lightness increases
/// monotonically with `t`. Monotone lightness is what makes the ramp readable in
/// greyscale and to most colour-vision deficiencies.
fn sequential_color(t: f64) -> [f32; 3] {
    const STOPS: [(f64, [f32; 3]); 5] = [
        (0.00, [0.04, 0.06, 0.28]),
        (0.25, [0.10, 0.36, 0.65]),
        (0.50, [0.13, 0.64, 0.60]),
        (0.75, [0.62, 0.80, 0.31]),
        (1.00, [0.98, 0.95, 0.996]),
    ];

    let t = t.clamp(0.0, 1.0);
    for pair in STOPS.windows(2) {
        let (t0, c0) = pair[0];
        let (t1, c1) = pair[1];
        if t <= t1 {
            let u = ((t - t0) / (t1 - t0)) as f32;
            return [
                c0[0] + (c1[0] - c0[0]) * u,
                c0[1] + (c1[1] - c0[1]) * u,
                c0[2] + (c1[2] - c0[2]) * u,
            ];
        }
    }
    STOPS[STOPS.len() - 1].1
}

/// Convert curvature values to vertex colors using a blue-white-red colormap.
///
/// - Negative curvature (concave): blue
/// - Zero curvature (flat): white
/// - Positive curvature (convex): red
fn curvature_to_vertex_colors(curvature: &[f64]) -> VertexColors {
    // Compute robust range using percentiles to avoid outliers
    let (min_curv, max_curv) = compute_robust_range(curvature);

    curvature
        .iter()
        .map(|&value| curvature_to_color(value, min_curv, max_curv))
        .collect()
}

/// Compute a robust range for curvature values using percentiles.
fn compute_robust_range(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 1.0);
    }

    let mut sorted: Vec<f64> = values.iter().filter(|v| v.is_finite()).copied().collect();
    if sorted.is_empty() {
        return (0.0, 1.0);
    }

    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Use 5th and 95th percentiles
    let low_idx = (sorted.len() as f64 * 0.05) as usize;
    let high_idx = ((sorted.len() as f64 * 0.95) as usize).min(sorted.len() - 1);

    let min = sorted[low_idx];
    let max = sorted[high_idx];

    // Ensure we have a valid range
    if (max - min).abs() < 1e-10 {
        (min - 1.0, max + 1.0)
    } else {
        (min, max)
    }
}

/// Map a curvature value to an RGB color using a blue-white-red diverging colormap.
fn curvature_to_color(value: f64, min: f64, max: f64) -> [f32; 3] {
    // Handle NaN/infinity - use bright magenta to make them visible
    if !value.is_finite() {
        return [1.0, 0.0, 1.0]; // Magenta for NaN/Inf
    }

    // Normalize to [0, 1]
    let normalized = if max > min {
        (value - min) / (max - min)
    } else {
        0.5
    };

    // Clamp to [0, 1]
    let t = normalized.clamp(0.0, 1.0);

    // Convert to [-1, 1] for diverging colormap
    let diverging = t * 2.0 - 1.0;

    // Blue-white-red diverging colormap
    // Keep colors bright by using linear interpolation but with a minimum brightness
    let (r, g, b) = if diverging < 0.0 {
        // Blue side (negative/low curvature)
        // Goes from saturated blue (0.3, 0.3, 1.0) to white (1,1,1)
        let s = 0.3 + 0.7 * (1.0 + diverging); // Range [0.3, 1.0] to keep it bright
        (s, s, 1.0)
    } else {
        // Red side (positive/high curvature)
        // Goes from white (1,1,1) to saturated red (1.0, 0.3, 0.3)
        let s = 0.3 + 0.7 * (1.0 - diverging); // Range [0.3, 1.0] to keep it bright
        (1.0, s, s)
    };

    [r as f32, g as f32, b as f32]
}

/// Smooth per-vertex values using Laplacian smoothing.
///
/// Each vertex's value is averaged with its neighbors' values.
fn smooth_vertex_values(mesh: &HalfEdgeMesh, values: &[f64], iterations: usize) -> Vec<f64> {
    let mut current = values.to_vec();
    let mut next = vec![0.0; values.len()];

    for _ in 0..iterations {
        for vid in mesh.vertex_ids() {
            let idx = vid.index();
            let neighbor_indices: Vec<usize> =
                mesh.vertex_neighbors(vid).map(|n| n.index()).collect();

            if neighbor_indices.is_empty() {
                next[idx] = current[idx];
            } else {
                // Average of neighbors (filter out non-finite values)
                let mut neighbor_sum = 0.0;
                let mut neighbor_count = 0usize;
                for &ni in &neighbor_indices {
                    let val = current[ni];
                    if val.is_finite() {
                        neighbor_sum += val;
                        neighbor_count += 1;
                    }
                }

                if neighbor_count > 0 {
                    // Blend: 50% self, 50% neighbors
                    let self_val = if current[idx].is_finite() {
                        current[idx]
                    } else {
                        neighbor_sum / neighbor_count as f64
                    };
                    next[idx] = 0.5 * self_val + 0.5 * (neighbor_sum / neighbor_count as f64);
                } else {
                    next[idx] = current[idx];
                }
            }
        }
        std::mem::swap(&mut current, &mut next);
    }

    current
}
