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
    ) -> Self {
        // Disable backface culling by default for curvature visualization
        // (meshes often have inconsistent winding at high-curvature areas)
        let backface_culling = curvature_type.is_none();

        Self {
            mesh_path,
            texture_path,
            parameterize,
            curvature_type,
            window: None,
            renderer: None,
            gpu_mesh: None,
            camera: OrbitCamera::default(),
            wireframe: false,
            backface_culling,
            textured: true, // Enabled by default when texture is available
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

        // Load mesh (and UVs if present in file)
        log::info!("Loading mesh from: {}", self.mesh_path);
        let (mesh, file_uvs): (HalfEdgeMesh, Option<UVMap>) =
            obj::load_with_uvs(&self.mesh_path).expect("Failed to load mesh");
        log::info!(
            "Loaded mesh: {} vertices, {} faces",
            mesh.num_vertices(),
            mesh.num_faces()
        );

        // Use UVs from file, or compute them if requested
        let uv_map: Option<UVMap> = if let Some(uvs) = file_uvs {
            log::info!("Loaded UV coordinates from file");
            Some(uvs)
        } else if self.parameterize {
            log::info!("Computing UV parameterization (cylindrical projection)...");
            let uvs = cylindrical_projection(&mesh);
            log::info!("UV parameterization complete");
            Some(uvs)
        } else {
            None
        };

        // Compute vertex colors from curvature if requested
        let vertex_colors: Option<VertexColors> = if let Some(curv_type) = self.curvature_type {
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

            // Warn if there are NaN/Inf curvature values
            let nan_count = curvature_values.iter().filter(|v| !v.is_finite()).count();
            if nan_count > 0 {
                eprintln!("WARNING: {} vertices have NaN/Inf curvature!", nan_count);
            }

            // Smooth curvature values to reduce noise
            log::info!("Smoothing curvature values...");
            let smoothed = smooth_vertex_values(&mesh, &curvature_values, 2);

            log::info!("Computing vertex colors...");
            let colors = curvature_to_vertex_colors(&smoothed);
            log::info!("Vertex colors computed");
            Some(colors)
        } else {
            None
        };

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
                    match renderer.render(gpu_mesh, &self.camera, self.wireframe, self.backface_culling, self.textured, self.show_colors) {
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
        eprintln!("  --curvature <type>     Visualize curvature as vertex colors (mean or gaussian)");
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
                            eprintln!("Error: --curvature requires 'mean' or 'gaussian', got '{}'", other);
                            std::process::exit(1);
                        }
                    };
                    i += 2;
                } else {
                    eprintln!("Error: --curvature requires 'mean' or 'gaussian'");
                    std::process::exit(1);
                }
            }
            _ => {
                eprintln!("Unknown option: {}", args[i]);
                std::process::exit(1);
            }
        }
    }

    // Create event loop and run app
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = App::new(mesh_path, texture_path, parameterize, curvature_type);
    event_loop.run_app(&mut app).expect("Event loop error");
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
fn smooth_vertex_values(
    mesh: &HalfEdgeMesh,
    values: &[f64],
    iterations: usize,
) -> Vec<f64> {
    let mut current = values.to_vec();
    let mut next = vec![0.0; values.len()];

    for _ in 0..iterations {
        for vid in mesh.vertex_ids() {
            let idx = vid.index();
            let neighbor_indices: Vec<usize> = mesh
                .vertex_neighbors(vid)
                .map(|n| n.index())
                .collect();

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
