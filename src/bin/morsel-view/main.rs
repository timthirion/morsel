//! 3D mesh viewer binary for morsel.
//!
//! Usage: cargo run --bin viewer --features viewer -- <mesh_file>
//!
//! Controls:
//! - Left mouse drag: Rotate camera
//! - Scroll wheel: Zoom in/out
//! - W: Toggle wireframe mode
//! - B: Toggle backface culling
//! - R: Reset camera
//! - Escape: Quit

mod camera;
mod mesh_gpu;
mod renderer;

use std::env;
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
use mesh_gpu::GpuMesh;
use renderer::Renderer;

use morsel::io;
use morsel::mesh::HalfEdgeMesh;

/// Application state.
struct App {
    /// Path to the mesh file to load.
    mesh_path: String,
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
    /// Whether the left mouse button is pressed.
    mouse_pressed: bool,
    /// Last mouse position.
    last_mouse_pos: Option<PhysicalPosition<f64>>,
}

impl App {
    fn new(mesh_path: String) -> Self {
        Self {
            mesh_path,
            window: None,
            renderer: None,
            gpu_mesh: None,
            camera: OrbitCamera::default(),
            wireframe: false,
            backface_culling: true, // Enabled by default
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
        let renderer = pollster::block_on(Renderer::new(window.clone()));

        // Load mesh
        log::info!("Loading mesh from: {}", self.mesh_path);
        let mesh: HalfEdgeMesh = io::load(&self.mesh_path).expect("Failed to load mesh");
        log::info!(
            "Loaded mesh: {} vertices, {} faces",
            mesh.num_vertices(),
            mesh.num_faces()
        );

        // Create GPU mesh
        let gpu_mesh = GpuMesh::from_halfedge_mesh(renderer.device(), &mesh);

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
                    match renderer.render(gpu_mesh, &self.camera, self.wireframe, self.backface_culling) {
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
        eprintln!("Usage: {} <mesh_file>", args[0]);
        eprintln!("Supported formats: .obj, .stl, .ply, .gltf, .glb");
        eprintln!();
        eprintln!("Controls:");
        eprintln!("  Left mouse drag: Rotate camera");
        eprintln!("  Scroll wheel:    Zoom in/out");
        eprintln!("  W:               Toggle wireframe");
        eprintln!("  B:               Toggle backface culling");
        eprintln!("  R:               Reset camera");
        eprintln!("  Escape:          Quit");
        std::process::exit(1);
    }

    let mesh_path = args[1].clone();

    // Create event loop and run app
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = App::new(mesh_path);
    event_loop.run_app(&mut app).expect("Event loop error");
}
