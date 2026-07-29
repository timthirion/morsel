//! Morsel CLI - mesh processing command-line tool.
//!
//! Usage: morsel <COMMAND> [OPTIONS] <INPUT> [OUTPUT]
//!
//! Run `morsel --help` for available commands.

use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use clap::{Parser, Subcommand, ValueEnum};

use morsel::algo::{
    curvature, decimate, parameterize, remesh, smooth, subdivide, Progress,
};
use morsel::io;
use morsel::mesh::HalfEdgeMesh;

#[derive(Parser)]
#[command(name = "morsel")]
#[command(author, version, about = "Mesh processing CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Display mesh information
    Info {
        /// Input mesh file
        input: PathBuf,

        /// Show curvature statistics
        #[arg(long)]
        curvature: bool,
    },

    /// Smooth a mesh
    Smooth {
        /// Input mesh file
        input: PathBuf,

        /// Output mesh file
        output: PathBuf,

        /// Smoothing method
        #[arg(short, long, value_enum, default_value = "laplacian")]
        method: SmoothMethod,

        /// Number of iterations
        #[arg(short, long, default_value = "1")]
        iterations: usize,

        /// Smoothing factor (0.0 to 1.0)
        #[arg(short, long, default_value = "0.5")]
        lambda: f64,

        /// Allow boundary vertices to move
        #[arg(long)]
        move_boundary: bool,

        /// Use single-threaded execution (for benchmarking)
        #[arg(long)]
        sequential: bool,
    },

    /// Subdivide a mesh
    Subdivide {
        /// Input mesh file
        input: PathBuf,

        /// Output mesh file
        output: PathBuf,

        /// Subdivision method
        #[arg(short, long, value_enum, default_value = "loop")]
        method: SubdivideMethod,

        /// Number of subdivision iterations
        #[arg(short, long, default_value = "1")]
        iterations: usize,

        /// Use single-threaded execution (for benchmarking)
        #[arg(long)]
        sequential: bool,
    },

    /// Decimate (simplify) a mesh
    Decimate {
        /// Input mesh file
        input: PathBuf,

        /// Output mesh file
        output: PathBuf,

        /// Target number of faces
        #[arg(short = 'f', long, conflicts_with = "ratio")]
        faces: Option<usize>,

        /// Target ratio of faces to keep (0.0 to 1.0)
        #[arg(short, long, default_value = "0.5")]
        ratio: f64,

        /// Allow boundary edges to be collapsed
        #[arg(long)]
        collapse_boundary: bool,

        /// Use single-threaded execution (for benchmarking)
        #[arg(long)]
        sequential: bool,
    },

    /// Compute UV coordinates and write a UV-bearing mesh
    Parameterize {
        /// Input mesh file
        input: PathBuf,

        /// Output mesh file (UVs written as `vt` entries when
        /// the format supports them — `.obj` does, `.stl` does
        /// not).
        output: PathBuf,

        /// Parameterization method
        #[arg(short, long, value_enum, default_value = "cylindrical")]
        method: ParameterizeMethod,

        /// Optional material/texture name for the MTL reference
        #[arg(long)]
        material: Option<String>,
    },

    /// Remesh to improve triangle quality
    Remesh {
        /// Input mesh file
        input: PathBuf,

        /// Output mesh file
        output: PathBuf,

        /// Remeshing method
        #[arg(short, long, value_enum, default_value = "isotropic")]
        method: RemeshMethod,

        /// Target edge length (default: average edge length)
        #[arg(short = 'l', long)]
        target_length: Option<f64>,

        /// Number of iterations
        #[arg(short, long, default_value = "5")]
        iterations: usize,

        /// Use single-threaded execution (for benchmarking)
        #[arg(long)]
        sequential: bool,
    },
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum SmoothMethod {
    /// Uniform Laplacian smoothing
    Laplacian,
    /// Taubin smoothing (shrinkage-resistant)
    Taubin,
    /// Cotangent-weighted Laplacian smoothing
    Cotangent,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum SubdivideMethod {
    /// Loop subdivision (for triangle meshes)
    Loop,
    /// Catmull-Clark subdivision (for quad meshes)
    CatmullClark,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum RemeshMethod {
    /// Isotropic remeshing (uniform edge lengths)
    Isotropic,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum ParameterizeMethod {
    /// Cylindrical projection (works on closed meshes; seam along
    /// the back of the projection axis where `atan2` wraps)
    Cylindrical,
    /// LSCM — Least Squares Conformal Maps. Angle-preserving;
    /// requires the mesh to have boundary (disk topology). Closed
    /// meshes need a manual cut before this works.
    Lscm,
    /// ARAP — As-Rigid-As-Possible. Higher quality than LSCM but
    /// also needs boundary. Iterative.
    Arap,
    /// OMT — Optimal Mass Transport. Area-preserving; runs LSCM
    /// first and corrects its area distortion. Also needs boundary.
    Omt,
}

/// Methods that solve over a UV domain and therefore require the mesh
/// to have boundary (disk topology). Cylindrical projection does not.
impl ParameterizeMethod {
    fn requires_boundary(self) -> bool {
        !matches!(self, ParameterizeMethod::Cylindrical)
    }
}

fn main() {
    let cli = Cli::parse();

    if let Err(e) = run(cli) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        Commands::Info { input, curvature: show_curvature } => {
            cmd_info(&input, show_curvature)?;
        }

        Commands::Smooth {
            input,
            output,
            method,
            iterations,
            lambda,
            move_boundary,
            sequential,
        } => {
            cmd_smooth(&input, &output, method, iterations, lambda, move_boundary, sequential)?;
        }

        Commands::Subdivide {
            input,
            output,
            method,
            iterations,
            sequential,
        } => {
            cmd_subdivide(&input, &output, method, iterations, sequential)?;
        }

        Commands::Decimate {
            input,
            output,
            faces,
            ratio,
            collapse_boundary,
            sequential,
        } => {
            cmd_decimate(&input, &output, faces, ratio, collapse_boundary, sequential)?;
        }

        Commands::Parameterize {
            input,
            output,
            method,
            material,
        } => {
            cmd_parameterize(&input, &output, method, material.as_deref())?;
        }

        Commands::Remesh {
            input,
            output,
            method,
            target_length,
            iterations,
            sequential,
        } => {
            cmd_remesh(&input, &output, method, target_length, iterations, sequential)?;
        }
    }

    Ok(())
}

/// Create a progress reporter that displays a progress bar on the terminal.
fn create_progress() -> Progress {
    let max_percent = Arc::new(AtomicUsize::new(0)); // Track highest percent seen (monotonic)

    Progress::new(move |current, total, message| {
        if total == 0 {
            return;
        }

        // Use rounding instead of truncation for smoother progress
        let raw_percent = if current >= total {
            100
        } else {
            ((current * 100) + (total / 2)) / total
        };

        // Ensure monotonic progress: only increase, never decrease
        // This prevents bouncing when sub-tasks transition or estimates change
        let (percent, increased) = loop {
            let old_max = max_percent.load(Ordering::Relaxed);
            let new_max = old_max.max(raw_percent);
            if new_max == old_max {
                break (old_max, false);
            }
            match max_percent.compare_exchange_weak(
                old_max,
                new_max,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break (new_max, true),
                Err(_) => continue,
            }
        };

        // Only update display if percent increased (reduce flickering)
        if !increased && percent != 100 {
            return;
        }

        // Create progress bar
        let bar_width = 30;
        let filled = (percent * bar_width) / 100;
        let empty = bar_width - filled;

        let bar: String = std::iter::repeat('=').take(filled).collect();
        let space: String = std::iter::repeat(' ').take(empty).collect();

        // Use carriage return to overwrite the line
        eprint!("\r[{}{}] {:3}% {}", bar, space, percent, message);

        // Flush to ensure immediate display
        let _ = std::io::stderr().flush();

        // Print newline on completion
        if current >= total {
            eprintln!();
        }
    })
}

fn cmd_info(input: &PathBuf, show_curvature: bool) -> Result<(), Box<dyn std::error::Error>> {
    let mesh: HalfEdgeMesh = io::load(input)?;

    println!("File: {}", input.display());
    println!("Vertices: {}", mesh.num_vertices());
    println!("Faces: {}", mesh.num_faces());
    println!("Half-edges: {}", mesh.num_halfedges());

    // Compute some statistics
    let mut total_area = 0.0;
    let mut min_area = f64::MAX;
    let mut max_area = 0.0_f64;

    for fid in mesh.face_ids() {
        let area = mesh.face_area(fid);
        total_area += area;
        min_area = min_area.min(area);
        max_area = max_area.max(area);
    }

    println!("Surface area: {:.6}", total_area);
    println!("Face area range: [{:.6}, {:.6}]", min_area, max_area);

    // Bounding box
    if let Some((min, max)) = mesh.bounding_box() {
        println!("Bounding box: ({:.3}, {:.3}, {:.3}) to ({:.3}, {:.3}, {:.3})",
            min.x, min.y, min.z, max.x, max.y, max.z);
        let diag = max - min;
        println!("Dimensions: {:.3} x {:.3} x {:.3}", diag.x, diag.y, diag.z);
    }

    // Edge length statistics
    let avg_edge = remesh::average_edge_length(&mesh);
    println!("Average edge length: {:.6}", avg_edge);

    // Check mesh type
    if mesh.is_triangle_mesh() {
        println!("Mesh type: Triangle mesh");
    } else if mesh.is_quad_mesh() {
        println!("Mesh type: Quad mesh");
    } else {
        println!("Mesh type: Mixed polygon mesh");
    }

    // Boundary info
    let boundary_verts: Vec<_> = mesh.vertex_ids()
        .filter(|&v| mesh.is_boundary_vertex(v))
        .collect();
    if boundary_verts.is_empty() {
        println!("Topology: Closed (no boundary)");
    } else {
        println!("Topology: Open ({} boundary vertices)", boundary_verts.len());
    }

    // Curvature statistics
    if show_curvature {
        println!("\nCurvature:");
        let curv = curvature::compute_curvature(&mesh);

        let gaussian = curv.gaussian_values();
        let mean = curv.mean_values();

        let g_min = gaussian.iter().cloned().fold(f64::INFINITY, f64::min);
        let g_max = gaussian.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let g_avg: f64 = gaussian.iter().sum::<f64>() / gaussian.len() as f64;

        let m_min = mean.iter().cloned().fold(f64::INFINITY, f64::min);
        let m_max = mean.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let m_avg: f64 = mean.iter().sum::<f64>() / mean.len() as f64;

        println!("  Gaussian: min={:.4}, max={:.4}, avg={:.4}", g_min, g_max, g_avg);
        println!("  Mean:     min={:.4}, max={:.4}, avg={:.4}", m_min, m_max, m_avg);

        // Gauss-Bonnet check
        let total_gaussian: f64 = gaussian.iter().sum();
        let euler_from_curv = total_gaussian / (2.0 * std::f64::consts::PI);
        println!("  Gauss-Bonnet Euler characteristic: {:.2}", euler_from_curv);
    }

    Ok(())
}

fn cmd_smooth(
    input: &PathBuf,
    output: &PathBuf,
    method: SmoothMethod,
    iterations: usize,
    lambda: f64,
    move_boundary: bool,
    sequential: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut mesh: HalfEdgeMesh = io::load(input)?;

    println!("Loaded: {} vertices, {} faces", mesh.num_vertices(), mesh.num_faces());

    let options = smooth::SmoothOptions {
        iterations,
        lambda,
        preserve_boundary: !move_boundary,
        parallel: !sequential,
    };

    let mode = if sequential { "sequential" } else { "parallel" };
    let progress = create_progress();

    let start = Instant::now();
    match method {
        SmoothMethod::Laplacian => {
            println!("Applying Laplacian smoothing ({} iterations, lambda={}, {})...", iterations, lambda, mode);
            smooth::laplacian_smooth_with_progress(&mut mesh, &options, &progress);
        }
        SmoothMethod::Taubin => {
            println!("Applying Taubin smoothing ({} iterations, lambda={}, {})...", iterations, lambda, mode);
            smooth::taubin_smooth_with_progress(&mut mesh, &options, &progress);
        }
        SmoothMethod::Cotangent => {
            println!("Applying cotangent smoothing ({} iterations, lambda={}, {})...", iterations, lambda, mode);
            smooth::cotangent_smooth_with_progress(&mut mesh, &options, &progress);
        }
    }
    let elapsed = start.elapsed();

    io::save(&mesh, output)?;
    println!("Saved: {} ({:.2?})", output.display(), elapsed);

    Ok(())
}

fn cmd_subdivide(
    input: &PathBuf,
    output: &PathBuf,
    method: SubdivideMethod,
    iterations: usize,
    sequential: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut mesh: HalfEdgeMesh = io::load(input)?;

    println!("Loaded: {} vertices, {} faces", mesh.num_vertices(), mesh.num_faces());

    let options = subdivide::SubdivideOptions::new(iterations).with_parallel(!sequential);
    let mode = if sequential { "sequential" } else { "parallel" };
    let progress = create_progress();

    let start = Instant::now();
    match method {
        SubdivideMethod::Loop => {
            println!("Applying Loop subdivision ({} iterations, {})...", iterations, mode);
            subdivide::loop_subdivide_with_progress(&mut mesh, &options, &progress);
        }
        SubdivideMethod::CatmullClark => {
            println!("Applying Catmull-Clark subdivision ({} iterations, {})...", iterations, mode);
            subdivide::catmull_clark_subdivide_with_progress(&mut mesh, &options, &progress);
        }
    }
    let elapsed = start.elapsed();

    println!("Result: {} vertices, {} faces", mesh.num_vertices(), mesh.num_faces());
    io::save(&mesh, output)?;
    println!("Saved: {} ({:.2?})", output.display(), elapsed);

    Ok(())
}

fn cmd_decimate(
    input: &PathBuf,
    output: &PathBuf,
    faces: Option<usize>,
    ratio: f64,
    collapse_boundary: bool,
    sequential: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut mesh: HalfEdgeMesh = io::load(input)?;

    println!("Loaded: {} vertices, {} faces", mesh.num_vertices(), mesh.num_faces());

    let mode = if sequential { "sequential" } else { "parallel" };
    let options = if let Some(target_faces) = faces {
        println!("Decimating to {} faces ({})...", target_faces, mode);
        decimate::DecimateOptions::with_target_faces(target_faces)
            .with_preserve_boundary(!collapse_boundary)
            .with_parallel(!sequential)
    } else {
        println!("Decimating to {:.0}% of faces ({})...", ratio * 100.0, mode);
        decimate::DecimateOptions::with_target_ratio(ratio)
            .with_preserve_boundary(!collapse_boundary)
            .with_parallel(!sequential)
    };

    let progress = create_progress();

    let start = Instant::now();
    decimate::qem_decimate_with_progress(&mut mesh, &options, &progress);
    let elapsed = start.elapsed();

    println!("Result: {} vertices, {} faces", mesh.num_vertices(), mesh.num_faces());
    io::save(&mesh, output)?;
    println!("Saved: {} ({:.2?})", output.display(), elapsed);

    Ok(())
}

fn cmd_remesh(
    input: &PathBuf,
    output: &PathBuf,
    method: RemeshMethod,
    target_length: Option<f64>,
    iterations: usize,
    sequential: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut mesh: HalfEdgeMesh = io::load(input)?;

    println!("Loaded: {} vertices, {} faces", mesh.num_vertices(), mesh.num_faces());

    let avg_edge = remesh::average_edge_length(&mesh);
    let target = target_length.unwrap_or(avg_edge);

    println!("Current average edge length: {:.6}", avg_edge);
    println!("Target edge length: {:.6}", target);

    let mode = if sequential { "sequential" } else { "parallel" };
    let progress = create_progress();

    let start = Instant::now();
    match method {
        RemeshMethod::Isotropic => {
            println!("Applying isotropic remeshing ({} iterations, {})...", iterations, mode);
            let options = remesh::RemeshOptions::with_target_length(target)
                .with_iterations(iterations)
                .with_parallel(!sequential);
            remesh::isotropic_remesh_with_progress(&mut mesh, &options, &progress);
        }
    }
    let elapsed = start.elapsed();

    let new_avg = remesh::average_edge_length(&mesh);
    println!("Result: {} vertices, {} faces (avg edge: {:.6})",
        mesh.num_vertices(), mesh.num_faces(), new_avg);
    io::save(&mesh, output)?;
    println!("Saved: {} ({:.2?})", output.display(), elapsed);

    Ok(())
}

fn cmd_parameterize(
    input: &PathBuf,
    output: &PathBuf,
    method: ParameterizeMethod,
    material: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    use morsel::io::obj as obj_io;

    let mesh: HalfEdgeMesh = io::load(input)?;
    println!("Loaded: {} vertices, {} faces", mesh.num_vertices(), mesh.num_faces());

    // Every method except cylindrical projection solves over a UV domain
    // pinned to the boundary, so a closed mesh has to be cut first.
    if method.requires_boundary() {
        let boundary_count = mesh
            .vertex_ids()
            .filter(|&v| mesh.is_boundary_vertex(v))
            .count();
        if boundary_count == 0 {
            return Err("Mesh has no boundary; this method requires disk topology (open mesh). \
                        Cut the mesh first, or use `--method cylindrical`."
                .into());
        }
        println!("Boundary vertices: {}", boundary_count);
    }

    let start = Instant::now();
    let uvs = match method {
        ParameterizeMethod::Cylindrical => {
            println!("Computing cylindrical UV projection around Y axis...");
            parameterize::cylindrical_projection(&mesh)
        }
        ParameterizeMethod::Lscm => {
            println!("Computing LSCM parameterization (conformal/angle-preserving)...");
            parameterize::lscm(&mesh, &parameterize::LSCMOptions::default())
                .map_err(|e| format!("LSCM failed: {e}"))?
        }
        ParameterizeMethod::Arap => {
            println!("Computing ARAP parameterization (as-rigid-as-possible)...");
            parameterize::arap(&mesh, &parameterize::ARAPOptions::default())
                .map_err(|e| format!("ARAP failed: {e}"))?
        }
        ParameterizeMethod::Omt => {
            println!("Computing LSCM parameterization as base...");
            let lscm_uvs = parameterize::lscm(&mesh, &parameterize::LSCMOptions::default())
                .map_err(|e| format!("LSCM (OMT base) failed: {e}"))?;

            let (min_lscm, max_lscm, rms_lscm) =
                parameterize::compute_area_distortion(&mesh, &lscm_uvs);
            println!(
                "LSCM area distortion: min={:.3}, max={:.3}, rms={:.4}",
                min_lscm, max_lscm, rms_lscm
            );

            println!("Applying OMT area-preserving correction...");
            let opts = parameterize::OMTOptions::default();
            let (omt_uvs, report) = parameterize::omt_with_report(&mesh, &lscm_uvs, &opts)
                .map_err(|e| format!("OMT failed: {e}"))?;

            let (min_omt, max_omt, rms_omt) =
                parameterize::compute_area_distortion(&mesh, &omt_uvs);
            println!(
                "OMT area distortion:  min={:.3}, max={:.3}, rms={:.4}",
                min_omt, max_omt, rms_omt
            );
            println!(
                "OMT transport: {} iterations, worst cell area error {:.2e}{}",
                report.iterations,
                report.max_relative_error,
                if report.converged {
                    " (converged)"
                } else {
                    " (hit iteration limit)"
                }
            );

            if rms_omt >= rms_lscm {
                eprintln!(
                    "warning: OMT did not improve on the conformal input \
                     (rms {rms_lscm:.4} -> {rms_omt:.4}); consider using `--method lscm`."
                );
            }

            omt_uvs
        }
    };
    let elapsed = start.elapsed();

    // OMT already printed its own before/after comparison above.
    if method != ParameterizeMethod::Omt {
        let (min_ratio, max_ratio, rms_error) = parameterize::compute_area_distortion(&mesh, &uvs);
        println!(
            "Area distortion: min={:.3}, max={:.3}, rms={:.4}",
            min_ratio, max_ratio, rms_error
        );
    }

    let ext = output.extension().and_then(|e| e.to_str()).unwrap_or("");
    match ext.to_ascii_lowercase().as_str() {
        "obj" => {
            obj_io::save_with_uvs(&mesh, &uvs, output, material)?;
        }
        other => {
            return Err(format!(
                "parameterize output must be .obj for UV-preserving save (got .{other}); \
                 OBJ is the only format `save_with_uvs` supports today."
            )
            .into());
        }
    }
    println!("Saved: {} ({:.2?})", output.display(), elapsed);

    Ok(())
}
