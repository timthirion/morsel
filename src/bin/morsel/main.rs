//! Morsel CLI - mesh processing command-line tool.
//!
//! Usage: morsel <COMMAND> [OPTIONS] <INPUT> [OUTPUT]
//!
//! Run `morsel --help` for available commands.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use clap::{Parser, Subcommand, ValueEnum};

use morsel::algo::{
    curvature, decimate, parameterize, quality, remesh, smooth, subdivide, Progress,
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

        /// Cut the mesh to disk topology first, so that methods
        /// requiring boundary work on closed or multi-hole meshes.
        /// The written mesh is the cut one, since the UVs index it.
        #[arg(long)]
        cut: bool,

        /// Also write the UV layout as a flat mesh, with each vertex
        /// at (u, v, 0). Useful for seeing the flattening itself:
        /// folds and collapsed regions are plain here and nearly
        /// invisible on the textured 3D model.
        #[arg(long, value_name = "FILE")]
        layout: Option<PathBuf>,
    },

    /// Report triangle-quality statistics
    Quality {
        /// Input mesh file
        input: PathBuf,
    },

    /// Cut a mesh open until it has one boundary loop (disk topology)
    Cut {
        /// Input mesh file
        input: PathBuf,

        /// Output mesh file
        output: PathBuf,
    },

    /// Compute geodesic distances from a source vertex
    Geodesic {
        /// Input mesh file
        input: PathBuf,

        /// Source vertex index
        #[arg(short, long, default_value = "0")]
        source: usize,

        /// Which solver to use
        #[arg(short, long, value_enum, default_value = "heat")]
        method: GeodesicMethod,

        /// Also report the distance to this vertex
        #[arg(short, long)]
        target: Option<usize>,
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

        /// Target edge length (default: average edge length).
        /// Used by isotropic and anisotropic.
        #[arg(short = 'l', long)]
        target_length: Option<f64>,

        /// Target vertex count, for `--method cvt`. Must be below the
        /// input count for Lloyd's iteration to have anything to move.
        #[arg(long)]
        target_vertices: Option<usize>,

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
enum GeodesicMethod {
    /// Heat method (Crane, Weischedel & Wardetzky). Solves two linear systems and
    /// measures distance across faces, so it is not restricted to travelling along
    /// edges.
    Heat,
    /// Dijkstra over the edge graph. Exact along edges, but it can only
    /// *overestimate* a geodesic that cuts across a face — by up to 41% on a
    /// cylinder, where geodesics are helices.
    Dijkstra,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum RemeshMethod {
    /// Isotropic remeshing, aiming for uniform edge lengths. The
    /// one that reliably improves quality; see `tests/remesh_quality.rs`.
    Isotropic,
    /// Curvature-adaptive remeshing. Improves quality on some
    /// meshes and degrades it badly on others (a cylinder's worst
    /// angle drops from 43.7 to about 10), and shrinks the surface
    /// by up to 15%. Reports whether it converged.
    Anisotropic,
    /// CVT resampling by Lloyd's algorithm. Needs `--target-vertices`
    /// below the input count; without it each Voronoi cell holds one
    /// vertex and the iteration has nothing to move.
    Cvt,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum ParameterizeMethod {
    /// Cylindrical projection (works on closed meshes; seam along
    /// the back of the projection axis where `atan2` wraps)
    Cylindrical,
    /// LSCM — Least Squares Conformal Maps. Angle-preserving;
    /// requires disk topology (exactly one boundary loop). Pass
    /// --cut to open a closed or multi-hole mesh first.
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
        Commands::Info {
            input,
            curvature: show_curvature,
        } => {
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
            cmd_smooth(
                &input,
                &output,
                method,
                iterations,
                lambda,
                move_boundary,
                sequential,
            )?;
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
            cut,
            layout,
        } => {
            cmd_parameterize(
                &input,
                &output,
                method,
                material.as_deref(),
                cut,
                layout.as_deref(),
            )?;
        }

        Commands::Quality { input } => {
            cmd_quality(&input)?;
        }

        Commands::Cut { input, output } => {
            cmd_cut(&input, &output)?;
        }

        Commands::Geodesic {
            input,
            source,
            method,
            target,
        } => {
            cmd_geodesic(&input, source, method, target)?;
        }

        Commands::Remesh {
            input,
            output,
            method,
            target_length,
            target_vertices,
            iterations,
            sequential,
        } => {
            cmd_remesh(
                &input,
                &output,
                method,
                target_length,
                target_vertices,
                iterations,
                sequential,
            )?;
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

        let bar: String = "=".repeat(filled);
        let space: String = " ".repeat(empty);

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
        println!(
            "Bounding box: ({:.3}, {:.3}, {:.3}) to ({:.3}, {:.3}, {:.3})",
            min.x, min.y, min.z, max.x, max.y, max.z
        );
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
    let boundary_verts: Vec<_> = mesh
        .vertex_ids()
        .filter(|&v| mesh.is_boundary_vertex(v))
        .collect();
    if boundary_verts.is_empty() {
        println!("Topology: Closed (no boundary)");
    } else {
        println!(
            "Topology: Open ({} boundary vertices)",
            boundary_verts.len()
        );
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

        println!(
            "  Gaussian: min={:.4}, max={:.4}, avg={:.4}",
            g_min, g_max, g_avg
        );
        println!(
            "  Mean:     min={:.4}, max={:.4}, avg={:.4}",
            m_min, m_max, m_avg
        );

        // Gauss-Bonnet check
        let total_gaussian: f64 = gaussian.iter().sum();
        let euler_from_curv = total_gaussian / (2.0 * std::f64::consts::PI);
        println!(
            "  Gauss-Bonnet Euler characteristic: {:.2}",
            euler_from_curv
        );
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

    println!(
        "Loaded: {} vertices, {} faces",
        mesh.num_vertices(),
        mesh.num_faces()
    );

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
            println!(
                "Applying Laplacian smoothing ({} iterations, lambda={}, {})...",
                iterations, lambda, mode
            );
            smooth::laplacian_smooth_with_progress(&mut mesh, &options, &progress);
        }
        SmoothMethod::Taubin => {
            println!(
                "Applying Taubin smoothing ({} iterations, lambda={}, {})...",
                iterations, lambda, mode
            );
            smooth::taubin_smooth_with_progress(&mut mesh, &options, &progress);
        }
        SmoothMethod::Cotangent => {
            println!(
                "Applying cotangent smoothing ({} iterations, lambda={}, {})...",
                iterations, lambda, mode
            );
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

    println!(
        "Loaded: {} vertices, {} faces",
        mesh.num_vertices(),
        mesh.num_faces()
    );

    let options = subdivide::SubdivideOptions::new(iterations).with_parallel(!sequential);
    let mode = if sequential { "sequential" } else { "parallel" };
    let progress = create_progress();

    let start = Instant::now();
    match method {
        SubdivideMethod::Loop => {
            println!(
                "Applying Loop subdivision ({} iterations, {})...",
                iterations, mode
            );
            let report = subdivide::loop_subdivide_with_progress(&mut mesh, &options, &progress);
            warn_if_subdivision_stopped_early(&report, iterations);
        }
        SubdivideMethod::CatmullClark => {
            // A quad scheme. Say so rather than silently returning the input.
            if !mesh.is_quad_mesh() {
                return Err(
                    "Catmull-Clark requires a quad mesh; this one is not made of quads. \
                     Use `--method loop` for triangle meshes."
                        .into(),
                );
            }
            println!(
                "Applying Catmull-Clark subdivision ({} iterations, {})...",
                iterations, mode
            );
            let report =
                subdivide::catmull_clark_subdivide_with_progress(&mut mesh, &options, &progress);
            warn_if_subdivision_stopped_early(&report, iterations);
        }
    }
    let elapsed = start.elapsed();

    println!(
        "Result: {} vertices, {} faces",
        mesh.num_vertices(),
        mesh.num_faces()
    );
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

    println!(
        "Loaded: {} vertices, {} faces",
        mesh.num_vertices(),
        mesh.num_faces()
    );

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

    let faces_before = mesh.num_faces();
    let requested = faces.unwrap_or_else(|| ((faces_before as f64) * ratio).round() as usize);

    let start = Instant::now();
    let report = decimate::qem_decimate_with_progress(&mut mesh, &options, &progress);
    let elapsed = start.elapsed();

    println!(
        "Result: {} vertices, {} faces",
        mesh.num_vertices(),
        mesh.num_faces()
    );

    // The report names which case this was, rather than the shortfall being inferred
    // from the face count and every cause described as if it might apply.
    match report.outcome {
        decimate::DecimateOutcome::Completed | decimate::DecimateOutcome::NothingRequested => {}
        decimate::DecimateOutcome::Exhausted => {
            eprintln!(
                "note: stopped at {} faces rather than the requested {}. No remaining edge \
                 can be collapsed without tearing the surface — often the request is simply \
                 not reachable, and an interior edge with both endpoints on a boundary never \
                 is.",
                mesh.num_faces(),
                requested
            );
        }
        decimate::DecimateOutcome::BackedOff => {
            eprintln!(
                "warning: the requested {} faces produced a non-manifold mesh, so a milder \
                 reduction to {} was used instead ({} attempts). Individually legal \
                 collapses can still combine into a bowtie vertex.",
                requested,
                mesh.num_faces(),
                report.attempts
            );
        }
        decimate::DecimateOutcome::Refused => {
            eprintln!(
                "warning: no reduction toward {} faces produced a valid mesh after {} \
                 attempts, so the input is unchanged at {} faces. Remaining edges either \
                 fail the link condition or would produce a non-manifold mesh.",
                requested, report.attempts, faces_before
            );
        }
    }
    io::save(&mesh, output)?;
    println!("Saved: {} ({:.2?})", output.display(), elapsed);

    Ok(())
}

fn cmd_geodesic(
    input: &PathBuf,
    source: usize,
    method: GeodesicMethod,
    target: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    use morsel::algo::geodesic;
    use morsel::mesh::VertexId;

    let mesh: HalfEdgeMesh = io::load(input)?;
    println!(
        "Loaded: {} vertices, {} faces",
        mesh.num_vertices(),
        mesh.num_faces()
    );

    if source >= mesh.num_vertices() {
        return Err(format!(
            "source vertex {source} is out of range (mesh has {} vertices)",
            mesh.num_vertices()
        )
        .into());
    }
    if let Some(t) = target {
        if t >= mesh.num_vertices() {
            return Err(format!(
                "target vertex {t} is out of range (mesh has {} vertices)",
                mesh.num_vertices()
            )
            .into());
        }
    }

    let src = VertexId::new(source);
    let start = Instant::now();
    let result = match method {
        GeodesicMethod::Heat => {
            println!("Computing geodesic distances by the heat method...");
            geodesic::heat_method(&mesh, src, &geodesic::HeatMethodOptions::default())
                .map_err(|e| format!("heat method failed: {e}"))?
        }
        GeodesicMethod::Dijkstra => {
            println!("Computing graph distances by Dijkstra...");
            let opts = geodesic::DijkstraOptions {
                // Only needed to reconstruct a path, so ask for it only when one
                // was requested.
                store_predecessors: target.is_some(),
                ..Default::default()
            };
            geodesic::dijkstra(&mesh, src, &opts)
        }
    };
    let elapsed = start.elapsed();

    let reachable = result.reachable_count();
    println!("Reachable: {reachable} of {} vertices", mesh.num_vertices());
    if reachable < mesh.num_vertices() {
        eprintln!(
            "warning: {} vertices are unreachable from vertex {source}; the mesh has \
             more than one connected component.",
            mesh.num_vertices() - reachable
        );
    }

    let finite: Vec<f64> = result
        .distances()
        .iter()
        .copied()
        .filter(|d| d.is_finite())
        .collect();
    if !finite.is_empty() {
        let mean = finite.iter().sum::<f64>() / finite.len() as f64;
        println!("Distance: mean {:.6}", mean);
    }
    if let Some((v, d)) = result.farthest_vertex() {
        println!("Farthest: vertex {} at {:.6}", v.index(), d);
    }
    if let Some(t) = target {
        let tv = VertexId::new(t);
        println!("To vertex {t}: {:.6}", result.distance(tv));
        match result.path_to(tv) {
            Some(path) => println!("Path: {} vertices along edges", path.len()),
            None => match method {
                // The heat method solves for a distance field rather than searching
                // a graph, so it has no predecessors to walk back.
                GeodesicMethod::Heat => {
                    println!("Path: unavailable — the heat method is not a graph search")
                }
                GeodesicMethod::Dijkstra => {
                    println!("Path: unavailable — vertex {t} is unreachable from the source")
                }
            },
        }
    }
    println!("Done ({:.2?})", elapsed);

    Ok(())
}

fn cmd_remesh(
    input: &PathBuf,
    output: &PathBuf,
    method: RemeshMethod,
    target_length: Option<f64>,
    target_vertices: Option<usize>,
    iterations: usize,
    sequential: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut mesh: HalfEdgeMesh = io::load(input)?;

    println!(
        "Loaded: {} vertices, {} faces",
        mesh.num_vertices(),
        mesh.num_faces()
    );

    let avg_edge = remesh::average_edge_length(&mesh);
    let target = target_length.unwrap_or(avg_edge);

    println!("Current average edge length: {:.6}", avg_edge);
    println!("Target edge length: {:.6}", target);

    let mode = if sequential { "sequential" } else { "parallel" };
    let progress = create_progress();
    let before = quality::mesh_quality(&mesh);

    let start = Instant::now();
    match method {
        RemeshMethod::Isotropic => {
            println!(
                "Applying isotropic remeshing ({} iterations, {})...",
                iterations, mode
            );
            let options = remesh::RemeshOptions::with_target_length(target)
                .with_iterations(iterations)
                .with_parallel(!sequential);
            let report = remesh::isotropic_remesh_with_progress(&mut mesh, &options, &progress);
            warn_if_remesh_stopped_early(&report, iterations, "isotropic");
        }
        RemeshMethod::Anisotropic => {
            println!(
                "Applying anisotropic remeshing ({} iterations, {})...",
                iterations, mode
            );
            let options = remesh::AnisotropicOptions::new(0.5 * target, 2.0 * target)
                .with_iterations(iterations);
            let report = remesh::anisotropic_remesh_with_progress(&mut mesh, &options, &progress);
            warn_if_remesh_stopped_early(&report, iterations, "anisotropic");
        }
        RemeshMethod::Cvt => {
            let vertices = target_vertices.unwrap_or_else(|| mesh.num_vertices() * 2 / 3);
            if vertices == 0 || vertices >= mesh.num_vertices() {
                return Err(format!(
                    "--target-vertices must be between 1 and {} (exclusive); at or above \
                     the input count each Voronoi cell holds a single vertex, whose \
                     centroid is that vertex, so Lloyd's iteration has nothing to move.",
                    mesh.num_vertices()
                )
                .into());
            }
            println!(
                "Applying CVT resampling to {vertices} vertices \
                 ({iterations} Lloyd iterations)..."
            );
            let options = remesh::CvtOptions {
                target_vertices: Some(vertices),
                iterations,
                ..Default::default()
            };
            let report = remesh::cvt_remesh_with_progress(&mut mesh, &options, &progress);
            if !report.converged {
                eprintln!(
                    "warning: CVT retriangulation produced a mesh that is not manifold, so \
                     the input is unchanged. Its dual over the relaxed seeds is not valid \
                     for every seed placement; try a different --target-vertices."
                );
            }
        }
    }
    let elapsed = start.elapsed();

    let new_avg = remesh::average_edge_length(&mesh);
    println!(
        "Result: {} vertices, {} faces (avg edge: {:.6})",
        mesh.num_vertices(),
        mesh.num_faces(),
        new_avg
    );

    // Remeshing is a claim about triangle quality, so report the quality rather than
    // leaving the caller to take it on trust. Worst and mean both, because they can
    // disagree sharply: on the bunny, isotropic remeshing lifts the mean from 35.9° to
    // 51.4° while emitting a face with no usable area.
    if let (Some(before), Some(after)) = (before, quality::mesh_quality(&mesh)) {
        println!(
            "Triangle quality: worst angle {:.2}° -> {:.2}°, mean {:.2}° -> {:.2}°, \
             radius ratio {:.3} -> {:.3}, edge cv {:.3} -> {:.3}",
            before.min_angle_deg,
            after.min_angle_deg,
            before.mean_min_angle_deg,
            after.mean_min_angle_deg,
            before.mean_radius_ratio,
            after.mean_radius_ratio,
            before.edge_length_cv,
            after.edge_length_cv
        );
    }
    io::save(&mesh, output)?;
    println!("Saved: {} ({:.2?})", output.display(), elapsed);

    Ok(())
}

/// Say when a remesh stopped short. These algorithms rebuild the mesh from a face list
/// each pass, and a rebuild the half-edge representation rejects leaves the mesh at the
/// last state that worked — which looks exactly like success from the outside.
fn warn_if_remesh_stopped_early(report: &remesh::RemeshReport, requested: usize, label: &str) {
    if report.converged {
        return;
    }
    // Isotropic runs every iteration even when a pass is rejected, since the mesh stays
    // valid and later passes still help; anisotropic stops, because its non-convergence
    // means a pass bound was hit. So both "ran fewer" and "ran all, some rejected" are
    // real, and saying "stopped after 5 of 5" for the second would be nonsense.
    if report.iterations_run < requested {
        eprintln!(
            "warning: {label} remeshing stopped after {} of {requested} iteration(s). The \
             mesh is the last state it reached, not a finished result.",
            report.iterations_run
        );
    } else {
        eprintln!(
            "warning: {label} remeshing ran all {requested} iteration(s) but at least one \
             pass was rejected as non-manifold and left the mesh unchanged, so the result \
             is less refined than requested."
        );
    }
}

/// Say when subdivision stopped short, and why.
fn warn_if_subdivision_stopped_early(report: &subdivide::SubdivideReport, requested: usize) {
    match report.outcome {
        subdivide::SubdivideOutcome::Completed | subdivide::SubdivideOutcome::NothingRequested => {}
        subdivide::SubdivideOutcome::NotAQuadMesh => {
            eprintln!("warning: Catmull-Clark needs an all-quad mesh; the input is unchanged.");
        }
        subdivide::SubdivideOutcome::RebuildRejected => {
            eprintln!(
                "warning: subdivision level {} produced a mesh that is not manifold, so it \
                 stopped after {} of {requested} level(s).",
                report.iterations_run + 1,
                report.iterations_run
            );
        }
    }
}

fn cmd_quality(input: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let mesh: HalfEdgeMesh = io::load(input)?;
    let report = quality::mesh_quality(&mesh).ok_or("mesh has no faces to measure")?;
    println!("{}", input.display());
    println!("{}", report.summary());
    Ok(())
}

fn cmd_cut(input: &PathBuf, output: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let mesh: HalfEdgeMesh = io::load(input)?;
    println!(
        "Loaded: {} vertices, {} faces, {} boundary loop(s), genus {}",
        mesh.num_vertices(),
        mesh.num_faces(),
        mesh.boundary_loop_count(),
        match mesh.genus() {
            Some(g) => g.to_string(),
            None => "undefined (disconnected)".to_string(),
        }
    );

    let start = Instant::now();
    let (cut, report) = morsel::algo::cut::cut_to_disk(&mesh)?;
    println!("Cut in {:.2?}", start.elapsed());
    println!(
        "  boundary loops: {} -> {}",
        report.loops_before, report.loops_after
    );
    println!("  paths cut:      {}", report.paths_cut);
    println!("  vertices added: {}", report.vertices_added);

    io::save(&cut, output)?;
    println!("Wrote {}", output.display());
    Ok(())
}

fn cmd_parameterize(
    input: &PathBuf,
    output: &PathBuf,
    method: ParameterizeMethod,
    material: Option<&str>,
    cut: bool,
    layout: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    use morsel::io::obj as obj_io;

    let mut mesh: HalfEdgeMesh = io::load(input)?;
    println!(
        "Loaded: {} vertices, {} faces",
        mesh.num_vertices(),
        mesh.num_faces()
    );

    // Every method except cylindrical projection solves over a UV domain pinned to the
    // boundary, and needs exactly one boundary loop. Counting boundary *vertices* is
    // not the same test: the bunny has hundreds of them spread over four holes.
    if method.requires_boundary() {
        let loops = mesh.boundary_loop_count();
        println!("Boundary loops: {}", loops);
        if loops != 1 {
            if !cut {
                return Err(format!(
                    "Mesh has {loops} boundary loops; this method needs exactly one (disk \
                     topology). Pass --cut to cut it open, or use `--method cylindrical`."
                )
                .into());
            }
            let (opened, report) = morsel::algo::cut::cut_to_disk(&mesh)?;
            println!(
                "Cut along {} path(s), duplicating {} vertices -> {} boundary loop",
                report.paths_cut, report.vertices_added, report.loops_after
            );
            mesh = opened;
        }
    }
    let mesh = mesh;

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

    if let Some(layout) = layout {
        let flat = parameterize::layout_mesh(&mesh, &uvs)?;
        io::save(&flat, layout)?;
        println!("Wrote UV layout: {}", layout.display());
    }

    Ok(())
}
