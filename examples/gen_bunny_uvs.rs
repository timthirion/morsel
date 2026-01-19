//! Generate UV coordinates for the Stanford bunny mesh.
//!
//! Run with: cargo run --example gen_bunny_uvs

use morsel::algo::parameterize::cylindrical_projection;
use morsel::io::{self, obj};
use morsel::mesh::HalfEdgeMesh;

fn main() {
    // Load the bunny
    let mesh: HalfEdgeMesh =
        io::load("examples/stanford-bunny.obj").expect("Failed to load bunny");
    println!(
        "Loaded mesh: {} vertices, {} faces",
        mesh.num_vertices(),
        mesh.num_faces()
    );

    // Compute UV parameterization using cylindrical projection
    println!("Computing UV parameterization (cylindrical projection)...");
    let uvs = cylindrical_projection(&mesh);

    // Check bounding box
    if let Some((min, max)) = uvs.bounding_box() {
        println!(
            "UV bounds: ({:.3}, {:.3}) to ({:.3}, {:.3})",
            min.x, min.y, max.x, max.y
        );
    }

    println!("UV parameterization complete");

    // Save the mesh with UVs
    obj::save_with_uvs(
        &mesh,
        &uvs,
        "examples/stanford-bunny.obj",
        Some("stanford-bunny.mtl"),
    )
    .expect("Failed to save OBJ");
    println!("Saved examples/stanford-bunny.obj");

    // Create the MTL file (texture path relative to OBJ location)
    obj::write_mtl("examples/stanford-bunny.mtl", "../images/UV.png").expect("Failed to save MTL");
    println!("Saved examples/stanford-bunny.mtl");

    println!("Done!");
}
