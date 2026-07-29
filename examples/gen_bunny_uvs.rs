//! Provenance for the UV coordinates on `examples/stanford-bunny.obj`.
//!
//! The committed bunny carries `vt` entries and an accompanying
//! `stanford-bunny.mtl`; this is the code that produced them, kept so the asset is
//! reproducible rather than mysterious.
//!
//! ```text
//! cargo run --example gen_bunny_uvs             # verify the committed asset matches
//! cargo run --example gen_bunny_uvs -- --write  # regenerate it
//! ```
//!
//! Verification is the default deliberately. This writes *over* its own input, so a
//! bare run used to mutate a committed data file purely as a side effect of being
//! invoked. It happens to be idempotent today, but that is a property worth
//! checking rather than assuming — if `cylindrical_projection` ever changes, the
//! default run reports the drift instead of silently baking it in.
//!
//! Paths resolve against `CARGO_MANIFEST_DIR`, so it works from any directory.

use std::path::PathBuf;

use morsel::algo::parameterize::{cylindrical_projection, UVMap};
use morsel::io::{self, obj};
use morsel::mesh::HalfEdgeMesh;

/// Largest per-coordinate difference between two UV maps, or `None` if their
/// lengths disagree.
fn max_uv_drift(a: &UVMap, b: &UVMap) -> Option<f64> {
    if a.len() != b.len() {
        return None;
    }
    let mut worst = 0.0_f64;
    for ((_, p), (_, q)) in a.iter().zip(b.iter()) {
        worst = worst.max((p.x - q.x).abs()).max((p.y - q.y).abs());
    }
    Some(worst)
}

fn main() {
    let write = std::env::args().any(|a| a == "--write");

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let obj_path = root.join("examples/stanford-bunny.obj");
    let mtl_path = root.join("examples/stanford-bunny.mtl");

    let mesh: HalfEdgeMesh = io::load(&obj_path)
        .unwrap_or_else(|e| panic!("failed to load {}: {e}", obj_path.display()));
    println!(
        "Loaded {}: {} vertices, {} faces",
        obj_path.display(),
        mesh.num_vertices(),
        mesh.num_faces()
    );

    // Cylindrical projection, not LSCM or ARAP: the bunny has four boundary loops
    // around its base, so it is not a disk and those methods correctly refuse it.
    // A cut generator would be needed before a conformal map is even well posed.
    println!("Computing UVs by cylindrical projection around the Y axis...");
    let uvs = cylindrical_projection(&mesh);
    if let Some((min, max)) = uvs.bounding_box() {
        println!(
            "UV bounds: ({:.3}, {:.3}) to ({:.3}, {:.3})",
            min.x, min.y, max.x, max.y
        );
    }

    if !write {
        let (_, existing): (HalfEdgeMesh, Option<UVMap>) = obj::load_with_uvs(&obj_path)
            .unwrap_or_else(|e| panic!("failed to re-read {}: {e}", obj_path.display()));

        let Some(existing) = existing else {
            eprintln!(
                "MISMATCH: {} has no UV coordinates. Run with --write to add them.",
                obj_path.display()
            );
            std::process::exit(1);
        };

        let Some(drift) = max_uv_drift(&uvs, &existing) else {
            eprintln!(
                "MISMATCH: {} carries {} UVs but the mesh has {} vertices.",
                obj_path.display(),
                existing.len(),
                uvs.len()
            );
            std::process::exit(1);
        };

        // The file stores decimal text, so exact equality is not the bar; agreement
        // at printing precision means the asset is what this code produces.
        if drift >= 1e-6 {
            eprintln!(
                "MISMATCH: committed UVs differ by up to {drift:.3e}. Either this code \
                 changed or the asset was edited by hand. Run with --write to regenerate."
            );
            std::process::exit(1);
        }

        println!("OK: committed UVs match, largest difference {drift:.2e}");
        println!("Nothing written. Pass --write to regenerate the asset.");
        return;
    }

    obj::save_with_uvs(&mesh, &uvs, &obj_path, Some("stanford-bunny.mtl"))
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", obj_path.display()));
    println!("Wrote {}", obj_path.display());

    // The texture path is relative to the OBJ, so it climbs out of examples/.
    obj::write_mtl(&mtl_path, "../images/UV.png")
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", mtl_path.display()));
    println!("Wrote {}", mtl_path.display());
}
