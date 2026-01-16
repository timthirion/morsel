//! Benchmarks for mesh operations.

use criterion::{criterion_group, criterion_main, Criterion};
use morsel::prelude::*;
use nalgebra::Point3;

fn create_grid_mesh(n: usize) -> HalfEdgeMesh {
    let mut vertices = Vec::with_capacity((n + 1) * (n + 1));
    let mut faces = Vec::with_capacity(n * n * 2);

    // Create grid vertices
    for j in 0..=n {
        for i in 0..=n {
            vertices.push(Point3::new(i as f64, j as f64, 0.0));
        }
    }

    // Create triangles
    for j in 0..n {
        for i in 0..n {
            let v00 = j * (n + 1) + i;
            let v10 = v00 + 1;
            let v01 = v00 + (n + 1);
            let v11 = v01 + 1;

            faces.push([v00, v10, v11]);
            faces.push([v00, v11, v01]);
        }
    }

    build_from_triangles(&vertices, &faces).unwrap()
}

fn bench_mesh_construction(c: &mut Criterion) {
    c.bench_function("build_grid_10x10", |b| {
        let n = 10;
        let mut vertices = Vec::with_capacity((n + 1) * (n + 1));
        let mut faces = Vec::with_capacity(n * n * 2);

        for j in 0..=n {
            for i in 0..=n {
                vertices.push(Point3::new(i as f64, j as f64, 0.0));
            }
        }

        for j in 0..n {
            for i in 0..n {
                let v00 = j * (n + 1) + i;
                let v10 = v00 + 1;
                let v01 = v00 + (n + 1);
                let v11 = v01 + 1;

                faces.push([v00, v10, v11]);
                faces.push([v00, v11, v01]);
            }
        }

        b.iter(|| {
            let mesh: HalfEdgeMesh = build_from_triangles(&vertices, &faces).unwrap();
            mesh
        });
    });
}

fn bench_mesh_traversal(c: &mut Criterion) {
    let mesh = create_grid_mesh(50);

    c.bench_function("vertex_neighbors_all", |b| {
        b.iter(|| {
            let mut count = 0;
            for v in mesh.vertex_ids() {
                count += mesh.vertex_neighbors(v).count();
            }
            count
        });
    });

    c.bench_function("face_normals_all", |b| {
        b.iter(|| {
            let mut sum = nalgebra::Vector3::zeros();
            for f in mesh.face_ids() {
                sum += mesh.face_normal(f);
            }
            sum
        });
    });
}

criterion_group!(benches, bench_mesh_construction, bench_mesh_traversal);
criterion_main!(benches);
