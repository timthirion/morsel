//! GPU mesh buffer management for the viewer.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use morsel::mesh::{HalfEdgeMesh, MeshIndex};

/// GPU vertex with position and normal.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

impl Vertex {
    /// Vertex buffer layout for wgpu.
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // position
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // normal
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

/// Mesh data uploaded to the GPU.
pub struct GpuMesh {
    /// Vertex buffer containing positions and normals.
    pub vertex_buffer: wgpu::Buffer,
    /// Index buffer for triangle rendering (used for both solid and wireframe).
    pub index_buffer: wgpu::Buffer,
    /// Number of indices.
    pub num_indices: u32,
    /// Mesh center (for camera targeting).
    pub center: [f32; 3],
    /// Mesh bounding radius (for camera distance).
    pub radius: f32,
}

impl GpuMesh {
    /// Create GPU buffers from a half-edge mesh.
    ///
    /// Uses flat shading (per-face normals) for correct rendering of hard edges.
    pub fn from_halfedge_mesh<I: MeshIndex>(device: &wgpu::Device, mesh: &HalfEdgeMesh<I>) -> Self {
        // Use per-face vertices with face normals for flat shading.
        // This duplicates vertices but gives correct shading for hard edges.
        let mut vertices = Vec::with_capacity(mesh.num_faces() * 3);
        let mut indices: Vec<u32> = Vec::with_capacity(mesh.num_faces() * 3);

        for fid in mesh.face_ids() {
            let [v0, v1, v2] = mesh.face_triangle(fid);
            let p0 = mesh.position(v0);
            let p1 = mesh.position(v1);
            let p2 = mesh.position(v2);
            let normal = mesh.face_normal(fid);

            let base_idx = vertices.len() as u32;

            // Add 3 vertices with the face normal
            vertices.push(Vertex {
                position: [p0.x as f32, p0.y as f32, p0.z as f32],
                normal: [normal.x as f32, normal.y as f32, normal.z as f32],
            });
            vertices.push(Vertex {
                position: [p1.x as f32, p1.y as f32, p1.z as f32],
                normal: [normal.x as f32, normal.y as f32, normal.z as f32],
            });
            vertices.push(Vertex {
                position: [p2.x as f32, p2.y as f32, p2.z as f32],
                normal: [normal.x as f32, normal.y as f32, normal.z as f32],
            });

            // Triangle indices
            indices.push(base_idx);
            indices.push(base_idx + 1);
            indices.push(base_idx + 2);
        }

        // Compute centroid (center of mass) as the average of all vertex positions
        let (center, radius) = if mesh.num_vertices() > 0 {
            let mut sum = [0.0_f64; 3];
            for vid in mesh.vertex_ids() {
                let p = mesh.position(vid);
                sum[0] += p.x;
                sum[1] += p.y;
                sum[2] += p.z;
            }
            let n = mesh.num_vertices() as f64;
            let center = [
                (sum[0] / n) as f32,
                (sum[1] / n) as f32,
                (sum[2] / n) as f32,
            ];

            // Compute radius as max distance from centroid to any vertex
            let mut max_dist_sq = 0.0_f32;
            for vid in mesh.vertex_ids() {
                let p = mesh.position(vid);
                let dx = p.x as f32 - center[0];
                let dy = p.y as f32 - center[1];
                let dz = p.z as f32 - center[2];
                let dist_sq = dx * dx + dy * dy + dz * dz;
                if dist_sq > max_dist_sq {
                    max_dist_sq = dist_sq;
                }
            }
            (center, max_dist_sq.sqrt())
        } else {
            ([0.0, 0.0, 0.0], 1.0)
        };

        // Create GPU buffers
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mesh Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mesh Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            vertex_buffer,
            index_buffer,
            num_indices: indices.len() as u32,
            center,
            radius,
        }
    }
}
