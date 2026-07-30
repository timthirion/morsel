//! Orbit camera controller for the mesh viewer.

use std::f32::consts::PI;

/// Orbit camera that rotates around a target point.
pub struct OrbitCamera {
    /// Target point to orbit around.
    pub target: [f32; 3],
    /// Distance from target.
    pub distance: f32,
    /// Horizontal angle (radians).
    pub azimuth: f32,
    /// Vertical angle (radians), clamped to avoid gimbal lock.
    pub elevation: f32,
    /// Field of view in radians.
    pub fov: f32,
    /// Near clip plane.
    pub near: f32,
    /// Far clip plane.
    pub far: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: [0.0, 0.0, 0.0],
            distance: 3.0,
            azimuth: 0.0,
            elevation: 0.3,
            fov: PI / 4.0, // 45 degrees
            near: 0.01,
            far: 100.0,
        }
    }
}

impl OrbitCamera {
    /// Create a new orbit camera looking at the given target from a distance.
    #[allow(dead_code)]
    pub fn new(target: [f32; 3], distance: f32) -> Self {
        Self {
            target,
            distance,
            ..Default::default()
        }
    }

    /// Get the camera's eye position in world space.
    pub fn eye_position(&self) -> [f32; 3] {
        let cos_elev = self.elevation.cos();
        let sin_elev = self.elevation.sin();
        let cos_azim = self.azimuth.cos();
        let sin_azim = self.azimuth.sin();

        [
            self.target[0] + self.distance * cos_elev * sin_azim,
            self.target[1] + self.distance * sin_elev,
            self.target[2] + self.distance * cos_elev * cos_azim,
        ]
    }

    /// Get the view matrix (world to camera transform).
    pub fn view_matrix(&self) -> [[f32; 4]; 4] {
        let eye = self.eye_position();
        look_at(eye, self.target, [0.0, 1.0, 0.0])
    }

    /// Get the projection matrix.
    pub fn projection_matrix(&self, aspect: f32) -> [[f32; 4]; 4] {
        perspective(self.fov, aspect, self.near, self.far)
    }

    /// Get combined view-projection matrix.
    pub fn view_projection_matrix(&self, aspect: f32) -> [[f32; 4]; 4] {
        let view = self.view_matrix();
        let proj = self.projection_matrix(aspect);
        mat4_mul(proj, view)
    }

    /// Rotate the camera by the given deltas (in radians).
    pub fn rotate(&mut self, delta_azimuth: f32, delta_elevation: f32) {
        self.azimuth += delta_azimuth;
        self.elevation += delta_elevation;

        // Clamp elevation to avoid flipping
        let limit = PI / 2.0 - 0.01;
        self.elevation = self.elevation.clamp(-limit, limit);
    }

    /// Zoom the camera by the given factor.
    pub fn zoom(&mut self, factor: f32) {
        self.distance *= factor;
        self.distance = self.distance.clamp(0.1, 100.0);
    }

    /// Reset the camera to default position.
    pub fn reset(&mut self, target: [f32; 3], distance: f32) {
        self.target = target;
        self.distance = distance;
        self.azimuth = 0.0;
        self.elevation = 0.3;
    }
}

/// Create a look-at view matrix.
fn look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
    let f = normalize(sub(target, eye));
    let s = normalize(cross(f, up));
    let u = cross(s, f);

    [
        [s[0], u[0], -f[0], 0.0],
        [s[1], u[1], -f[1], 0.0],
        [s[2], u[2], -f[2], 0.0],
        [-dot(s, eye), -dot(u, eye), dot(f, eye), 1.0],
    ]
}

/// Create a perspective projection matrix.
fn perspective(fov: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let f = 1.0 / (fov / 2.0).tan();
    let nf = 1.0 / (near - far);

    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (far + near) * nf, -1.0],
        [0.0, 0.0, 2.0 * far * near * nf, 0.0],
    ]
}

/// Multiply two 4x4 matrices.
fn mat4_mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut result = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                result[i][j] += a[k][j] * b[i][k];
            }
        }
    }
    result
}

// Vector operations
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1e-10 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, 1.0]
    }
}
