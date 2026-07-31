//! Distance from a point to a surface, and between two surfaces.
//!
//! This is the measurement that was missing. Triangle quality says whether a mesh is
//! *well shaped*; nothing said whether it is still the *same shape*. A remesher can score
//! perfectly on quality by discarding the geometry, and all three of ours quietly shrank
//! the surface because none projected its smoothed vertices back onto the input — a defect
//! that could only be described, not measured, until there was a distance to measure it
//! with.
//!
//! Two things live here, sharing one nearest-triangle query:
//!
//! - [`SurfaceIndex`], which answers "where is the closest point of this surface, and how
//!   far is it" in `O(log n)` rather than by scanning every triangle.
//! - [`hausdorff_distance`], the largest distance from either surface to the other, which
//!   is the standard way to say "these two meshes are the same shape to within ε".
//!
//! # On sampling
//!
//! Hausdorff distance is defined over the *whole* surface, but computing it exactly for
//! two triangle meshes is a hard geometric problem. Like the Metro tool that established
//! the convention, this samples one surface and measures each sample exactly against the
//! other. Sampling only the vertices is fast and badly biased — a vertex of a fine mesh
//! usually sits close to a coarse one even where the surfaces diverge between vertices —
//! so [`HausdorffOptions::samples_per_face`] adds interior samples, and the default is not
//! zero. Every result is therefore a *lower bound*, and the report says how many samples
//! produced it.

use nalgebra::Point3;

use crate::mesh::{HalfEdgeMesh, MeshIndex};

/// An axis-aligned box.
#[derive(Debug, Clone, Copy)]
struct Aabb {
    min: Point3<f64>,
    max: Point3<f64>,
}

impl Aabb {
    fn empty() -> Self {
        Self {
            min: Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
            max: Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
        }
    }

    fn add(&mut self, p: &Point3<f64>) {
        for k in 0..3 {
            if p[k] < self.min[k] {
                self.min[k] = p[k];
            }
            if p[k] > self.max[k] {
                self.max[k] = p[k];
            }
        }
    }

    /// Squared distance from `p` to the box; zero when inside.
    fn distance_squared(&self, p: &Point3<f64>) -> f64 {
        let mut total = 0.0;
        for k in 0..3 {
            let d = if p[k] < self.min[k] {
                self.min[k] - p[k]
            } else if p[k] > self.max[k] {
                p[k] - self.max[k]
            } else {
                0.0
            };
            total += d * d;
        }
        total
    }

    fn longest_axis(&self) -> usize {
        let extent = self.max - self.min;
        if extent.x >= extent.y && extent.x >= extent.z {
            0
        } else if extent.y >= extent.z {
            1
        } else {
            2
        }
    }
}

/// The closest point of triangle `abc` to `p`.
///
/// The seven-case form from Ericson's *Real-Time Collision Detection*: the plane of the
/// triangle is divided into the interior, three edge regions and three vertex regions, and
/// the answer is a projection onto whichever one `p` falls in. Solving it this way rather
/// than by projecting onto the plane and clamping barycentric coordinates is what makes it
/// correct for obtuse triangles, where the perpendicular foot can lie outside the triangle
/// on the far side of an edge it is not nearest to.
///
/// Degenerate triangles are handled by the same arithmetic: a zero-area triangle reduces to
/// its longest edge, and a single point to itself.
pub fn closest_point_on_triangle(
    p: &Point3<f64>,
    a: &Point3<f64>,
    b: &Point3<f64>,
    c: &Point3<f64>,
) -> Point3<f64> {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;

    // Vertex region A.
    let d1 = ab.dot(&ap);
    let d2 = ac.dot(&ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return *a;
    }

    // Vertex region B.
    let bp = p - b;
    let d3 = ab.dot(&bp);
    let d4 = ac.dot(&bp);
    if d3 >= 0.0 && d4 <= d3 {
        return *b;
    }

    // Edge region AB.
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let denom = d1 - d3;
        let v = if denom != 0.0 { d1 / denom } else { 0.0 };
        return *a + ab * v;
    }

    // Vertex region C.
    let cp = p - c;
    let d5 = ab.dot(&cp);
    let d6 = ac.dot(&cp);
    if d6 >= 0.0 && d5 <= d6 {
        return *c;
    }

    // Edge region AC.
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let denom = d2 - d6;
        let w = if denom != 0.0 { d2 / denom } else { 0.0 };
        return *a + ac * w;
    }

    // Edge region BC.
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let denom = (d4 - d3) + (d5 - d6);
        let w = if denom != 0.0 { (d4 - d3) / denom } else { 0.0 };
        return *b + (c - b) * w;
    }

    // Interior.
    let denom = va + vb + vc;
    if denom == 0.0 {
        // Fully degenerate: every region test failed, so fall back to the nearest corner.
        let da = (p - a).norm_squared();
        let db = (p - b).norm_squared();
        let dc = (p - c).norm_squared();
        return if da <= db && da <= dc {
            *a
        } else if db <= dc {
            *b
        } else {
            *c
        };
    }
    let v = vb / denom;
    let w = vc / denom;
    *a + ab * v + ac * w
}

/// Distance from `p` to triangle `abc`.
pub fn point_triangle_distance(
    p: &Point3<f64>,
    a: &Point3<f64>,
    b: &Point3<f64>,
    c: &Point3<f64>,
) -> f64 {
    (closest_point_on_triangle(p, a, b, c) - p).norm()
}

/// A node of the bounding-volume hierarchy. Leaves own a contiguous run of triangles.
#[derive(Debug, Clone, Copy)]
struct Node {
    bounds: Aabb,
    /// Index of the first triangle for a leaf; index of the left child otherwise.
    first: usize,
    /// Triangle count for a leaf; zero for an interior node.
    count: usize,
    /// Index of the right child; unused for a leaf.
    right: usize,
}

/// A surface prepared for nearest-point queries.
///
/// A bounding-volume hierarchy rather than a uniform grid, because triangle sizes on a real
/// mesh are not uniform: the Stanford bunny's edges span a factor of 27, and a grid sized
/// for the small ones wastes memory while a grid sized for the large ones puts hundreds of
/// triangles in a cell.
///
/// Construction is deterministic — the median split sorts by centroid with a tiebreak on
/// triangle index — so queries against an index built twice from the same mesh agree
/// exactly.
#[derive(Debug, Clone)]
pub struct SurfaceIndex {
    triangles: Vec<[Point3<f64>; 3]>,
    /// Triangle indices, permuted so each leaf owns a contiguous run.
    order: Vec<usize>,
    nodes: Vec<Node>,
}

impl SurfaceIndex {
    /// Largest number of triangles in a leaf. Small enough that the linear scan in a leaf
    /// is cheap, large enough that the tree stays shallow.
    const LEAF_SIZE: usize = 8;

    /// Build an index over a mesh's faces.
    pub fn new<I: MeshIndex>(mesh: &HalfEdgeMesh<I>) -> Self {
        let triangles: Vec<[Point3<f64>; 3]> = mesh
            .face_ids()
            .map(|f| {
                let [a, b, c] = mesh.face_triangle(f);
                [*mesh.position(a), *mesh.position(b), *mesh.position(c)]
            })
            .collect();
        Self::from_triangles(triangles)
    }

    /// Build an index over explicit triangles.
    pub fn from_triangles(triangles: Vec<[Point3<f64>; 3]>) -> Self {
        let mut order: Vec<usize> = (0..triangles.len()).collect();
        let mut nodes = Vec::new();
        if !triangles.is_empty() {
            let count = order.len();
            build(&triangles, &mut order, &mut nodes, 0, count);
        }
        Self {
            triangles,
            order,
            nodes,
        }
    }

    /// Whether the index holds no triangles, in which case every query returns `None`.
    pub fn is_empty(&self) -> bool {
        self.triangles.is_empty()
    }

    /// Triangles indexed.
    pub fn len(&self) -> usize {
        self.triangles.len()
    }

    /// The closest point of the surface to `p`, and its distance.
    ///
    /// `None` only for an empty surface.
    pub fn closest_point(&self, p: &Point3<f64>) -> Option<(Point3<f64>, f64)> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut best = Point3::origin();
        let mut best_sq = f64::INFINITY;
        self.search(0, p, &mut best, &mut best_sq);
        Some((best, best_sq.sqrt()))
    }

    /// Distance from `p` to the surface. `None` only for an empty surface.
    pub fn distance(&self, p: &Point3<f64>) -> Option<f64> {
        self.closest_point(p).map(|(_, d)| d)
    }

    /// Project `p` onto the surface, or return it unchanged if the surface is empty.
    pub fn project(&self, p: &Point3<f64>) -> Point3<f64> {
        self.closest_point(p).map(|(q, _)| q).unwrap_or(*p)
    }

    fn search(&self, node: usize, p: &Point3<f64>, best: &mut Point3<f64>, best_sq: &mut f64) {
        let n = &self.nodes[node];
        // Prune: nothing inside this box can beat what we already have.
        if n.bounds.distance_squared(p) >= *best_sq {
            return;
        }
        if n.count > 0 {
            for &ti in &self.order[n.first..n.first + n.count] {
                let t = &self.triangles[ti];
                let q = closest_point_on_triangle(p, &t[0], &t[1], &t[2]);
                let d = (q - p).norm_squared();
                if d < *best_sq {
                    *best_sq = d;
                    *best = q;
                }
            }
            return;
        }
        // Descend into the nearer child first, so the far one is more likely to be pruned.
        let (l, r) = (n.first, n.right);
        let dl = self.nodes[l].bounds.distance_squared(p);
        let dr = self.nodes[r].bounds.distance_squared(p);
        let (first, second) = if dl <= dr { (l, r) } else { (r, l) };
        self.search(first, p, best, best_sq);
        self.search(second, p, best, best_sq);
    }
}

/// Build a subtree over `order[start..end]`, returning its node index.
fn build(
    triangles: &[[Point3<f64>; 3]],
    order: &mut Vec<usize>,
    nodes: &mut Vec<Node>,
    start: usize,
    end: usize,
) -> usize {
    let mut bounds = Aabb::empty();
    for &ti in &order[start..end] {
        for p in &triangles[ti] {
            bounds.add(p);
        }
    }

    let node = nodes.len();
    let count = end - start;
    if count <= SurfaceIndex::LEAF_SIZE {
        nodes.push(Node {
            bounds,
            first: start,
            count,
            right: 0,
        });
        return node;
    }

    // Reserve this node's slot before recursing, so children get later indices.
    nodes.push(Node {
        bounds,
        first: 0,
        count: 0,
        right: 0,
    });

    let axis = bounds.longest_axis();
    let centroid = |ti: usize| {
        let t = &triangles[ti];
        (t[0][axis] + t[1][axis] + t[2][axis]) / 3.0
    };
    // Sorting by centroid with the triangle index as a tiebreak keeps the build
    // deterministic even when many triangles share a centroid coordinate.
    order[start..end].sort_by(|&x, &y| centroid(x).total_cmp(&centroid(y)).then_with(|| x.cmp(&y)));

    let mid = start + count / 2;
    let left = build(triangles, order, nodes, start, mid);
    let right = build(triangles, order, nodes, mid, end);
    nodes[node].first = left;
    nodes[node].right = right;
    node
}

/// How to sample when measuring Hausdorff distance.
#[derive(Debug, Clone)]
pub struct HausdorffOptions {
    /// Interior sample points per triangle, in addition to the vertices.
    ///
    /// Zero measures vertices only, which is fast and understates the distance: a vertex of
    /// a fine mesh generally lies close to a coarser one even where the surfaces part
    /// company between vertices. The default is deliberately not zero.
    pub samples_per_face: usize,
    /// Whether to measure both directions and take the larger.
    ///
    /// One direction alone is not a distance and can be badly misleading: every vertex of a
    /// coarse mesh can sit exactly on a fine one while the fine one has detail the coarse
    /// one omits entirely.
    pub symmetric: bool,
}

impl Default for HausdorffOptions {
    fn default() -> Self {
        Self {
            samples_per_face: 10,
            symmetric: true,
        }
    }
}

impl HausdorffOptions {
    /// Vertices only: fastest, and a weaker lower bound.
    pub fn vertices_only() -> Self {
        Self {
            samples_per_face: 0,
            symmetric: true,
        }
    }

    /// Denser sampling, for a tighter lower bound.
    pub fn with_samples_per_face(mut self, samples: usize) -> Self {
        self.samples_per_face = samples;
        self
    }
}

/// What a Hausdorff measurement found.
///
/// Every figure is a lower bound on the true distance, since it comes from samples. Compare
/// [`Self::relative`] rather than the absolute distance when meshes differ in scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HausdorffReport {
    /// Largest distance from a sample on the first mesh to the second.
    pub forward: f64,
    /// Largest distance from a sample on the second mesh to the first. Zero when
    /// [`HausdorffOptions::symmetric`] is false.
    pub backward: f64,
    /// The larger of the two, which is the Hausdorff distance proper.
    pub distance: f64,
    /// Root-mean-square of the forward sample distances. Far more stable than the maximum,
    /// which one bad sample sets.
    pub forward_rms: f64,
    /// Root-mean-square of the backward sample distances.
    pub backward_rms: f64,
    /// Samples measured, both directions together.
    pub samples: usize,
    /// Diagonal of the first mesh's bounding box, the scale [`Self::relative`] divides by.
    pub bounding_diagonal: f64,
}

impl HausdorffReport {
    /// [`Self::distance`] as a fraction of the bounding-box diagonal, which is how mesh
    /// comparisons are conventionally quoted.
    pub fn relative(&self) -> f64 {
        if self.bounding_diagonal > 0.0 {
            self.distance / self.bounding_diagonal
        } else {
            0.0
        }
    }
}

/// Sample points on a mesh: every vertex, plus `samples_per_face` interior points per face.
///
/// Interior points come from a triangular lattice in barycentric coordinates, which spreads
/// them evenly over the face and — unlike random sampling — makes the result reproducible.
fn sample_surface<I: MeshIndex>(
    mesh: &HalfEdgeMesh<I>,
    samples_per_face: usize,
) -> Vec<Point3<f64>> {
    let mut points: Vec<Point3<f64>> = mesh.vertex_ids().map(|v| *mesh.position(v)).collect();

    if samples_per_face == 0 {
        return points;
    }

    // A lattice with `n` points per side holds n(n+1)/2 points; pick the smallest `n` that
    // reaches the requested count.
    let mut n = 1;
    while n * (n + 1) / 2 < samples_per_face {
        n += 1;
    }

    for face in mesh.face_ids() {
        let [ia, ib, ic] = mesh.face_triangle(face);
        let (a, b, c) = (mesh.position(ia), mesh.position(ib), mesh.position(ic));
        let mut emitted = 0;
        for i in 0..n {
            for j in 0..(n - i) {
                if emitted == samples_per_face {
                    break;
                }
                // Offset by 1/3 so samples land inside the triangle rather than on its
                // edges and corners, which the vertex samples already cover.
                let u = (i as f64 + 1.0 / 3.0) / (n as f64 + 1.0 / 3.0);
                let v = (j as f64 + 1.0 / 3.0) / (n as f64 + 1.0 / 3.0);
                let w = 1.0 - u - v;
                if w <= 0.0 {
                    continue;
                }
                points.push(Point3::from(a.coords * w + b.coords * u + c.coords * v));
                emitted += 1;
            }
        }
    }

    points
}

/// Bounding-box diagonal of a mesh.
fn bounding_diagonal<I: MeshIndex>(mesh: &HalfEdgeMesh<I>) -> f64 {
    let mut bounds = Aabb::empty();
    for v in mesh.vertex_ids() {
        bounds.add(mesh.position(v));
    }
    if bounds.min.x > bounds.max.x {
        return 0.0;
    }
    (bounds.max - bounds.min).norm()
}

/// Measure one direction: the largest and RMS distance from samples on `from` to `to`.
fn one_sided(samples: &[Point3<f64>], to: &SurfaceIndex) -> (f64, f64) {
    if to.is_empty() || samples.is_empty() {
        return (0.0, 0.0);
    }
    let mut worst = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    for p in samples {
        let d = to.distance(p).unwrap_or(0.0);
        if d > worst {
            worst = d;
        }
        sum_sq += d * d;
    }
    (worst, (sum_sq / samples.len() as f64).sqrt())
}

/// Hausdorff distance between two meshes.
///
/// See the module docs on sampling: the result is a lower bound, tightened by
/// [`HausdorffOptions::samples_per_face`].
///
/// # Example
///
/// ```no_run
/// use morsel::algo::distance::{hausdorff_distance, HausdorffOptions};
/// use morsel::mesh::HalfEdgeMesh;
///
/// let original: HalfEdgeMesh = morsel::io::load("input.obj").unwrap();
/// let simplified: HalfEdgeMesh = morsel::io::load("output.obj").unwrap();
///
/// let report = hausdorff_distance(&original, &simplified, &HausdorffOptions::default());
/// println!("{:.3}% of the bounding diagonal", 100.0 * report.relative());
/// ```
pub fn hausdorff_distance<I: MeshIndex, J: MeshIndex>(
    a: &HalfEdgeMesh<I>,
    b: &HalfEdgeMesh<J>,
    options: &HausdorffOptions,
) -> HausdorffReport {
    let index_b = SurfaceIndex::new(b);
    let samples_a = sample_surface(a, options.samples_per_face);
    let (forward, forward_rms) = one_sided(&samples_a, &index_b);

    let (backward, backward_rms, backward_samples) = if options.symmetric {
        let index_a = SurfaceIndex::new(a);
        let samples_b = sample_surface(b, options.samples_per_face);
        let n = samples_b.len();
        let (worst, rms) = one_sided(&samples_b, &index_a);
        (worst, rms, n)
    } else {
        (0.0, 0.0, 0)
    };

    HausdorffReport {
        forward,
        backward,
        distance: forward.max(backward),
        forward_rms,
        backward_rms,
        samples: samples_a.len() + backward_samples,
        bounding_diagonal: bounding_diagonal(a),
    }
}
