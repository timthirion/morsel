# morsel
Mesh processing in Rust

## Tools

### morsel-view

A 3D mesh viewer for inspecting meshes.

**Install:**
```bash
cargo install --path . --features viewer
```

**Usage:**
```bash
morsel-view path/to/mesh.obj
```

Supported formats: `.obj`, `.stl`, `.ply`, `.gltf`, `.glb`

**Controls:**

| Input | Action |
|-------|--------|
| Left mouse drag | Rotate camera |
| Scroll wheel | Zoom in/out |
| `W` | Toggle wireframe |
| `R` | Reset camera |
| `Escape` | Quit |
