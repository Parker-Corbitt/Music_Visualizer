# Music Visualizer

Interactive Metal app for exploring music-derived point clouds as points, isosurfaces, or ray-marched volumes. Two datasets are supported:

- **Song mode:** a single track's point cloud, generated from its time–frequency features.
- **Timeline mode:** a stitched ribbon that places all songs in chronological order along `x`, letting you scrub history as one continuous cloud.

## Rendering options
- **Style:** Points, Mesh (marching tetrahedra or dual contouring), or Volume ray marching.
- **Isovalue:** Adjust mesh extraction threshold.
- **Lighting:** Toggle Phong shading on meshes.
- **Volume transfer functions:** Classic, Ember (warm/high contrast), or Glacial (cool/sharp falloff).

## Data pipeline
Run `audio_prep.py` to build inputs under `data/processed/`:
- Per-song `<stem>_points.bin` files for Song mode.
- `timeline_points.bin` for the stitched ribbon.
Per-song files are produced automatically during extraction; the timeline cloud can also be rebuilt alone via `python audio_prep.py --timeline-only`.

## Controls
- Mode: Song ↔ Timeline.
- Render style: Points / Mesh / Volume.
- Mesh algorithm: Tetra / Dual (mesh only).
- Lighting: enable/disable Phong (mesh only).
- Isovalue slider (mesh only).
- Transfer function preset (volume only).
- Load point cloud: opens a `.bin` for Song mode.

## Notes
- Voxel grids are fit to the active cloud's bounds, so large timeline ribbons are fully voxelized for volume and isosurface rendering.
- Volume rendering uses full-screen ray marching; transfer presets adjust color/opacity curves without rebuilding the grid.
