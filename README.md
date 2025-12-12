# Music Visualizer

## Abstract

This project visualizes the relationship between loudness and amplitude in popular music using a dual-mode system that operates at both micro and macro scales. Audio features from representative tracks are converted into dense 3D point clouds, which are then voxelized on the GPU into scalar fields and visualized using two high-level computer graphics methods: isosurface extraction and GPU-based ray-marched volume rendering. In this formulation, each volume is treated as an audio density field, and ray marching integrates that density along viewing rays to reveal where particular loudness behaviors are persistent or dominant. Isosurfaces extracted from these point-cloud-derived volumes are color-mapped using additional audio attributes (such as harmonic–percussive balance), encoding a fourth data dimension into hue. The system supports an individual Song Mode that reveals fine-scale loudness structure within a single track, and an aggregate Timeline Mode that lays many songs out along a chronological axis to show how amplitude, loudness, and dynamics behavior evolve over time. This provides a comprehensive visualization framework for analyzing and exploring audio dynamics through advanced rendering techniques.

## Motivation

Loudness and dynamic range in popular music have changed significantly over time, especially over the last half century. However, these changes are rarely visualized in a way that reveals structure, continuity, and evolution. This project tackles that gap by converting audio-derived features into volumetric data and visualizing them using two complementary rendering techniques. By treating these volumes as audio density fields, volume ray marching turns each pixel into the integral of loudness-related energy along a viewing ray. In Timeline Mode, this reveals which loudness–amplitude–dynamics regimes are thick and persistent across long stretches of time; in Song Mode, it highlights coherent 3D structures of loudness across time and frequency within a single track, rather than isolated peaks. The Timeline Mode highlights how the loudness–amplitude relationship and dynamics behavior morph across songs and eras, while the Song Mode provides detailed insight into individual tracks. Together, these modes create a powerful exploration tool that is meaningful to both audio and graphics communities, while demonstrating high-level concepts in GPU rendering and surface extraction.

## Methodology

This project treats loudness–amplitude behavior as a scalar field to be visualized through both isosurfaces and ray-marched volumes. Two distinct data representations feed a single unified rendering pipeline:

### Timeline Mode (Aggregate Timeline Point Cloud)

Songs are processed into per-song point clouds and then stitched into a single chronological timeline point cloud. The primary horizontal axis is a continuous timeline that marches through songs in order, while the other axes encode amplitude and dynamics:

- **x-axis:** global timeline position (chronological ordering of songs / frames)
- **y-axis:** physical amplitude or energy level (mapped from frame-wise amplitude bins)
- **z-axis:** crest factor (peak-to-RMS behavior), normalized to [-1, 1] based on the 1st and 99th percentiles
- **Scalar value:** loudness-related density derived from aggregated, A-weighted features
- **Hue:** harmonic–percussive balance aggregated into the timeline

In this formulation, the timeline volume acts as a “loudness and dynamics history” density field: regions of high scalar value indicate amplitude–loudness–crest-factor regimes where the music spends a lot of time along the chronological axis. Isosurfaces extracted from this voxelized timeline trace out level sets of loudness density, revealing how the most common loudness and dynamics behaviors drift, thicken, and thin out across the timeline.

### Song Mode (Per-Song Point Clouds and Volumes)

Each track is processed independently into a dense 3D point cloud derived from its STFT. At preprocessing time, each time–frequency bin becomes a vertex with the following layout:

- **x-axis:** normalized time within the song (after dropping fully silent frames), mapped to [-1, 1]
- **y-axis:** log amplitude (dB-like) normalized over the song and mapped to [-1, 1]
- **z-axis:** normalized frequency index (low to high) mapped to [-1, 1]
- **Scalar value:** A-weighted loudness at that (time, frequency) bin, normalized to [0, 1]
- **Hue:** harmonic vs percussive ratio (`hp_ratio`) per time–frequency region

At render time, this point cloud is voxelized on the GPU into a scalar field defined over the fitted 3D grid bounds. The scalar field encodes how perceived loudness varies across time and frequency, while a secondary grid encodes harmonic–percussive balance. Isosurfaces in Song Mode are level sets of this loudness field; an isovalue selects a specific perceived loudness level, and the extracted surface shows where in the time–amplitude–frequency space the music achieves that loudness. Volume rendering in Song Mode ray-marches this point-derived scalar field to reveal the full internal structure of loudness and harmonic/percussive content.

### Isosurface Extraction

The system implements these two extraction algorithms:

- Marching Tetrahedrons
- Dual Contouring

In Song Mode, this field is a point-cloud-derived loudness density evaluated over (time, amplitude/log-amplitude, frequency). In Timeline Mode, it is a loudness/dynamics density evaluated over (timeline position, amplitude, crest factor). The algorithms generate meshes representing **isosurfaces of this scalar field**. For example, constant perceived-loudness surfaces in Song Mode and constant loudness/dynamics-density surfaces in Timeline Mode. An adjustable isovalue selects which scalar level to visualize, and surfaces are shaded with per-vertex normals and hue derived from the harmonic–percussive grid.

### Ray-Marched Volume Rendering

A full-screen ray marcher samples the voxelized scalar field (built from point clouds in either mode) to produce continuous volumetric visualizations.

- Implemented initially in a fragment shader
- Optional compute shader version for improved performance if time allows

Transfer functions map scalar values to color and opacity, enabling users to highlight different loudness features.

### Unified Rendering + UI

A single renderer supports both visualization modes and both rendering methods. Users can toggle:

- Timeline Mode ↔ Song Mode (with song selection)
- Isosurface ↔ Volume Rendering
- Transfer function presets
- Active isovalue

## Milestones

### Week 1 — Data Pipeline + Basic Rendering Setup

Goal: Build both datasets and get Metal environment ready.

### Tasks

- Gather songs for the decades to represent
- Extract audio features:
- Build Song Mode point clouds (per-song STFT-derived [x, y, z, scalar, hp_ratio])
- Build Timeline Mode point cloud (chronological stitching of per-song features into a single [x, y, z, scalar, hp_ratio] dataset)
- Implement voxelization from point clouds into scalar grids for both modes
- Normalize and smooth the resulting scalar grids as needed
- Set up Metal rendering environment (shaders, pipelines, buffers)

#### Deliverable

Both datasets built + Metal renderer initialized with volume textures loaded.

---

## Week 2 — Isosurface System (Core Algorithms + UI)

Goal: Get isosurfaces working for both modes.

### Tasks

- Implement Marching Tetrahedrons (first, simplest)
- Add ability to extract surfaces from:
  - Song Mode volumes
  - Trend Mode volume
- Compute normals + basic Phong shading
- Create UI for:
  - Song Mode ↔ Trend Mode toggle
  - Isovalue slider
- Begin implementing Dual Contouring
  
### Deliverable

Isosurface extraction working interactively on both datasets (at least Marching Tetrahedrons).

## Week 3 — Ray-Marched Volume Rendering + Integration

Goal: Implement GPU volume ray marching.
### Tasks

- Implement ray marching in fragment shader (Metal fragment function)
- Add transfer function(s):
  - density-based
  - loudness-based
- Integrate ray marcher with both datasets
- UI toggle:
  - Isosurface Rendering ↔ Volume Rendering

### Deliverable

Both datasets fully viewable in ray-marched volume rendering + shader toggle.

---

## Week 4 — Comparison, Polishing, and Final Report

Goal: Evaluate, polish, and document.
### Tasks

- Compare isosurface vs. volume rendering:
  - clarity
  - runtime
  - complexity
- Compare Marching Tetrahedrons vs. Dual Contouring
- Capture screenshots, figures, and renderings for report
- Polish UI and parameter sliders
- Write final report:
  - methodology
  - dataset design
  - rendering techniques
  - results
  - discussion
- Prepare final demo (video or live)

### Deliverable

Final polished visualization tool + full report + demo assets.