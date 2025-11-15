# Music Visualizer

## Abstract

This project visualizes the relationship between loudness and amplitude in popular music using a dual-mode system that operates at both macro and micro scales. Audio features from representative tracks across multiple decades are processed into 3D scalar fields, which are visualized using two high-level computer graphics methods: isosurface extraction and GPU-based ray-marched volume rendering. The system supports an aggregate trend mode that shows decade-level evolutions in loudness behavior, and an individual song mode that reveals fine-scale loudness structure within each track. This provides a comprehensive visualization framework for analyzing and exploring audio dynamics through advanced rendering techniques.

## Motivation

Loudness and dynamic range in popular music have changed significantly over time, especially over the last half century. However, these changes are rarely visualized in a way that reveals structure, continuity, and evolution. This project tackles that gap by converting audio-derived features into volumetric data and visualizing them using two complementary rendering techniques. The trend mode highlights how the loudness–amplitude relationship morphs across decades, while the song mode provides detailed insight into individual tracks. Together, these modes create a powerful exploration tool that is meaningful to both audio and graphics communities, while demonstrating high-level concepts in GPU rendering and surface extraction.
## Methodology

This project treats loudness–amplitude behavior as a scalar field to be visualized through both isosurfaces and ray-marched volumes. Two distinct data representations feed a single unified rendering pipeline:
### Trend Mode (Aggregate Volume)
Songs are grouped by decade. For each decade, amplitude, LUFS loudness, and optional frequency-band statistics are computed and aggregated into a 3D scalar field.
- **x-axis:** amplitude bin
- **y-axis:** loudness metric 
- **z-axis:** decade index
This produces a smooth volumetric field showing how loudness characteristics shift over long timescales.
### Song Mode (Individual Volumes)
Each track is processed independently into its own 3D scalar field, normalizing the time axis so that songs of different lengths fit a fixed resolution.
- **x-axis:** normalized time bin
- **y-axis:** loudness or amplitude band
- **z-axis:** frequency bin
### Isosurface Extraction
The system implements *at least* these two extraction algorithms
- Marching Cubes
- Marching Tetrahedrons
These generate mesh surfaces representing constant loudness–amplitude relationships. Surfaces are shaded with per-vertex normals and adjustable isovalues.
### Ray-Marched Volume Rendering
A full-screen ray marcher samples the 3D texture (from either mode) to produce continuous volumetric visualizations.
- Implemented initially in a fragment shader
- Optional compute shader version for improved performance if time allows
Transfer functions map scalar values to color and opacity, enabling users to highlight different loudness features.
### Unified Rendering + UI
A single renderer supports both visualization modes and both rendering methods. Users can toggle:
- Trend Mode ↔ Song Mode
- Isosurface ↔ Volume Rendering
- Transfer function presets
- Active isovalue
- Song/decade selection
## Milestones

### Week 1 — Data Pipeline + Basic Rendering Setup

Goal: Build both datasets and get Metal environment ready.
### Tasks
- Gather songs for the decades to represent
- Extract audio features:
    - RMS amplitude
    - LUFS loudness
    - FFT → frequency bins
- Build Song Mode volume (x = time, y = loudness/amplitude, z = frequency
- Build Trend Mode volume (amplitude × loudness histograms per decade)
- Normalize + smooth both volumes
- Set up Metal rendering environment (shaders, pipelines, buffers)
- Load 3D textures into GPU memory
#### Deliverable
Both datasets built + Metal renderer initialized with volume textures loaded.

---

## Week 2 — Isosurface System (Core Algorithms + UI)

Goal: Get isosurfaces working for both modes.
### Tasks
- Implement Marching Cubes (first, simplest)
- Add ability to extract surfaces from:
    - Song Mode volumes
    - Trend Mode volume
- Compute normals + basic Phong shading
- Create UI for:
    - Song Mode ↔ Trend Mode toggle
    - Isovalue slider   
- Begin implementing Marching Tetrahedrons 
### Deliverable
Isosurface extraction working interactively on both datasets (at least Marching Cubes).

---
## Week 3 — Ray-Marched Volume Rendering + Integration

Goal: Implement GPU volume ray marching.
### Tasks

- Implement ray marching in fragment shader (Metal fragment function)
- Add transfer function(s):
    - density-based
    - loudness-based
- Integrate ray marcher with both datasets
- UI toggle:
    - Isosurface Rendering ↔ Volume Renderin
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
- Compare Marching Cubes vs. Tetrahedrons
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
