# Music_Visualizer

## Abstract

***
This project proposes a visualization system that models the relationship between amplitude, perceptual loudness, and time for popular music across multiple decades using isosurfaces extracted from a 3D scalar field. Audio features will be converted into volumetric data, from which multiple isosurface extraction algorithms (Marching Cubes, Marching Tetrahedrons, etc.) will be implemented and compared. The goal is to reveal how loudness characteristics have changed over time within different musical eras and evaluate how different surface extraction techniques affect the clarity, topology, and interpretability of the resulting visualizations.

## Motivation

***
Modern music production has undergone substantial shifts in loudness and dynamic range over the last half-century, which is often discussed but rarely visualized in a structured, graphics-driven way. This project matters because it turns a familiar cultural topic into a volumetric dataset that can be explored through computer graphics techniques. It provides a tangible example of how rendering algorithms translate abstract data into interpretable 3D structure. The visualization may help illustrate long-term loudness trends while also demonstrating strengths and weaknesses of widely used isosurface algorithms.

## Methodology

***

## Milestones

***

#### Week 1 - Dataset & Preprocessing

- Gather songs
- Implement audio feature extraction
- Build and validate volumetric data representation

#### Week 2 - Rendering Environment

- Set up Metal rendering environment 
- Create a volume using extracted audio features
-  Implement first isosurface extraction algorithm (Marching Cubes, Marching tetrahedrons, etc) with static isovalue

#### Week 3

- Continue implementing isosurface extraction algorithms
- implement dynamic isovalue rendering
- implement algorithm selection toggles

#### Week 4

- Compare algorithms (runtime mesh complexity)
- Prepare figures, screenshots, performance tables
- Write final report and prepare demo