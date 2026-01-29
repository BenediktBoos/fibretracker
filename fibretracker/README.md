# FibreTracker Analysis Pipeline (`main.py`)

This repository contains a Python pipeline for quantitative analysis of 3D fibre networks from volumetric microscopy data. The main entry point is `main.py`, which orchestrates fibre detection, tracking, visualization, and multiple downstream quantitative analyses.

The pipeline is designed for reproducible, logged execution and produces both numerical outputs and publication-ready visualizations.

---

## Overview

The script performs the following high-level steps:

1. Load a 3D TIFF volume  
2. Load precomputed fibre tracks **or** detect and track fibres from scratch  
3. Compute fibre-level geometric and statistical descriptors  
4. Generate visualizations and quantitative plots  
5. Log parameters, timing, and settings for reproducibility  

All outputs are written to a timestamped result directory next to the input data.

---

## Input Data

### Required
- **3D image volume** (`.tif`)  
  A volumetric TIFF file representing the fibre network.

### Optional
- **Precomputed tracks** (`tracks.pkl`)  
  If provided, fibre detection and tracking are skipped.

---

## Output Structure

For each run, a new result folder is created:

```
result/
└── YYYY-MM-DD_HH_MM_SS_<volume_name>/
    ├── settings/
    │   ├── general_settings.json
    │   ├── *.json              # logged parameters and timings
    ├── found_tracks.pkl
    ├── length_vs_slice_heatmap.png
    ├── track_lengths_*.png
    ├── FOD_*.png
    ├── overlap_*.png
    └── ...
```

---

## Main Processing Steps

### Fibre Detection (Optional)
If `read_tracks = False`, fibre center points are detected using a multi-scale Laplacian-of-Gaussian approach, followed by tracking based on spatial continuity and momentum constraints.

### Fibre Tracking
Detected fibre centers are linked into 3D tracks. Optional filtering removes short or cropped tracks.

### Fibre Length Computation
Fibre lengths are computed as the cumulative Euclidean distance along each track and converted to physical units using the voxel size.

### Visualization
- 3D track rendering  
- Slice overlays with fibre tracks  
- Alpha-shape overlays for qualitative inspection  

### Fibre Length vs Slice Heatmap
A 2D heatmap showing fibre-length distributions as a function of z-slice, highlighting spatial heterogeneity.

### Fibre Orientation Distribution (FOD)
Signed fibre orientation angles are computed relative to a reference axis (default: z-axis) and visualized as orientation distributions.

### Length Distribution and CDF
Fibre length histograms and cumulative distribution functions (CDFs) are generated for quantitative comparison.

### Fibre Overlap Analysis
Local fibre overlap is computed using k-nearest-neighbour adjacency in the lateral plane and summarized statistically.

### Alpha-Shape Analysis
Slice-wise alpha-shape masks estimate fibre occupancy and structural connectivity.

---

## Configuration Parameters

Key parameters defined in `main.py`:

```python
voxel_size = 0.8        # voxel size in micrometers
read_tracks = True     # load existing tracks instead of recomputing
manipulate_tracks = False
geodict_export = True
```

---

## Dependencies

- numpy  
- scipy  
- tifffile  
- matplotlib  

Project-specific modules:
- fibretracker  
- fibretracker_helper  
- helper  

---

## Reproducibility

- All parameters and execution times are logged
- Each run produces a self-contained result directory
- No files are overwritten between runs

---

## Usage

1. Run `main.py`
2. Select a 3D `.tif` volume when prompted
3. Optionally select a `tracks.pkl` file
4. Inspect results in the generated `result/` directory

---

## Author

**boos**  
Created September 16, 2025
