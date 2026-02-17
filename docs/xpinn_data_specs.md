# XPINN Data Specifications for Simultaneous Inversion

This document provides an executive summary of the required input data format for the XPINN framework when performing simultaneous inversion of ice viscosity (floating/grounded) and basal friction (grounded).

## Domain Structure
The ice shelf/stream domain is partitioned into $N$ sub-regions. Each sub-region is associated with a `basal_mask` boolean:
- `False`: Floating ice (SSA equations).
- `True`: Grounded ice (SSA + basal friction).

## Input `data` Dictionary Format
The training data is structured as a dictionary where each key contains a list of arrays (one per sub-region).

### 1. Sample Data (`data['smp']`)
Observed data points used for the data-fitting term of the loss function.
- **Index 0 (`X_v`):** $(x, y)$ coordinates for velocity [Shape: $N^d_{pts} \times 2$].
- **Index 1 (`U`):** Velocity vector $(u, v)$ [Shape: $N^d_{pts} \times 2$] (normalized).
- **Index 2 (`X_h`):** $(x, y)$ coordinates for thickness and elevation [Shape: $N^h_{pts} \times 2$].
- **Index 3 (`H`):** Ice thickness $h$ [Shape: $N^h_{pts} \times 1$] (normalized).
- **Index 4 (`S`):** Surface elevation $s$.
    - **Floating Region:** `None` (thickness-dependent elevation is calculated analytically).
    - **Grounded Region:** Measured surface elevation [Shape: $N^h_{pts} \times 1$] (normalized).

Note that $N^d_{pts}$ and $N^h_{pts}$ are not necessarily the same, and the set of points in general are not the same. This is because ice sheet data is collected using different instruments at different times. 

### 2. Boundary Data (`data['bd']`)
External boundary conditions.
- **Index 0 (`X_bd`):** $(x, y)$ coordinates of boundary points [Shape: $N^{col}_{pts} \times 2$].
- **Index 1 (`Val_1`):** 
    - **Floating Region:** Unit outward normal vector $\mathbf{n} = (n_x, n_y)$ for calving front stress balance.
    - **Grounded Region:** `None` (Boundary values are typically imposed organically via interface matching).

### 3. Collocation Points (`data['col']`)
Points where the governing physical equations (PDEs) are enforced.
- **Index 0 (`X_col`):** $(x, y)$ coordinates [Shape: $N_{pts} \times 2$].

### 4. Matching Data (`data['md']`)
Points on the interfaces between adjacent sub-regions used to enforce continuity.
- **Index 0 (`X_md`):** Paired coordinates $[x_1, y_1, x_2, y_2]$ where $(x_1, y_1)$ is in the current region and $(x_2, y_2)$ is the identical point in the adjacent region.

## Coordination Across Boundaries
Continuity is enforced at the interface (grounding line or sub-region boundary) for:
- **Velocity** $(u, v)$
- **Thickness** $h$
- **Viscosity** $\mu$

For grounded regions, the viscosity boundary condition is not explicitly provided but is "organically" constrained by matching the viscosity of the adjacent floating ice networks.
