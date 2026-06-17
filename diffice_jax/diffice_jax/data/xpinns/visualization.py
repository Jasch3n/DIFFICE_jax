"""
Visualization utilities for preprocessed XPINN data.

Plots the normalized data returned by normalize_data(), including:
- Per-subregion velocity, thickness, and surface elevation
- Boundary data (calving front with normals)
- Interface (matching) data between subregions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def plot_xpinn_data(data_all, idxgall, basal_mask=None, figsize_scale=4):
    """
    Plot preprocessed XPINN data after normalization.

    Parameters
    ----------
    data_all : list
        Output of normalize_data(). List of tuples, one per subregion.
        Each tuple: (X_star, U_star, X_ct, nnct, data_info, X_md, [boundary_star])
        - X_star[0]: (N_v, 2) normalized velocity coordinates
        - X_star[1]: (N_h, 2) normalized thickness coordinates
        - U_star[0]: (N_v, 2) normalized velocity (u, v)
        - U_star[1]: (N_h, 1) normalized thickness
        - U_star[2]: (N_h, 1) normalized surface elevation (basal only)
        - X_ct: (N_bd, 2) normalized boundary coordinates
        - nnct: (N_bd, 2) outward normals
        - data_info: [data_mean, data_range, data_norm, data_raw, ...]
        - X_md: (N_md, 2) normalized interface coordinates
    idxgall : list of int
        Index list for subregions.
    basal_mask : list of bool, optional
        Per-subregion flag: True = grounded (basal), False = floating.
        Defaults to all False.
    figsize_scale : float
        Scaling factor for figure size.
    """
    ng = len(idxgall)
    if basal_mask is None:
        basal_mask = [False] * ng

    # ------------------------------------------------------------------
    # Figure 1: Sample data per subregion (velocity magnitude + thickness)
    # ------------------------------------------------------------------
    n_rows = 3 if any(basal_mask) else 2
    fig, axes = plt.subplots(n_rows, ng, figsize=(figsize_scale * ng, figsize_scale * n_rows),
                             squeeze=False)
    fig.suptitle('Normalized Sample Data per Subregion', fontsize=14, y=1.02)

    for i in idxgall:
        X_v = np.array(data_all[i][0][0])    # velocity coords (N_v, 2)
        U_v = np.array(data_all[i][1][0])     # velocity (N_v, 2)
        X_h = np.array(data_all[i][0][1])    # thickness coords (N_h, 2)
        H   = np.array(data_all[i][1][1])     # thickness (N_h, 1)

        is_basal = basal_mask[i]
        label = 'grounded' if is_basal else 'floating'

        # Velocity magnitude
        vel_mag = np.sqrt(U_v[:, 0]**2 + U_v[:, 1]**2)
        ax = axes[0, i]
        sc = ax.scatter(X_v[:, 0], X_v[:, 1], c=vel_mag, s=1, cmap='jet')
        ax.set_title(f'Region {i} ({label})\n|V| (norm.)')
        ax.set_aspect('equal')
        plt.colorbar(sc, ax=ax, fraction=0.046)

        # Thickness
        ax = axes[1, i]
        sc = ax.scatter(X_h[:, 0], X_h[:, 1], c=H[:, 0], s=1, cmap='viridis')
        ax.set_title('H (norm.)')
        ax.set_aspect('equal')
        plt.colorbar(sc, ax=ax, fraction=0.046)

        # Surface elevation (grounded only)
        if n_rows == 3:
            ax = axes[2, i]
            if is_basal and len(data_all[i][1]) > 2:
                S = np.array(data_all[i][1][2])
                sc = ax.scatter(X_h[:, 0], X_h[:, 1], c=S[:, 0], s=1, cmap='terrain')
                ax.set_title('S (norm.)')
                plt.colorbar(sc, ax=ax, fraction=0.046)
            else:
                ax.text(0.5, 0.5, 'N/A\n(floating)', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12, color='gray')
                ax.set_title('S (norm.)')
            ax.set_aspect('equal')

    for ax in axes.flat:
        ax.set_xlabel('x (norm.)')
        ax.set_ylabel('y (norm.)')

    fig.tight_layout()

    # ------------------------------------------------------------------
    # Figure 2: Boundary & interface data (all subregions overlaid)
    # ------------------------------------------------------------------
    fig2, ax2 = plt.subplots(1, 1, figsize=(figsize_scale * 2, figsize_scale * 2))
    ax2.set_title('Boundaries & Interfaces (normalized coords)')

    # Define colors for subregions
    region_colors = plt.cm.Set1(np.linspace(0, 1, max(ng, 3)))

    for i in idxgall:
        X_v = np.array(data_all[i][0][0])
        color = region_colors[i]
        label_str = f'Region {i} ({"grounded" if basal_mask[i] else "floating"})'

        # Plot sample points (faded)
        ax2.scatter(X_v[:, 0], X_v[:, 1], c=[color], s=0.5, alpha=0.15, label=label_str)

        # Boundary data (calving front)
        X_ct = np.array(data_all[i][2])
        nnct = np.array(data_all[i][3])
        if X_ct.size > 0 and nnct.size > 0:
            ax2.scatter(X_ct[:, 0], X_ct[:, 1], c='k', s=8, zorder=5)
            ax2.quiver(X_ct[:, 0], X_ct[:, 1], nnct[:, 0], nnct[:, 1],
                       scale=30, width=0.003, color='k', zorder=5)

    # Interface data
    # X_md for subregion i contains the interface points relevant to that region.
    # For region 0: x_md[0]; for region ng-1: x_md[ng-2]; for middle: both.
    # We plot the raw interface from data_all[i][5] for each region.
    plotted_interfaces = set()
    for i in idxgall:
        X_md = np.array(data_all[i][5])
        if X_md.size > 0:
            # Avoid plotting duplicate interface data
            key = (X_md.shape[0], round(float(X_md[0, 0]), 6))
            if key not in plotted_interfaces:
                ax2.scatter(X_md[:, 0], X_md[:, 1], c='red', s=6, zorder=4,
                            marker='x', linewidths=0.5,
                            label=f'Interface' if len(plotted_interfaces) == 0 else '')
                plotted_interfaces.add(key)

    ax2.set_xlabel('x (norm.)')
    ax2.set_ylabel('y (norm.)')
    ax2.set_aspect('equal')
    ax2.legend(loc='best', markerscale=5, fontsize=8)
    fig2.tight_layout()

    # ------------------------------------------------------------------
    # Figure 3: Data statistics per subregion
    # ------------------------------------------------------------------
    fig3, axes3 = plt.subplots(1, ng, figsize=(figsize_scale * ng, figsize_scale),
                               squeeze=False)
    fig3.suptitle('Velocity Distribution per Subregion (normalized)', fontsize=13)

    for i in idxgall:
        U_v = np.array(data_all[i][1][0])
        ax = axes3[0, i]
        ax.hist(U_v[:, 0], bins=50, alpha=0.6, label='u', density=True)
        ax.hist(U_v[:, 1], bins=50, alpha=0.6, label='v', density=True)
        ax.set_title(f'Region {i}')
        ax.set_xlabel('Normalized velocity')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)

    fig3.tight_layout()

    # ------------------------------------------------------------------
    # Figure 4: Normalization info summary
    # ------------------------------------------------------------------
    print("=" * 60)
    print("XPINN Preprocessing Summary")
    print("=" * 60)
    for i in idxgall:
        data_info = data_all[i][4]
        dmean = np.array(data_info[0])
        drange = np.array(data_info[1])
        X_v = data_all[i][0][0]
        X_h = data_all[i][0][1]
        X_ct = data_all[i][2]
        X_md = data_all[i][5]
        is_basal = basal_mask[i]

        label = 'grounded' if is_basal else 'floating'
        print(f"\nRegion {i} ({label}):")
        print(f"  Velocity points:   {X_v.shape[0]}")
        print(f"  Thickness points:  {X_h.shape[0]}")
        print(f"  Boundary points:   {X_ct.shape[0]}")
        print(f"  Interface points:  {X_md.shape[0]}")
        var_names = ['x', 'y', 'u', 'v', 'h']
        if is_basal and len(dmean) > 5:
            var_names.append('s')
        print(f"  Mean:  {dict(zip(var_names, dmean))}")
        print(f"  Range: {dict(zip(var_names, drange))}")
    print("=" * 60)

    plt.show()
    return fig, fig2, fig3
