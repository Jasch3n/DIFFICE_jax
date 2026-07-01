# Synthetic Data for X-PINN Joint Inversion

![Image showing an example of the generated data](data-1/syndata_full.svg)

## Overview

Let the velocity components be denoted as $(u,v)$, the thickness be $h$, the surface elevation be $s$, the density of ice be $\rho$, and let subscripts denote partial derivatives, then the shelfy-stream approximation (SSA) equation residuals are given by 

$$ g_1=[2\mu h(2u_x + v_y)]_x + [\mu h(u_y + v_x)]_y - \tau_{bx} - \rho g h s_x $$

$$ g_2=[2\mu h(u_x + 2v_y)]_y + [\mu h(u_y + v_x)]_x - \tau_{by} - \rho g h s_y $$

The data in this folder was generated with the finite-element package Ice-Sheet and Sea-Level System Model (ISSM). The model equations were the shelf-stream (for grounded ice) and shallow-shelf (for floating ice) approximations (altogether referred to as SSA).

There are many ways in which basal friction is parametrized. We have used both a power-law parametrization and a regularized-Coulomb parametrization to generate the synthetic datasets.

$$ \mathbf\tau_{b} = C |\mathbf u|^{1/m -1} \mathbf u \,\,\,\,\,\, (\text{Power Law}) $$

$$ \mathbf\tau_{b} = C \bigg(\frac{|\mathbf u|}{\mathbf u + u_0}\bigg)^{1/m -1}  \mathbf u\,\,\,\,\,\, (\text{Regularized Coulomb})$$

When performing inversion, however, it is sufficient to represent the friction simply

$$ \mathbf\tau_b = \alpha^2 \mathbf u $$

or 

$$ \mathbf \tau_b = \beta^2 \frac{\mathbf u}{|\mathbf u|} $$

Note that $\mathbf u$ is strictly the sliding speed at the ice-bed interface, and not the surface. In the SSA limit however, these two fields are the same (depth-independent flow).

where $\tau_{ij}$ is related to strain rate by Glen's Law in ISSM:

$$\tau_{ij} = 2\mu\dot\epsilon_{ij} = B\dot\epsilon_{eff}^{\frac{1}{n}-1} \dot\epsilon_{ij} $$

$$\dot\epsilon_{eff}^2 = \dot\epsilon_{xx}^2 + \dot\epsilon_{yy}^2 +\dot\epsilon_{xy}^2 + \dot\epsilon_{xx}\dot\epsilon_{yy} $$

No slip boundary conditions were applied at the boundary walls and the ice divide. Hydrostatic stress balance was applied at the ice shelf front.

Each `*.mat` file includes synthetic simulations of ice-stream-ice-shelf systems to be inverted by the PINN algorithms. WHen loaded into Python, each file yields an array of `n_region` arrays, where `n_region` is the number of partitions of the whole domain. For each of the `n_region` arrays, the following variables are stored:
- `xd` and `yd`: The x- and y- coordinates of FEM vertices at which velocities are calculated
- `xcol` and `ycol`: The x- and y-coordinates of the points in the domain where PINNs sample collocation points to evaluate the equation residuals. This may or may not coincide with `xd`/`yd`. In general, more collocation points means more accurate equation solutions.
- `xd_h` and `yd_h`: The x- and y- coordinates of FEM vertices at which thickness are calculated. These are the same as `xd` and `yd`, but this distinction is used in expectation of real data having different points where thickness data is available vs. those where velocity data is available.
- `ud` and `vd`: The x- and y- components (a.k.a. $u$ and $v$) of velocity corresponding to the entries in `xd` and `yd`
- `xct` and `yct`: The x- and y- coordinates of FEM vertices on the boundary of the domain (only relevant for floating regions).
- `nnct`: outward-pointing normal vectors at each boundary vertex (only relevant for floating regions).
- `bd_ud`, `bd_vd`, and `bd_mu`: The values of $u$, $v$, and $\mu$ at all FEM vertices on the boundary of the domain corresponding to entires in `xct` and `yct`. 
- `alpha2d` and `mud`: The values of $\alpha^2$ (*not* $\beta^2$ per the notation above) and $\mu$ corresponding to entries in `xd` and `yd`
- `basal_mask`: boolean variable indicating whether the region is grounded (`True`) or not (`False`). 
- `hd` and `sd`: The thickness and surface elevation at FEM vertices corresponding to entries in `xd_h` and `yd_h`. In applicable cases, tje `hd` field contains artificially sparse measurements to emulate radar survey tracks, and the full set of dense measurement can be found in `h_dense`.
- `ols_d` contains the "ocean level-set" variable from ISSM. Positive `ols_d` means that the corresponding point is grounded. Negative means that it is floating. Zero means that the point is on the grounding line.

The other variables are not too relevant for the purposes here. 

The nomenclature here mostly follows the DIFFICE_jax code developed by our group.