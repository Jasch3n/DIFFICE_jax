# ISSMAdjoint Shelf Inversion Configs

Each YAML file defines one ISSM adjoint shelf inversion experiment. Paths are
resolved relative to the YAML file unless they are absolute. Shared defaults
such as ISSM paths, mesh controls, physics constants, and inversion controls
live in `../shared/config/config_defaults.m`.

Typical MATLAB usage from `examples/real_data/ISSMAdjoint`:

```matlab
run_inversion
```

Change `config_name` inside `run_inversion.m` to pick a config.

Override values directly in a copied YAML file when running experiments with
different mesh settings, regularization weights, solver iteration counts, or
output directories.

Dataset paths are specified in each shelf config:

```yaml
data:
  bedmachine_file: /path/to/BedMachineAntarctica-v4.nc
  measures_file: /path/to/insar_antarctica_ice_velocity_450m_v2.nc
  bedmachine_bounds: [1.63e6, 2.29e6, 0.56e6, 0.89e6]
  bedmachine_clip: {}
```

## Domain Selection

The working ISSM mesh domain is built directly from BedMachine v4 using:

- `data.bedmachine_bounds` - the EPSG:3031 bounding box used to read the
  BedMachine mask
- `data.bedmachine_clip` - optional extra clipping inside the bounds, using
  `xmin`, `xmax`, `ymin`, and/or `ymax`

Step `0` writes `Geometry/<Shelf>_Outline.exp` from BedMachine floating-ice mask
transitions. For shelves that share a broad BedMachine window, use
`bedmachine_clip` to isolate the shelf. `larsenc.yaml` and `larsend.yaml` are
the current examples: they use the same BedMachine bounds and opposite x clips.

`bedmachine_clip` is applied after `bedmachine_bounds`. The bounds choose the
BedMachine window to read, then the clip removes floating-ice mask cells outside
any configured `xmin`, `xmax`, `ymin`, or `ymax` limits. You only need to set
the limits that matter:

```yaml
bedmachine_clip:
  xmax: -2.0e6
```

For example, Larsen C and Larsen D share:

```yaml
bedmachine_bounds: [-2.38e6, -1.55e6, 0.88e6, 1.34e6]
```

Then `larsenc.yaml` keeps the part with `x <= -2.0e6`:

```yaml
bedmachine_clip:
  xmax: -2.0e6
```

and `larsend.yaml` keeps the part with `x >= -2.0e6`:

```yaml
bedmachine_clip:
  xmin: -2.0e6
```

Each config is single-weight:

```yaml
inversion:
  regularization_weight: 1.0e-17
```

A future L-curve analysis should sweep over a series of config files, each with
a different `inversion.regularization_weight`, then compare the saved result
files.

The smoke inversion can reuse the config's inversion weight:

```yaml
smoke:
  regularization_weight:
    use_inversion_weight: true
```
