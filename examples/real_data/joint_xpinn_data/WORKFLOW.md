# joint_xpinn_data workflow guide

This is the user-facing how-to for the pipeline. For architecture/design rationale, see the root `../CLAUDE.md`. For known limitations and suggested next work, see `HANDOFF.md`.

## What this produces

A `.mat` file for one ice-shelf + grounding-zone pair, in the exact field layout DIFFICE_jax's synthetic-data loader expects (two regions — grounded grounding-zone, floating shelf — each with velocity/thickness/collocation/boundary data). Built from real observational data: MEaSURES InSAR velocity, BedMachine + BEDMAP1 thickness, and a choice of grounding-line/calving-front products.

## Setup

```bash
pyenv activate Cpu-Diffice-Env
cd /Users/jiapchen/Software/DIFFICE_jax/examples/real_data
```

All commands below assume this working directory (so `joint_xpinn_data` resolves as a package) and this environment (has DIFFICE_jax, numpy, scipy, h5py, shapely, pyproj, pandas, netCDF4, rasterio, scikit-image, matplotlib already installed).

## 1. Check what shelf/basin names are available

The basin and ice-shelf names are exact strings from the MEaSURES Antarctic Boundaries dataset — check spelling before using a new one:

```python
from joint_xpinn_data.data_sources import boundaries
from joint_xpinn_data.config import DEFAULT_PATHS

print(boundaries.list_names(str(DEFAULT_PATHS["iceshelves"])))       # e.g. "Amery"
print(boundaries.list_names(str(DEFAULT_PATHS["basins_refined"])))   # e.g. "Lambert", "Mellor", "Fisher"
```

Note: basin names are the *individual tributary glacier* granularity (Lambert/Mellor/Fisher are separate), not the coarser IMBIE-style whole-catchment basins — that's what makes picking "one grounding zone" for a multi-tributary shelf possible.

## 2. Build a dataset

### Config-driven (recommended)

Each dataset you want is one YAML file under `joint_xpinn_data/configs/` — `buffer_km` (how far upstream of the grounding line the grounded region extends) and any data-source overrides live there. Copy `configs/TEMPLATE.yaml` for the full list of fields, or one of the existing `configs/amery_*.yaml` files as a starting point.

```bash
# build one config
/Users/jiapchen/.pyenv/versions/3.13.3/envs/Cpu-Diffice-Env/bin/python -m joint_xpinn_data.build_dataset \
    joint_xpinn_data/configs/amery_lambert.yaml

# build every *.yaml in the directory (TEMPLATE.yaml is skipped)
/Users/jiapchen/.pyenv/versions/3.13.3/envs/Cpu-Diffice-Env/bin/python -m joint_xpinn_data.build_dataset \
    joint_xpinn_data/configs/
```

Output defaults to `joint_xpinn_data/output/<ice_shelf>_<grounding_zone>_<buffer_km>km/<ice_shelf>_<grounding_zone>_<buffer_km>km.mat` (override the root with `--out-dir`) — everything scoped to this (ice_shelf, grounding_zone, buffer_km) triple (the `.mat`, its validation figures, its source-comparison and consistency-check figures) lives together in this one per-config folder.

An unknown field in a YAML raises a clear error listing the valid `PipelineConfig` fields, so a typo doesn't fail silently.

### Programmatic (for one-off experiments, or driving the pipeline from other code)

```python
from joint_xpinn_data.config import PipelineConfig
from joint_xpinn_data.build_dataset import save_dataset, default_output_path

config = PipelineConfig(ice_shelf="Amery", grounding_zone="Lambert", buffer_km=100)
save_dataset(config, str(default_output_path(config, "joint_xpinn_data/output")))
```

### Swapping data sources

Every source is a config field (works the same whether set in YAML or in `PipelineConfig(...)` directly), defaulting to the currently-recommended one:

```yaml
grounding_line_source: bedmachine_mask       # default: measures_boundaries_2008
calving_front_source: antarctic_boundaries_mask  # default: bedmachine_mask
dense_thickness_source: bedmachine_v3         # default
sparse_thickness_source: bedmap1_csv          # default (also: bedmap2_csv, concat, custom_xy)
dense_surface_source: bedmachine_v3           # default
sparse_surface_source: bedmap1_csv            # default
velocity_source: measures_v2                  # default
```

Surface elevation (`sd`/`s_dense`) is sourced independently of thickness — see root `CLAUDE.md`'s note on `xd_s`/`yd_s` and `docs/adr/0001-surface-elevation-independent-coordinates.md` — so it has its own pair of source fields rather than riding along with `dense_thickness_source`/`sparse_thickness_source`.

Check the `*_SOURCES` dict in the relevant `data_sources/*.py` module for what's registered. Source-specific tuning knobs (e.g. `tol_km`, `pad_km`) go through the matching `*_kwargs` field, e.g. `grounding_line_kwargs: {tol_km: 20.0}`.

For thickness, `concat` merges multiple named sources into one — e.g. all three Amery configs combine `bedmap1_csv` (1966-2000) with `bedmap2_csv` (BGR's 2002-2003 PCMEGA survey) for sparse thickness, since neither alone covers the Lambert/Mellor/Fisher area as densely as the two together:

```yaml
sparse_thickness_source: concat
sparse_thickness_kwargs:
  sources:
    - source: bedmap1_csv
    - source: bedmap2_csv
```

To plug in a source not yet registered (e.g. a newer/different GL or front product), use `"custom_xy"` with either a raw array or a file path — see `process_custom_geometry`/`process_custom_points` docstrings in `grounding_line.py`/`velocity.py`.

A `paths:` block in a YAML config overrides individual file paths (e.g. to point at BedMachine v4 instead of v3) without needing to repeat every other path — only the keys you list are changed.

### Excluding slow-moving grounded ice

`grounded_min_speed_myr` (default `None`) drops grounded-region velocity points at or below the given speed in m/yr, along with everything resampled/derived from them (`xd`/`yd`/`ud`/`vd`, `xcol`/`ycol`, `h_dense`/`s_dense`, `ols_d`). Only affects the grounded region — there's no floating-region equivalent. All three Amery configs set this to `100.0`:

```yaml
grounded_min_speed_myr: 100.0
```

### Controlling grounding-line resolution

Before anything else, `domain.build_regions` trims the loaded grounding line to just the points actually near where the grounded and floating regions touch (`_trim_gl_to_interface`) — a GL source's own inclusion criterion doesn't know about the ice-shelf polygon at all, so the raw data can include points trailing off along a basin's lateral wall well past the true interface (this was a real, pre-existing bug for Amery, not a resampling artifact — see HANDOFF.md). This trimming always happens, regardless of `grounding_line_resample_m`.

`grounding_line_resample_m` (default `None`) then artificially resamples the *trimmed* grounding line to an even arc-length spacing (meters) via linear interpolation — the grounded region's cut boundary, `ols_d`, and the reported `x_md`/`y_md` all pick up the resampled version (the corridor buffer uses the raw, untrimmed points, since it's built before trimming and the buffer is huge relative to a GL source's stray tail points anyway). All three Amery configs now set this to `500.0` — no two consecutive `x_md`/`y_md` points are more than 500m apart, vs. `measures_boundaries_2008`'s native 54/24/23-point (Lambert/Mellor/Fisher) resolution, which HANDOFF.md flags as a source of `ols_d` overshoot near vertices:

```yaml
grounding_line_resample_m: 500.0   # no two consecutive points more than 500m apart
```

This assumes the source is a genuinely ordered polyline (`Geometry.ordered=True`, true for `measures_boundaries_2008`) — a real line to interpolate along. A mask-based unordered pixel-scatter source (`bedmachine_mask`, `Geometry.ordered=False`) has no such line; resampling one anyway triggers a loud `UNORDERED GEOMETRY IS BEING RESAMPLED` warning and falls back to sorting the points counterclockwise around their centroid first (`utils/geometry_utils.order_counterclockwise`) — a heuristic that's only exact for a star-shaped point cloud, not a real boundary reconstruction. If you see that warning, plot the result before trusting it.

## 3. Visually validate the build

```bash
/Users/jiapchen/.pyenv/versions/3.13.3/envs/Cpu-Diffice-Env/bin/python -m joint_xpinn_data.diagnostics.plot_validation \
    joint_xpinn_data/output/Amery_Lambert_100km/Amery_Lambert_100km.mat
```

Produces 4 PNGs in `<mat's folder>/figures/` (or pass `--out-dir`): regions/grounding-line/calving-front/cut-boundary overlay, per-region collocation point clouds, sparse-vs-dense thickness, velocity magnitude + `ols_d`. Add `--show` to also open them interactively.

## 4. Compare alternative GL/front sources before trusting a swap

`compare_sources` and both `checks` subcommands (below) accept **`--config <path>`** as an alternative to bare `--ice-shelf`/`--grounding-zone`/`--buffer-km` flags — pass it once you have a saved config YAML. This matters more than it sounds like it should: without `--config`, these tools build a `PipelineConfig` from *only* the three bare flags, silently defaulting every other field (`region_strategy`, `floating_region_source`, `grounding_line_kwargs`, etc.) rather than reading the shelf's actual saved config. For a shelf whose real config matches those defaults (every Amery config today) this is invisible. For one that doesn't — e.g. Ross East/Byrd's `region_strategy: flow_restricted` — it silently evaluates a *different, unbuilt* domain instead of the one that was actually saved: confirmed directly, the bare-flag form built a floating region 6.7x larger than Byrd's real corridor (188,548 km² vs. 28,335 km²) with no error or warning. Use bare flags only for the genuinely config-less case in step 2 below, where no YAML exists yet; use `--config` for everything after a config file exists (step 5).

```bash
# before a config file exists: quick source comparison for a candidate shelf/zone
/Users/jiapchen/.pyenv/versions/3.13.3/envs/Cpu-Diffice-Env/bin/python -m joint_xpinn_data.diagnostics.compare_sources \
    --ice-shelf Amery --grounding-zone Lambert --buffer-km 100

# once a config file exists: use it, so any grounding_line_kwargs/calving_front_kwargs it sets are honored
/Users/jiapchen/.pyenv/versions/3.13.3/envs/Cpu-Diffice-Env/bin/python -m joint_xpinn_data.diagnostics.compare_sources \
    --config joint_xpinn_data/configs/amery_lambert.yaml
```

Overlays each registered GL source pair and each front source pair, prints bidirectional nearest-neighbor distance stats (mean/median/max), and saves comparison PNGs into the same per-config `figures/` folder as `plot_validation`. Useful for judging whether two products actually agree before picking one as default, or for understanding how much a boundary moved between two products' epochs. (This particular tool's own GL/front comparisons operate on the raw basin/shelf polygons regardless of `region_strategy`/`floating_region_source` — so for it specifically, `--config` only matters if the shelf sets `grounding_line_kwargs`/`calving_front_kwargs`. Still use `--config` once one exists, for consistency and so this doesn't become another silent gap later.)

## 5. Run consistency checks

Every registered check (`joint_xpinn_data.checks.CHECKS`) is a subcommand of `diagnostics/checks.py` — each takes `(config, regions, **kwargs)` and returns a `CheckResult` (see root `CLAUDE.md`). **Always use `--config` here** once a config file exists — unlike `compare_sources`, these checks call `domain.build_regions(config)` directly, so a bare-flags config genuinely builds and checks the wrong regions for any shelf that isn't `region_strategy: simple`/`floating_region_source: whole_shelf` (Amery's case, not Byrd's).

**`velocity_vs_front`** — velocity and calving-front data are essentially never from the same epoch, and ice fronts move — this quantifies the mismatch rather than assuming it away:

```bash
/Users/jiapchen/.pyenv/versions/3.13.3/envs/Cpu-Diffice-Env/bin/python -m joint_xpinn_data.diagnostics.checks velocity_vs_front \
    --config joint_xpinn_data/configs/amery_lambert.yaml --search-radius-km 5
```

Reports the signed distance to the calving front for velocity points within `search_radius_km` (positive = seaward = misfit) and a strict pass/fail (any seaward point at all fails), plus a spatial PNG colored by signed distance. `search_radius_km` only controls which velocity points get pulled in for evaluation — it's not a pass/fail tolerance (the check itself allows zero seaward points, always) and pushing it too far can start pulling in unrelated coastline the front geometry was never tracked to, which can turn a PASS into a FAIL rather than the other way around. Keep it small.

Note this check will still report FAILED for a systematic front/velocity disagreement even though `build_dataset` now resolves it automatically: `domain.build_regions` compares the configured front against the boundary implied by the velocity data's own coverage (`calving_front_source="velocity_mask"`) and, on a >500m median disagreement, warns and either erodes the floating polygon inward (front ocean-ward of real velocity coverage) or drops velocity points seaward of the front (front landward of it, the case observed for Amery and Ross East/Byrd) — see `CLAUDE.md`'s "Pipeline order" section. This check deliberately re-queries velocity fresh around the *configured* front, independent of that reconciliation, so it's still the right tool for detecting the raw disagreement — just don't expect a FAILED result here to mean the built `.mat` still has the problem.

**`hydrostatic_equilibrium`** — checks whether the floating region's thickness and surface elevation satisfy flotation (`delta = rho_ice*hd + rho_seawater*(surface-hd)`, zero at exact flotation):

```bash
/Users/jiapchen/.pyenv/versions/3.13.3/envs/Cpu-Diffice-Env/bin/python -m joint_xpinn_data.diagnostics.checks hydrostatic_equilibrium \
    --config joint_xpinn_data/configs/amery_lambert.yaml
```

Purely descriptive by default (no threshold) — `delta` is never exactly zero even for good data, both from measurement noise and firn's lower density near the surface (not modeled by this two-density formula). Pass `--threshold-kgm2` once you've decided what an acceptable residual looks like empirically. Reports the distribution (mean/median/percentiles) and saves a spatial PNG colored by `delta`, clipped to the 90th percentile of `|delta|` so a handful of extreme outliers (e.g. rock-outcrop/margin sentinel artifacts) don't wash out the pattern.

## Typical order for a new shelf/zone

1. `boundaries.list_names(...)` to confirm exact spelling for the shelf and the tributary/basin you want as the grounding zone.
2. `compare_sources` (bare `--ice-shelf`/`--grounding-zone`/`--buffer-km` flags — no config file exists yet) to see whether the default GL/front sources agree well enough, before committing to a `buffer_km`.
3. Write a `joint_xpinn_data/configs/<shelf>_<zone>.yaml` with a first-guess `buffer_km` and build it.
4. `plot_validation` on the result — check the grounded-region shape looks like a sensible glacier corridor (not implausibly tiny/huge) and the calving front only covers the true open-ocean-facing arc.
5. Run both `checks` subcommands **with `--config joint_xpinn_data/configs/<shelf>_<zone>.yaml`** to see how much near-front data is ambiguous and how well thickness/surface satisfy flotation, and decide whether to act on either before training. Using bare flags here instead is a real risk, not just a style choice, for any shelf whose config sets `region_strategy`, `floating_region_source`, or other non-default region-building fields — see step 4's note above.
