# joint_xpinn_data

Builds PINN training datasets for a chosen ice-shelf + grounding-zone pair, from real observational data instead of synthetic ISSM output.

## Language

**Config**:
One `PipelineConfig` — the (ice_shelf, grounding_zone, buffer_km) triple plus any source overrides, identifying one dataset build. Canonically named `<ice_shelf>_<grounding_zone>_<buffer_km>km`; this stem names the output `.mat`, its output folder, and the figures inside it, regardless of whether the config came from a saved YAML or a one-off programmatic call.
_Avoid_: "run", "build" (as nouns) for this concept — reserve "build" for the act of producing one.

**Consistency check**:
A quantitative comparison between two independently-observed physical quantities that should agree if the region model and source data are both correct (e.g. velocity vs. mapped calving front, thickness vs. surface elevation via flotation). Produces a `CheckResult`: a per-point signed `metric` plus an optional `threshold`/`passed` judgement. Distinct from a code-correctness test (verifies pipeline logic regardless of which data is fed in) — `joint_xpinn_data/tests/regression_baseline.py` is the one test of that kind (frozen-output regression against known-good reference configs, not a per-shelf physical judgement); "check" always means this data-consistency sense.
_Avoid_: "test", "validation" for this concept — "validation" also collides with `plot_validation.py`, which visualizes rather than checks.

**Surface elevation**:
The ice surface height above sea level (the `s`/`sd`/`s_dense` fields in the output schema). Sourced independently of thickness — its own data kind (own registry, own native x/y per source) — even when a given provider happens to read it from the same file as thickness (e.g. BEDMAP1's `surface_altitude` column, same rows as its `land_ice_thickness` column). Never derived *from* thickness.

**Hydrostatic equilibrium (flotation) delta**:
`delta = rho_ice * thickness + rho_seawater * (surface_elevation - thickness)`, with `rho_ice=917`, `rho_seawater=1023` kg/m^3. Zero delta means the ice column is exactly floating (weight of ice = weight of displaced seawater). Only meaningful for the **floating** region — grounded ice is supported by bedrock, not buoyancy, so a large delta there is expected, not a data error.
