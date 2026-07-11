---
name: sync-intern-diffice-scripts
description: >-
  Sync/verify an intern's DIFFICE_jax launcher and YAML configs against the
  current repo after a pull or dataset rebuild. Use when asked to check that
  the intern's (e.g. Oscar's) sbatch script and configs are still consistent
  with the repo, that their data paths still resolve, that sampling counts are
  legal against the rebuilt datasets, or that config comments aren't stale.
---

# sync-intern-diffice-scripts

An intern keeps their own launcher (`submit_run_inversion.sbatch`) and a folder
of YAML configs *outside* this repo, but those configs point at repo data files
and are run through the repo's `examples/run_inversion.py`. When the repo is
pulled/synced — output buckets get restructured, datasets get rebuilt, configs
get new fields — the intern's copies drift and silently break (missing `.mat`,
over-count sampling, stale comments).

This skill's harness is **`.claude/skills/sync-intern-diffice-scripts/sync_check.py`**
(paths below are relative to the repo root). It replays, programmatically, the
manual consistency review: launcher entry point, YAML/workflow schema, data-path
resolution, sampling-count legality against the *actual* rebuilt datasets, and
stale library-size comments. It never imports `diffice_jax` (that pulls in JAX
and hangs on a login node) — it uses PyYAML + `scipy.io.loadmat` only.

## Prerequisites

The DIFFICE GPU venv already has PyYAML + SciPy — just activate it. No
`apt-get`/`pip` needed. (Override the venv path if the intern uses a different
one.)

```bash
source /oak/stanford/groups/cyaolai/JasperChen/VirtualEnv/DIFFICE_gpu_env/bin/activate
```

## Run (agent path)

From the repo root, run the driver. With no arguments it defaults to this repo
and to the intern this skill was written for (Oscar):

```bash
python .claude/skills/sync-intern-diffice-scripts/sync_check.py
```

Point it at a different intern / launcher explicitly:

```bash
python .claude/skills/sync-intern-diffice-scripts/sync_check.py \
  --intern-configs /oak/stanford/groups/cyaolai/OscarVarodayan/DIFFICE_configs \
  --launcher /oak/stanford/groups/cyaolai/OscarVarodayan/submit_run_inversion.sbatch
```

`--repo` defaults to the repo this skill lives in. `python .claude/skills/sync-intern-diffice-scripts/sync_check.py --help` lists all flags.

**Reading the output.** Each check prints `OK` / `WARN` / `FAIL`, then a summary.
Exit code is `0` unless there is at least one `FAIL`. Real output from a clean
run (warnings only) looks like:

```
== config: xpinn_joint_AmeryLambert_kfac_RAD_50k_gpu.yaml ==
  OK   workflow/model pair valid (joint-inversion / xpinn)
  OK   data.source exists: .../sparse_bedmap/Amery_Lambert_100km/Amery_Lambert_100km.mat
  OK   ...: velocity_data[0]=1028 <= coverage 8631
  OK   ...: collocation[1]=1500 <= library 105335
  OK   ...: matching=500 <= interface 632
  WARN ...: comment cites library ~3047/~27967 but measured 8633/105335 -- update the comment
==================== SUMMARY ====================
  OK:   28
  WARN: 2
  FAIL: 0
RESULT: intern scripts/configs are consistent with the repo (warnings above are non-blocking).
```

## What each finding means, and how to fix it

- **FAIL `data.source does not exist`** — the repo moved/renamed the dataset
  (e.g. the `output/` → `output/{sparse_bedmap,dense_bedmachine}/` restructure).
  Fix: edit the config's `data.source` to the new path. The pre-split single
  build maps to `sparse_bedmap`.
- **FAIL `velocity/thickness/surface[i] exceeds region coverage`** — these are
  sampled **without replacement**, so an over-count crashes the run. Fix: lower
  that count to `<=` the reported coverage.
- **FAIL `matching exceeds interface points`** — same, for interface-matching
  points (`x_md`/`y_md`). Lower `matching`.
- **FAIL `workflow ... expects model.workflow=...`** or `unknown workflow` — the
  config's `workflow`/`model.workflow` pair is invalid. Valid public workflows:
  `ice-shelf-only` (→ `pinn`), `joint-inversion` / `joint-inversion-regression`
  (→ `xpinn`). Underscores are fine (`joint_inversion` is normalized).
- **WARN `collocation[i] > library`** — collocation is sampled **with
  replacement**, so this only oversamples with duplicates; it does not crash.
  Only a problem if the intern wanted that many *unique* points (then the
  dataset needs a denser collocation library — a data-build change, not a
  config edit).
- **WARN `comment cites library ~X/~Y but measured G/F`** — a dataset rebuild
  changed the collocation-library size; the config's explanatory comment is now
  stale. Fix: update the `~X / ~Y` figures in the comment to the measured ones.
- **FAIL/WARN on the launcher** — the repo entry point (`examples/run_inversion.py`)
  or the referenced venv is missing. The `run_inversion.py` interface is stable;
  if it's gone, the repo layout changed and the launcher's hardcoded
  `SCRIPT_DIR` needs updating.

After editing the intern's files, **re-run the driver** until only WARNs (or
nothing) remain.

## Gotchas (learned the hard way)

- **Never `import diffice_jax` to validate a config on the login node.** The
  package import pulls in JAX and blocks for minutes (killed at timeout). The
  driver reproduces the schema/sampling logic with PyYAML + SciPy instead. Keep
  it that way.
- **Datasets are rebuilt *in place* — paths stay, numbers move.** A pull can
  regenerate `Amery_Lambert_100km.mat` with a different collocation-library
  density without changing its path. Coverage/library counts (and therefore the
  correctness of comments and of any near-library sampling) drift silently. This
  is the single most common source of "it worked yesterday." Re-run this check
  after every pull *and* after any local data rebuild — in the session that
  produced this skill the library went `1016/9323 → 3047/27967 → 8633/105335`
  across successive rebuilds.
- **With- vs without-replacement is the whole ballgame.** velocity/thickness/
  surface/matching are without replacement (over-count ⇒ crash ⇒ FAIL);
  collocation is with replacement (over-count ⇒ duplicates ⇒ WARN). The driver
  encodes this distinction; don't "simplify" it into one rule.
- **Grounded calving-front library is legitimately 0** (grounded ice has no
  calving front). The solver fills zero-placeholders for empty boundary data, so
  a non-zero `calving_front` grounded count is *not* a failure — the driver
  doesn't flag it.
- **`interface_collocation` is not inert even with gPINN off.** Those points are
  appended to the ordinary collocation set every step; only the extra
  gradient-residual term is gated on `use_gpinn`. (Not checked by the driver, but
  relevant when reasoning about a config's effective collocation count.)
- **This is a login-node-safe check** (a `loadmat` of a ~15 MB `.mat` is sub-second
  I/O). Do **not** escalate it into launching training here — that belongs in an
  sbatch GPU job.

## Troubleshooting

- `ModuleNotFoundError: No module named 'yaml'` / `'scipy'` → wrong Python; you
  didn't `source` the DIFFICE GPU venv (see Prerequisites).
- Driver hangs for minutes → you (or an edit) reintroduced an `import diffice_jax`.
  Remove it; use PyYAML/SciPy only.
- `venv activate referenced but missing` WARN → the intern's launcher hardcodes a
  venv path that isn't present; they can override with `DIFFICE_GPU_VENV`, or the
  path needs fixing.
