#!/usr/bin/env python3
"""
sync_check.py -- verify an intern's DIFFICE_jax launcher + YAML configs are
still consistent with the current repo after a pull/sync.

This is the harness for the `sync-intern-diffice-scripts` skill. It replays,
programmatically, the checks done by hand in the session that produced this
skill:

  * launcher entry point still resolves (examples/run_inversion.py) and the
    referenced virtualenv exists;
  * every intern config parses as YAML and uses a valid workflow/model pair;
  * each config's data.source resolves to a file that exists on disk;
  * sampling_counts are legal against the ACTUAL rebuilt dataset:
      - velocity/thickness/surface are drawn WITHOUT replacement, so each
        per-region count must be <= that region's finite-point coverage
        (over-count => the run crashes);
      - collocation is drawn WITH replacement, so exceeding the per-region
        library only forces duplicate points (a WARN, not a crash);
      - matching is drawn WITHOUT replacement, so it must be <= the number
        of interface (x_md/y_md) points;
  * library-size figures quoted in config comments still match the measured
    collocation library (they go stale when the dataset is rebuilt).

It deliberately does NOT `import diffice_jax` -- that pulls in JAX and hangs
on a login node. The config-schema logic is reproduced with PyYAML +
scipy.io.loadmat only (both live in the DIFFICE GPU venv).

Exit code: 0 if no FAIL-level findings, 1 otherwise. WARNs never fail.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

# --- repo-root default: this file lives at repo/.claude/skills/<skill>/sync_check.py
DEFAULT_REPO = Path(__file__).resolve().parents[3]
# --- the intern this skill was written for; override with --intern-configs / --launcher
DEFAULT_INTERN_CONFIGS = Path("/oak/stanford/groups/cyaolai/OscarVarodayan/DIFFICE_configs")
DEFAULT_LAUNCHER = Path("/oak/stanford/groups/cyaolai/OscarVarodayan/submit_run_inversion.sbatch")

VALID_WORKFLOWS = {"ice-shelf-only", "joint-inversion", "joint-inversion-regression"}

FAILS: list[str] = []
WARNS: list[str] = []
OKS: list[str] = []


def ok(msg):
    OKS.append(msg)
    print(f"  \033[32mOK\033[0m   {msg}")


def warn(msg):
    WARNS.append(msg)
    print(f"  \033[33mWARN\033[0m {msg}")


def fail(msg):
    FAILS.append(msg)
    print(f"  \033[31mFAIL\033[0m {msg}")


# --- dataset measurement (mirrors run_inversion.py's finite-per-region logic) ---
def _cells(d, key):
    v = d[key]
    if isinstance(v, np.ndarray) and v.dtype == object:
        return [np.asarray(x).reshape(-1) for x in v.reshape(-1)]
    return [np.asarray(v).reshape(-1)]


def _finite_per_region(d, *keys):
    if not all(k in d for k in keys):
        return None
    arrs = [_cells(d, k) for k in keys]
    out = []
    for parts in zip(*arrs):
        m = np.ones(parts[0].shape, bool)
        for p in parts:
            m &= np.isfinite(p)
        out.append(int(m.sum()))
    return out


def measure(mat_path: Path) -> dict:
    d = loadmat(str(mat_path))
    return {
        "velocity": _finite_per_region(d, "ud", "vd"),
        "thickness": _finite_per_region(d, "hd"),
        "surface": _finite_per_region(d, "sd"),
        "col_library": _finite_per_region(d, "xcol", "ycol"),
        "calving": _finite_per_region(d, "xct", "yct"),
        "interface": _finite_per_region(d, "x_md", "y_md"),
    }


# --- helpers ---
def _canonical_workflow(w):
    return None if w is None else str(w).replace("_", "-")


def _per_region(value, n_regions):
    """Normalise a sampling count to a per-region list (or None)."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    return [int(value)] * n_regions


def _check_no_replacement(name, counts, cover, findings_prefix):
    """velocity/thickness/surface/matching: count must be <= coverage/region."""
    if counts is None or cover is None:
        return
    for i, c in enumerate(counts):
        cap = cover[i] if i < len(cover) else cover[-1]
        if c > cap:
            fail(f"{findings_prefix}: {name}[{i}]={c} exceeds region coverage {cap} "
                 f"(sampled without replacement -> run will crash)")
        else:
            ok(f"{findings_prefix}: {name}[{i}]={c} <= coverage {cap}")


def check_launcher(launcher: Path, repo: Path):
    print(f"\n== launcher: {launcher} ==")
    if not launcher.is_file():
        fail(f"launcher not found at {launcher}")
        return
    text = launcher.read_text()

    # entry point: the python <...>/run_inversion.py invocation
    m = re.search(r'run_inversion\.py', text)
    if not m:
        warn("launcher does not reference run_inversion.py -- unusual, inspect by hand")
    entry = repo / "examples" / "run_inversion.py"
    if entry.is_file():
        ok(f"repo entry point exists: {entry}")
    else:
        fail(f"repo entry point missing: {entry}")

    # virtualenv activate referenced by the launcher
    for m in re.finditer(r'(/[^\s"\']+/bin/activate)', text):
        venv = Path(m.group(1))
        if venv.is_file():
            ok(f"venv activate exists: {venv}")
        else:
            warn(f"venv activate referenced but missing: {venv} "
                 "(may be overridable via DIFFICE_GPU_VENV)")


def check_config(cfg_path: Path):
    import yaml

    print(f"\n== config: {cfg_path.name} ==")
    try:
        raw = yaml.safe_load(cfg_path.read_text()) or {}
    except Exception as exc:  # noqa: BLE001
        fail(f"YAML parse error: {exc}")
        return
    text = cfg_path.read_text()

    # workflow / model pairing (mirrors runner._validate_workflow)
    public = _canonical_workflow(raw.get("workflow"))
    model_wf = (raw.get("model") or {}).get("workflow")
    if public is not None:
        if public not in VALID_WORKFLOWS:
            fail(f"unknown workflow {raw.get('workflow')!r} (valid: {sorted(VALID_WORKFLOWS)})")
        else:
            expected = "pinn" if public == "ice-shelf-only" else "xpinn"
            if model_wf != expected:
                fail(f"workflow {public!r} expects model.workflow={expected!r}, got {model_wf!r}")
            else:
                ok(f"workflow/model pair valid ({public} / {model_wf})")

    # data.source resolution
    data = raw.get("data") or {}
    src = data.get("source")
    if not src:
        fail("no data.source in config")
        return
    src_path = Path(src)
    if not src_path.is_absolute():
        src_path = (cfg_path.parent / src_path).resolve()
    if not src_path.is_file():
        fail(f"data.source does not exist: {src_path}")
        return
    ok(f"data.source exists: {src_path}")

    # sampling-count legality against the actual dataset
    regions = (raw.get("model") or {}).get("regions") or []
    n_regions = max(len(regions), 1)
    sc = data.get("sampling_counts") or {}
    try:
        m = measure(src_path)
    except Exception as exc:  # noqa: BLE001
        warn(f"could not measure dataset ({exc}); skipping sampling-count checks")
        m = None

    if m is not None:
        _check_no_replacement("velocity_data", _per_region(sc.get("velocity_data"), n_regions),
                              m["velocity"], cfg_path.name)
        _check_no_replacement("thickness_data", _per_region(sc.get("thickness_data"), n_regions),
                              m["thickness"], cfg_path.name)
        surf = sc.get("surface_data")
        if surf is not None:
            _check_no_replacement("surface_data", _per_region(surf, n_regions),
                                  m["surface"], cfg_path.name)

        # collocation: with replacement -> exceeding library only duplicates
        col = _per_region(sc.get("collocation"), n_regions)
        lib = m["col_library"]
        if col is not None and lib is not None:
            for i, c in enumerate(col):
                cap = lib[i] if i < len(lib) else lib[-1]
                if c > cap:
                    warn(f"{cfg_path.name}: collocation[{i}]={c} > library {cap} "
                         "(replace=True -> oversampled with duplicates, not a crash)")
                else:
                    ok(f"{cfg_path.name}: collocation[{i}]={c} <= library {cap}")

        # matching: without replacement -> <= interface points
        matching = sc.get("matching")
        if matching is not None and m["interface"] is not None:
            if int(matching) > m["interface"][0]:
                fail(f"{cfg_path.name}: matching={matching} exceeds interface points "
                     f"{m['interface'][0]} (sampled without replacement)")
            else:
                ok(f"{cfg_path.name}: matching={matching} <= interface {m['interface'][0]}")

        # stale library-size comment detection
        if lib is not None:
            for line in text.splitlines():
                if "librar" not in line.lower():
                    continue
                nums = re.findall(r'~?\s*(\d[\d,]*)', line)
                nums = [int(n.replace(",", "")) for n in nums if len(n) >= 3]
                if len(nums) >= 2:
                    quoted = nums[:2]
                    measured = lib[:2]
                    off = any(abs(q - mm) > max(0.10 * mm, 50) for q, mm in zip(quoted, measured))
                    if off:
                        warn(f"{cfg_path.name}: comment cites library ~{quoted[0]}/~{quoted[1]} "
                             f"but measured {measured[0]}/{measured[1]} -- update the comment")
                    else:
                        ok(f"{cfg_path.name}: library-size comment matches measured {measured}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=DEFAULT_REPO, help="DIFFICE_jax repo root")
    ap.add_argument("--intern-configs", type=Path, default=DEFAULT_INTERN_CONFIGS,
                    help="directory holding the intern's YAML configs")
    ap.add_argument("--launcher", type=Path, default=DEFAULT_LAUNCHER,
                    help="the intern's sbatch launcher script")
    args = ap.parse_args()

    print(f"repo           : {args.repo}")
    print(f"intern configs : {args.intern_configs}")
    print(f"launcher       : {args.launcher}")

    check_launcher(args.launcher, args.repo)

    cfgs = sorted(p for p in args.intern_configs.glob("*.y*ml"))
    if not cfgs:
        warn(f"no YAML configs found in {args.intern_configs}")
    for cfg in cfgs:
        check_config(cfg)

    print("\n==================== SUMMARY ====================")
    print(f"  OK:   {len(OKS)}")
    print(f"  WARN: {len(WARNS)}")
    print(f"  FAIL: {len(FAILS)}")
    for w in WARNS:
        print(f"  \033[33m- WARN\033[0m {w}")
    for f in FAILS:
        print(f"  \033[31m- FAIL\033[0m {f}")
    if FAILS:
        print("\nRESULT: inconsistencies found -- see FAIL lines above.")
        sys.exit(1)
    print("\nRESULT: intern scripts/configs are consistent with the repo"
          + (" (warnings above are non-blocking)." if WARNS else "."))


if __name__ == "__main__":
    main()
