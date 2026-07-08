# Surface elevation keeps its own coordinates, decoupled from thickness

Surface elevation (`sd`/`s_dense`) is sourced as its own data kind with its own native x/y per provider, rather than resampled onto the sparse-thickness grid (`xd_h`/`yd_h`) at `.mat`-generation time. We chose this even after confirming DIFFICE_jax's actual training code (not just its docs) hard-codes surface/thickness co-location — `data/pinns/preprocessing.py` reindexes surface by thickness's own NaN mask, `model/pinns/loss.py` predicts surface only at thickness's sample coordinates, and `workflow/runner.py` asserts the two sample counts match. That means today's `joint_xpinn_data` output with independent surface coordinates is **not yet consumable by DIFFICE_jax as-is**.

## Considered Options

- **Resample surface onto `xd_h`/`yd_h` at assembly time** (nearest-neighbor, same mechanism already used for `h_dense`/`s_dense`) — works with DIFFICE_jax unmodified today, but treats "surface must live on thickness's grid" as a data-generation-side constraint baked permanently into `joint_xpinn_data`, even when a source (e.g. BedMachine surface + BEDMAP1 thickness) has no real reason to share thickness's exact sparse points.
- **Independent coordinates (chosen)** — keeps `joint_xpinn_data`'s data model honest (surface really is an independently-observed quantity), at the cost of deferring DIFFICE_jax compatibility to a separate future task in that project.

## Consequences

- The `.mat` schema this pipeline emits now diverges from DIFFICE_jax's current fixed field layout (see root `CLAUDE.md`) for the surface-elevation fields specifically — anyone trying to train on this output today needs the DIFFICE_jax-side changes first (see the exploration notes: `data/pinns/preprocessing.py`, `model/pinns/loss.py`, `workflow/runner.py` — roughly 5 files, 150-300 lines, touches NaN masking, normalization, and the loss function's fixed output-channel assignment).
- Until that DIFFICE_jax work happens, this pipeline's output is for consistency-checking and development use, not actual training runs, for any config relying on non-co-located surface/thickness sources.
- New fields `xd_s`/`yd_s` are added to `EXPECTED_SAVE_VARIABLES` (each region's cell dict) to hold `sd`'s own coordinates, following the existing `xd_h`/`yd_h` naming convention. This does not affect `h_dense`/`s_dense`, which keep their existing behavior of being resampled onto the velocity grid (`xd`/`yd`) — only the sparse role (`sd`) was affected by this decision.
