# Repository Guidelines

## Project Structure & Module Organization
`diffice_jax/` contains the installable package. Core logic is split by concern: `data/` for preprocessing and sampling, `equation/` for PDE terms, `model/` for PINNs and XPINNs networks/losses/prediction, and `optimizer/` for Adam and L-BFGS routines. `tests/` holds unit and regression inputs, including `.mat` fixtures. `examples/` and `tutorial/` contain runnable scripts, notebooks, and sample data for synthetic and real ice-shelf workflows. `docs/` is a Sphinx site with source files in `docs/source/`.

## Build, Test, and Development Commands
Use the environment activated by `pyenv activate Metal-Env` to test and run scripts by default. Whenever KFAC is used as the optimizer, use `pyenv activate Cpu-Diffice-Env` instead.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, module-level functions, and concise inline comments only where the math or control flow is non-obvious. Use `snake_case` for functions and variables, keep public API names aligned with the current exports in `diffice_jax/__init__.py`, and mirror the existing PINNs/XPINNs directory split when adding new functionality. No formatter or linter is configured here, so keep changes stylistically consistent with surrounding files.

Write code as concise as possible with as little safeguarding as possible, assuming perfect knowledge from the user of what the code expects. Enforce typing consistent with JAX where appropriate. Code in a functional paradigm that's consistent with JAX's design.

## Testing Guidelines
Write tests with `pytest` and place them in `tests/` as `test_*.py`. Prefer small deterministic checks like the current initialization, normalization, and network-shape tests in `tests/test_func.py`. When adding data-dependent behavior, reuse or extend the existing `.mat` fixtures instead of introducing large new assets unless they are necessary for regression coverage.

## Commit & Pull Request Guidelines
No need to consider commit or pull requests. Ignore the any aspect of git.

## Repository-Specific Notes
Some of the current tests in the `tests` folder are outdated due to the addition of capabilities to support grounded ice domains. When implementing new features, therefore, do not retroactively run old tests as they are outdated. 
