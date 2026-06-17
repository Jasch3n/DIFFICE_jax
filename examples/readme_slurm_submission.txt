Slurm submission script for run_inversion.py
===========================================

Script location:
examples/submit_run_inversion.sbatch

Purpose:
Submit a Sherlock GPU job that runs examples/run_inversion.py with any specified config.

Usage:
sbatch examples/submit_run_inversion.sbatch <config> [run_inversion.py args...]

Examples:
sbatch examples/submit_run_inversion.sbatch configs/xpinn_joint_flatbed_kfac_gpinn_RAD_100k_gpu.yaml
sbatch examples/submit_run_inversion.sbatch configs/xpinn_joint_flatbed_kfac_gpinn_RAD_100_gpu_smoke.yaml --no-save

Config path handling:
- Absolute config paths are accepted.
- Relative config paths can be given relative to the repo root or the examples folder.

What the script does:
- unloads the CUDA module
- activates /oak/stanford/groups/cyaolai/JasperChen/VirtualEnv/DIFFICE_gpu_env
- unsets inherited JAX platform selectors
- sets OMP_NUM_THREADS from SLURM_CPUS_PER_TASK
- runs examples/run_inversion.py from the repo root

Notes:
- The first argument must be the config path.
- Any extra arguments after the config are passed directly to run_inversion.py.
