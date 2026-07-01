Slurm entry point for run_inversion.py
======================================

Script location:
examples/submit_inversion_workflow.sbatch

Purpose:
Run or submit a Sherlock job that runs examples/run_inversion.py with any specified config. GPU is the default backend; pass --cpu for CPU runs.

Usage:
examples/submit_inversion_workflow.sbatch [--gpu|--cpu] [--sbatch] <config> [run_inversion.py args...]

Examples:
examples/submit_inversion_workflow.sbatch --gpu synthetic_data/configs/pinn_synthetic_ice_shelf_gpu.yaml
examples/submit_inversion_workflow.sbatch --cpu synthetic_data/configs/pinn_synthetic_ice_shelf.yaml --no-save
examples/submit_inversion_workflow.sbatch --gpu --sbatch synthetic_data/configs/pinn_synthetic_ice_shelf_gpu.yaml

Config path handling:
- Absolute config paths are accepted.
- Relative config paths can be given relative to the repo root or the examples folder.

What the script does:
- with --sbatch, submits itself with matching SLURM resources
- without --sbatch, runs immediately on the current machine
- activates /oak/stanford/groups/cyaolai/JasperChen/VirtualEnv/DIFFICE_gpu_env for --gpu
- activates /oak/stanford/groups/cyaolai/JasperChen/VirtualEnv/DIFFICE_cpu_env for --cpu
- unsets inherited JAX platform selectors
- sets OMP_NUM_THREADS from SLURM_CPUS_PER_TASK
- runs examples/run_inversion.py from the repo root

Notes:
- Run the script directly instead of invoking sbatch yourself; this lets --sbatch, --gpu, and --cpu choose the correct scheduler resources before submission.
- Use --sbatch for a batch job. Omit --sbatch for immediate validation on the current machine or allocation.
- If no backend flag is provided, the script defaults to --gpu.
- Any extra arguments after the config are passed directly to run_inversion.py.
