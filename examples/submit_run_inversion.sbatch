#!/bin/bash
#SBATCH --job-name=DIFFICESolver           # create a short name for your job
#SBATCH --output=DIFFICESolver_%A.out      # stdout file
#SBATCH --error=DIFFICESolver_%A.out       # stderr file
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --partition=serc
#SBATCH --constraint=GPU_SKU:A100_SXM4
#SBATCH --mem-per-cpu=3G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=9:10:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
# No --mail-user: Slurm defaults to the submitting user, so this stays
# correct for whoever runs the job.

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: sbatch $0 <config> [run_inversion.py args...]" >&2
    echo "Example: sbatch $0 configs/xpinn_joint_flatbed_kfac_gpinn_RAD_100k_gpu.yaml --no-save" >&2
    exit 1
fi

# DIFFICE_REPO_ROOT lets this script be copied/launched from anywhere (e.g. a
# separate launcher folder that keeps its own configs) while still running
# run_inversion.py out of a shared DIFFICE_jax checkout. Falls back to
# auto-detection for the common case where this script lives inside the repo
# it's launching.
#
# Under sbatch, Slurm copies this script into a per-job spool directory and
# executes that copy, so ${BASH_SOURCE[0]} does not point at the real repo.
# $SLURM_SUBMIT_DIR (the cwd where `sbatch` was invoked) is unaffected by that
# copy, so prefer it whenever we're actually running as a Slurm job.
if [ -n "${DIFFICE_REPO_ROOT:-}" ]; then
    if [ ! -f "${DIFFICE_REPO_ROOT}/examples/run_inversion.py" ]; then
        echo "DIFFICE_REPO_ROOT=${DIFFICE_REPO_ROOT} has no examples/run_inversion.py" >&2
        exit 1
    fi
    REPO_ROOT="$(cd "${DIFFICE_REPO_ROOT}" && pwd)"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/examples/run_inversion.py" ]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/run_inversion.py" ]; then
    REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
else
    SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
    SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
SCRIPT_DIR="${REPO_ROOT}/examples"
CONFIG_ARG="$1"
shift

if [[ "${CONFIG_ARG}" = /* ]]; then
    CONFIG_PATH="${CONFIG_ARG}"
elif [ -f "${CONFIG_ARG}" ]; then
    CONFIG_PATH="$(realpath "${CONFIG_ARG}")"
elif [ -f "${SCRIPT_DIR}/${CONFIG_ARG}" ]; then
    CONFIG_PATH="$(realpath "${SCRIPT_DIR}/${CONFIG_ARG}")"
elif [ -f "${REPO_ROOT}/${CONFIG_ARG}" ]; then
    CONFIG_PATH="$(realpath "${REPO_ROOT}/${CONFIG_ARG}")"
else
    echo "Config not found: ${CONFIG_ARG}" >&2
    exit 1
fi

# DIFFICE_GPU_VENV lets other users point at their own virtualenv instead of
# the default below; the fallback is the original hardcoded path.
GPU_VENV="${DIFFICE_GPU_VENV:-/oak/stanford/groups/cyaolai/JasperChen/VirtualEnv/DIFFICE_gpu_env}"
if [ ! -f "${GPU_VENV}/bin/activate" ]; then
    echo "GPU virtualenv not found at ${GPU_VENV}/bin/activate. Set DIFFICE_GPU_VENV to your venv path." >&2
    exit 1
fi

module unload cuda || true
source "${GPU_VENV}/bin/activate"

unset JAX_PLATFORMS
unset JAX_PLATFORM_NAME

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

cd "${REPO_ROOT}"

echo "HOSTNAME=$(hostname)"
echo "PWD=$(pwd)"
echo "PYTHON=$(which python)"
echo "CONFIG_PATH=${CONFIG_PATH}"

python "${SCRIPT_DIR}/run_inversion.py" "${CONFIG_PATH}" "$@"
