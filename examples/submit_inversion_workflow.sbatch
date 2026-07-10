#!/bin/bash
#SBATCH --job-name=DIFFICEInversion
#SBATCH --output=DIFFICEInversion_%A.out
#SBATCH --error=DIFFICEInversion_%A.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=serc
#SBATCH --mem-per-cpu=3G
#SBATCH --time=23:10:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
# No --mail-user: Slurm defaults to the submitting user, so this stays
# correct for whoever runs the job.

set -euo pipefail

usage() {
    cat >&2 <<EOF
Usage:
  $0 [--gpu|--cpu] [--sbatch] <config> [run_inversion.py args...]

Examples:
  $0 --gpu synthetic_data/configs/pinn_synthetic_ice_shelf_gpu.yaml --no-save
  $0 --cpu synthetic_data/configs/pinn_synthetic_ice_shelf.yaml --no-save
  $0 --gpu --sbatch synthetic_data/configs/pinn_synthetic_ice_shelf_gpu.yaml

Without --sbatch, the workflow runs immediately on this machine. With --sbatch,
the script submits a batch job with resources matching --gpu or --cpu.
EOF
}

BACKEND="gpu"
SUBMIT_BATCH="false"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --gpu)
            BACKEND="gpu"
            shift
            ;;
        --cpu)
            BACKEND="cpu"
            shift
            ;;
        --sbatch)
            SUBMIT_BATCH="true"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        --*)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

if [ "$#" -lt 1 ]; then
    usage
    exit 1
fi

# SCRIPT_PATH is this file's own location, used below to resubmit itself via
# --sbatch. That resubmission always happens from a direct (non-sbatch) shell
# invocation, so ${BASH_SOURCE[0]} is reliable here regardless of where this
# launcher script lives (e.g. a separate folder outside the DIFFICE_jax repo).
SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"

# DIFFICE_REPO_ROOT lets this launcher live anywhere (e.g. a personal folder
# with its own configs) while still running run_inversion.py out of a shared
# DIFFICE_jax checkout. Falls back to auto-detection for the common case where
# this script lives inside the repo it's launching.
#
# Once running as the actual Slurm job (i.e. past the --sbatch resubmission
# below), Slurm has copied this script into a per-job spool directory, so
# ${BASH_SOURCE[0]} no longer points at the real repo; $SLURM_SUBMIT_DIR (the
# cwd where `sbatch` was invoked) is unaffected by that copy.
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
    REPO_ROOT="$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)"
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

if [ "${SUBMIT_BATCH}" = "true" ]; then
    SBATCH_ARGS=(
        --job-name="DIFFICE-${BACKEND}"
        --output="DIFFICEInversion_%A.out"
        --error="DIFFICEInversion_%A.out"
        --nodes=1
        --ntasks=1
        --cpus-per-task=8
        --partition=serc
        --mem-per-cpu=3G
        --time=23:10:00
        --mail-type=begin
        --mail-type=end
    )
    if [ "${BACKEND}" = "gpu" ]; then
        SBATCH_ARGS+=(--constraint=GPU_SKU:A100_SXM4 --gres=gpu:1)
    fi
    exec sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_PATH}" "--${BACKEND}" "${CONFIG_PATH}" "$@"
fi

# DIFFICE_GPU_VENV / DIFFICE_CPU_VENV let other users point at their own
# virtualenvs instead of the defaults below; the fallbacks are the original
# hardcoded paths.
case "${BACKEND}" in
    gpu)
        GPU_VENV="${DIFFICE_GPU_VENV:-/oak/stanford/groups/cyaolai/JasperChen/VirtualEnv/DIFFICE_gpu_env}"
        if [ ! -f "${GPU_VENV}/bin/activate" ]; then
            echo "GPU virtualenv not found at ${GPU_VENV}/bin/activate. Set DIFFICE_GPU_VENV to your venv path." >&2
            exit 1
        fi
        module unload cuda || true
        source "${GPU_VENV}/bin/activate"
        ;;
    cpu)
        CPU_VENV="${DIFFICE_CPU_VENV:-/oak/stanford/groups/cyaolai/JasperChen/VirtualEnv/DIFFICE_cpu_env}"
        if [ ! -f "${CPU_VENV}/bin/activate" ]; then
            echo "CPU virtualenv not found at ${CPU_VENV}/bin/activate. Set DIFFICE_CPU_VENV to your venv path." >&2
            exit 1
        fi
        source "${CPU_VENV}/bin/activate"
        ;;
    *)
        echo "Unknown backend: ${BACKEND}" >&2
        exit 1
        ;;
esac

unset JAX_PLATFORMS
unset JAX_PLATFORM_NAME

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

cd "${REPO_ROOT}"

echo "HOSTNAME=$(hostname)"
echo "PWD=$(pwd)"
echo "BACKEND=${BACKEND}"
echo "PYTHON=$(which python)"
echo "CONFIG_PATH=${CONFIG_PATH}"

python "${SCRIPT_DIR}/run_inversion.py" "${CONFIG_PATH}" "$@"
