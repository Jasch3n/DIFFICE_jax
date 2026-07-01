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
#SBATCH --mail-user=jasperchen@stanford.edu

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

SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
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
        --mail-user=jasperchen@stanford.edu
    )
    if [ "${BACKEND}" = "gpu" ]; then
        SBATCH_ARGS+=(--constraint=GPU_SKU:A100_SXM4 --gres=gpu:1)
    fi
    exec sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_PATH}" "--${BACKEND}" "${CONFIG_PATH}" "$@"
fi

case "${BACKEND}" in
    gpu)
        module unload cuda || true
        source /oak/stanford/groups/cyaolai/JasperChen/VirtualEnv/DIFFICE_gpu_env/bin/activate
        ;;
    cpu)
        source /oak/stanford/groups/cyaolai/JasperChen/VirtualEnv/DIFFICE_cpu_env/bin/activate
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
