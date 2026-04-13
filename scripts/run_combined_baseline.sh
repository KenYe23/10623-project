#!/bin/bash
#SBATCH --job-name=pb-baseline
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --time=03:00:00
#SBATCH --output=/ocean/projects/cis240137p/eshen3/PaperBanana/logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis240137p/eshen3/PaperBanana/logs/%x_%j.err

# ── Storage redirects (critical: 25 GB home quota on PSC) ──
export HF_HOME=$PROJECT/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export XDG_CACHE_HOME=$PROJECT/.cache
export HF_HUB_DISABLE_XET=1

# ── Activate environment ──
source /ocean/projects/cis240137p/eshen3/anaconda3/etc/profile.d/conda.sh
conda activate paperbanana
module load cuda
module load gcc/13.3.1-p20240614

# Export compilers for SGLang CUDA kernel compilation
export CC=$(which gcc)
export CXX=$(which g++)
export CUDAHOSTCXX=$CXX

# ── Credentials ──
export AWS_BEARER_TOKEN_BEDROCK="${AWS_BEARER_TOKEN_BEDROCK:?Set AWS_BEARER_TOKEN_BEDROCK}"
export AWS_BEDROCK_REGION="${AWS_BEDROCK_REGION:-us-east-1}"
export FLUX2_SERVER_URL="http://localhost:30000"

# ── Cleanup on exit ──
cleanup() {
  if [[ -n "${FLUX_PID:-}" ]]; then
    kill "$FLUX_PID" 2>/dev/null || true
    wait "$FLUX_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# ── 1. Start FLUX2 server ──
echo "[$(date)] Starting FLUX2 server..."
python scripts/flux2_http_server.py --port 30000 > /ocean/projects/cis240137p/eshen3/PaperBanana/logs/sglang_${SLURM_JOB_ID}.log 2>&1 &
FLUX_PID=$!

# Wait for server readiness (up to 30 min)
for i in $(seq 1 150); do
    if curl -sf http://127.0.0.1:30000/v1/models > /dev/null 2>&1; then
        echo "[$(date)] FLUX2 server ready after $((i * 10))s"
        READY=1
        break
    fi

    if ! kill -0 "$FLUX_PID" 2>/dev/null; then
        echo "ERROR: FLUX2 server exited early."
        tail -200 /ocean/projects/cis240137p/eshen3/PaperBanana/logs/sglang_${SLURM_JOB_ID}.log || true
        exit 1
    fi

    sleep 10
done

if [[ "$READY" -ne 1 ]]; then
    echo "ERROR: FLUX2 server did not start within 25 minutes."
    tail -200 /ocean/projects/cis240137p/eshen3/PaperBanana/logs/sglang_${SLURM_JOB_ID}.log || true
    exit 1
fi

# ── 2. Run baseline pipeline (single-critic, dev_full) ──
SPLIT_NAME="${SPLIT_NAME:-test_mini100}"
MAX_CONCURRENT="${MAX_CONCURRENT:-10}"

MAIN_ARGS=(
    --dataset_name PaperBananaBench
    --task_name diagram
    --split_name "$SPLIT_NAME"
    --exp_mode dev_full
    --retrieval_setting auto
    --max_critic_rounds 3
    --max_concurrent "$MAX_CONCURRENT"
    --main_model_name "bedrock/global.anthropic.claude-sonnet-4-6"
    --image_gen_model_name "flux2-dev"
    --resume
)

# Optional shard slicing by index range
if [[ -n "${START_IDX:-}" ]]; then
    MAIN_ARGS+=(--start_idx "$START_IDX")
fi
if [[ -n "${END_IDX:-}" ]]; then
    MAIN_ARGS+=(--end_idx "$END_IDX")
fi

# Optional hard sample cap for smoke tests
if [[ -n "${MAX_SAMPLES:-}" ]]; then
    MAIN_ARGS+=(--max_samples "$MAX_SAMPLES")
fi

python main.py "${MAIN_ARGS[@]}"

echo "[$(date)] Done."
