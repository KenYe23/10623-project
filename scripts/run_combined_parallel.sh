#!/bin/bash
#SBATCH --job-name=pb-parallel
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ── Storage redirects (critical: 25 GB home quota on PSC) ──
export HF_HOME=$PROJECT/.cache/huggingface
export TRANSFORMERS_CACHE=$PROJECT/.cache/huggingface/transformers
export HF_DATASETS_CACHE=$PROJECT/.cache/huggingface/datasets
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export CONDA_PKGS_DIRS=$PROJECT/.conda/pkgs
export CONDA_ENVS_DIRS=$PROJECT/.conda/envs
export PIP_CACHE_DIR=$PROJECT/.cache/pip
export TMPDIR=$PROJECT/tmp
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" \
         "$CONDA_PKGS_DIRS" "$CONDA_ENVS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR" logs

# ── Activate environment ──
module load anaconda3 cuda gcc/13.3.1-p20240614
conda activate paperbanana        # adjust if your env name differs

# Ensure large Hugging Face files can be fetched by accelerated downloader
python - <<'PY'
import importlib.util, sys
if importlib.util.find_spec("hf_transfer") is None:
    sys.stderr.write(
        "ERROR: hf_transfer is not installed in this environment.\n"
        "Run once on login node: pip install hf_transfer\n"
    )
    raise SystemExit(1)
PY

# ── Credentials ──
export AWS_BEARER_TOKEN_BEDROCK="${AWS_BEARER_TOKEN_BEDROCK:?Set AWS_BEARER_TOKEN_BEDROCK}"
export AWS_BEDROCK_REGION="${AWS_BEDROCK_REGION:-us-east-1}"
export FLUX2_SERVER_URL="http://localhost:30000"

# ── 1. Start FLUX2 server ──
echo "[$(date)] Starting FLUX2 server..."
FLUX_SERVER_ARGS="${FLUX_SERVER_ARGS:-}"
echo "[$(date)] FLUX_SERVER_ARGS='${FLUX_SERVER_ARGS}'"
if [[ "${FLUX_SERVER_ARGS}" == *"--no_cpu_offloading"* ]] && [[ "${ALLOW_NO_CPU_OFFLOADING:-0}" != "1" ]]; then
    echo "ERROR: --no_cpu_offloading is blocked by default to prevent H100 OOM during FLUX startup."
    echo "If you really want it, set ALLOW_NO_CPU_OFFLOADING=1 explicitly."
    exit 1
fi
python -u scripts/flux2_http_server.py --port 30000 ${FLUX_SERVER_ARGS} > logs/sglang_${SLURM_JOB_ID}.log 2>&1 &
FLUX_PID=$!

# Wait for server readiness (up to 40 min)
for i in $(seq 1 240); do
    if curl -sf http://localhost:30000/health > /dev/null 2>&1; then
        echo "[$(date)] FLUX2 server ready after $((i * 10))s"
        break
    fi

    if ! kill -0 "$FLUX_PID" 2>/dev/null; then
        echo "ERROR: FLUX2 server exited early."
        tail -200 logs/sglang_${SLURM_JOB_ID}.log || true
        exit 1
    fi

    if [ "$i" -eq 240 ]; then
        echo "ERROR: FLUX2 server did not start within 40 minutes."
        tail -200 logs/sglang_${SLURM_JOB_ID}.log || true
        kill $FLUX_PID 2>/dev/null
        exit 1
    fi
    sleep 10
done

# ── 2. Run parallel debate pipeline ──
SPLIT_NAME="${SPLIT_NAME:-test}"
MAX_CONCURRENT="${MAX_CONCURRENT:-10}"
MAIN_MODEL_NAME="${MAIN_MODEL_NAME:-bedrock/qwen.qwen3-vl-235b-a22b}"
CRITIC_B_MODEL_NAME="${CRITIC_B_MODEL_NAME:-bedrock/global.anthropic.claude-sonnet-4-6}"
IMAGE_GEN_MODEL_NAME="${IMAGE_GEN_MODEL_NAME:-flux2-dev}"

MAIN_ARGS=(
    --dataset_name PaperBananaBench
    --task_name diagram
    --split_name "$SPLIT_NAME"
    --exp_mode dev_parallel_debate
    --retrieval_setting auto
    --max_critic_rounds 3
    --max_concurrent "$MAX_CONCURRENT"
    --main_model_name "$MAIN_MODEL_NAME"
    --image_gen_model_name "$IMAGE_GEN_MODEL_NAME"
    --critic_b_model_name "$CRITIC_B_MODEL_NAME"
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

# ── 3. Cleanup ──
kill $FLUX_PID 2>/dev/null
echo "[$(date)] Done."
