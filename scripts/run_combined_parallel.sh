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
export CONDA_PKGS_DIRS=$PROJECT/.conda/pkgs
export CONDA_ENVS_DIRS=$PROJECT/.conda/envs
export PIP_CACHE_DIR=$PROJECT/.cache/pip
export TMPDIR=$PROJECT/tmp
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" \
         "$CONDA_PKGS_DIRS" "$CONDA_ENVS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR" logs

# ── Activate environment ──
module load anaconda3 cuda
conda activate paperbanana        # adjust if your env name differs

# ── Credentials ──
export AWS_BEARER_TOKEN_BEDROCK="${AWS_BEARER_TOKEN_BEDROCK:?Set AWS_BEARER_TOKEN_BEDROCK}"
export AWS_BEDROCK_REGION="${AWS_BEDROCK_REGION:-us-east-1}"
export GLM_IMAGE_URL="http://localhost:30000"

# ── 1. Start GLM-Image server ──
echo "[$(date)] Starting GLM-Image server..."
sglang serve --model-path zai-org/GLM-Image --port 30000 &
GLM_PID=$!

# Wait for server readiness (up to 10 min)
for i in $(seq 1 60); do
    if curl -s http://localhost:30000/v1/models > /dev/null 2>&1; then
        echo "[$(date)] GLM-Image server ready after $((i * 10))s"
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "ERROR: GLM-Image server did not start within 10 minutes."
        kill $GLM_PID 2>/dev/null
        exit 1
    fi
    sleep 10
done

# ── 2. Run parallel debate pipeline ──
python main.py \
    --dataset_name PaperBananaBench \
    --task_name diagram \
    --split_name test \
    --exp_mode dev_parallel_debate \
    --retrieval_setting auto \
    --max_critic_rounds 3 \
    --main_model_name "bedrock/us.anthropic.claude-opus-4-6-v1" \
    --image_gen_model_name "glm-image" \
    --critic_b_model_name "bedrock/qwen.qwen3-vl-235b-a22b" \
    --resume

# ── 3. Cleanup ──
kill $GLM_PID 2>/dev/null
echo "[$(date)] Done."
