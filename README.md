# Faithful Diagram Generation via Multi-Agent VLM Consensus

> **10-623 Course Project** — Extends [PaperBanana](https://github.com/dwzhu-pku/PaperBanana) with a *Parallel Debate* architecture: two diverse VLM critics independently evaluate generated diagrams, then a synthesizer reconciles their feedback into a unified refinement prompt. This targets the "fine-grained faithfulness gap" (missed spatial errors, hallucinated connections) identified in the original paper.

## Architecture

```
Retriever → Planner → Stylist → Visualizer ─┬─→ Critic A (Claude Opus 4.6)  ─→ Synthesizer (Claude Opus 4.6) → Visualizer → …
                                             └─→ Critic B (Qwen3-VL-235B)   ─↗
```

| Role | Model | Provider |
|------|-------|----------|
| Planner, Stylist, Synthesizer, Critic A | Claude Opus 4.6 | AWS Bedrock |
| Critic B | Qwen3-VL-235B-A22B | AWS Bedrock |
| Visualizer (image gen) | GLM-Image | Self-hosted (SGLang on H100) |

**Key files:**
- `agents/parallel_critic_agent.py` — Runs two critics in parallel via `asyncio.gather`
- `agents/synthesizer_agent.py` — Reconciles two critiques into unified refinement
- `agents/critic_agent.py` — Single critic (supports `output_prefix` for namespaced keys)
- `utils/paperviz_processor.py` — `dev_parallel_debate` pipeline mode

---

## Prerequisites

- **PSC Bridges-2 account** with GPU allocation (H100-80GB)
- **AWS Bedrock access** with an API key (ABSK bearer token) for Claude Opus 4.6 and Qwen3-VL-235B in `us-east-1`
- Python 3.12

---

## Setup

### 1. PSC Storage Configuration

Run this once on the PSC login node to automatically set up storage redirects every time you log in:

```bash
cat << 'EOF' >> ~/.bashrc

# Storage Redirects for PaperBanana (25GB Quota Fix)
export HF_HOME=$PROJECT/.cache/huggingface
export TRANSFORMERS_CACHE=$PROJECT/.cache/huggingface/transformers
export HF_DATASETS_CACHE=$PROJECT/.cache/huggingface/datasets
export CONDA_PKGS_DIRS=$PROJECT/.conda/pkgs
export CONDA_ENVS_DIRS=$PROJECT/.conda/envs
export PIP_CACHE_DIR=$PROJECT/.cache/pip
export TMPDIR=$PROJECT/tmp

EOF

source ~/.bashrc
```

### 2. Clone Repositories

```bash
cd $PROJECT
git clone https://github.com/KenYe23/10623-project.git PaperBanana
```

### 3. Create Environment

```bash
module load anaconda3 cuda gcc/13.3.1-p20240614
conda create -n paperbanana python=3.12 -y
conda activate paperbanana

# Install GCC 13 C++ standard libraries into Conda to prevent SGLang ABI crashes
conda install -c conda-forge libstdcxx-ng -y

# Export compilers for SGLang CUDA kernel compilation
export CC=$(which gcc)
export CXX=$(which g++)
export CUDAHOSTCXX=$CXX

cd PaperBanana
pip install -r requirements.txt
pip install httpx   # for Bedrock API calls

# GLM-Image serving via SGLang
pip install "sglang[diffusion] @ git+https://github.com/sgl-project/sglang.git#subdirectory=python"
pip install git+https://github.com/huggingface/transformers.git
pip install git+https://github.com/huggingface/diffusers.git
```

### 4. Pre-download GLM-Image Weights

Run this in an interactive session on a compute node (the download is ~30 GB):

```bash
pip install accelerate
hf download zai-org/GLM-Image --cache-dir $HF_HOME
```

### 5. Download the Dataset

Download [PaperBananaBench](https://huggingface.co/datasets/dwzhu/PaperBananaBench) and place it under `data/`:

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='dwzhu/PaperBananaBench', repo_type='dataset', local_dir='data/PaperBananaBench')
"
unzip -q data/PaperBananaBench/PaperBananaBench.zip -d data/PaperBananaBench
mv data/PaperBananaBench/PaperBananaBench/* data/PaperBananaBench/
rm -rf data/PaperBananaBench/PaperBananaBench data/PaperBananaBench/PaperBananaBench.zip
```

Expected structure:
```
data/PaperBananaBench/
├── diagram/
│   ├── images/
│   ├── test.json       # 292 test cases
│   └── ref.json        # reference examples for Retriever
└── plot/
```

---

## Running Experiments

You only need to complete the **Setup** section once. On subsequent PSC logins, ensure your environment is configured before running.

### 1. Preparation

Before running any job, you must set your AWS credentials:

```bash
# Required for all model calls (Claude, Qwen)
export AWS_BEARER_TOKEN_BEDROCK="ABSKQmVkcm9ja0FQSUtleS..."
export AWS_BEDROCK_REGION="us-east-1"
```

**Optional: Non-Bedrock Models**
If you wish to use providers like Gemini or OpenRouter directly (bypassing Bedrock), initialize the model config:
```bash
cp configs/model_config.template.yaml configs/model_config.yaml
# Then edit configs/model_config.yaml with your API keys
```

### 2. Quick Local Test (1 sample)

To verify the pipeline works end-to-end before submitting full Slurm jobs, request an interactive node with an 80GB GPU to avoid OutOfMemory errors:

```bash
# Request an interactive H100 80GB node (1 hour limit)
srun --partition=GPU-shared --gres=gpu:h100-80:1 --time=1:00:00 --pty bash

# Activate environment and load compilers
module load anaconda3 cuda gcc/13.3.1-p20240614
export CC=$(which gcc)
export CXX=$(which g++)
export CUDAHOSTCXX=$CXX

conda activate paperbanana
cd $PROJECT/PaperBanana

# Start GLM-Image server in background
sglang serve --model-path zai-org/GLM-Image --port 30000 &

# Run 1-sample smoke test
python main.py \
    --dataset_name PaperBananaBench \
    --task_name diagram \
    --split_name test \
    --exp_mode dev_parallel_debate \
    --max_critic_rounds 1 \
    --main_model_name "bedrock/global.anthropic.claude-opus-4-6-v1" \
    --image_gen_model_name "glm-image" \
    --critic_b_model_name "bedrock/qwen.qwen3-vl-235b-a22b"
```

### Full Runs via Slurm

Both scripts start the GLM-Image SGLang server as a background process, wait for readiness, then run the pipeline. They use `--resume` to skip already-processed samples (safe to re-submit after wall-time interrupts).

```bash
cd $PROJECT/PaperBanana
mkdir -p logs

# Baseline: single-critic (dev_full)
sbatch scripts/run_combined_baseline.sh

# Proposed: parallel debate (dev_parallel_debate)
sbatch scripts/run_combined_parallel.sh
```

### Wall-Time Resumption

PSC jobs on the `GPU-shared` partition have a maximum 48-hour wall time. For runs that you suspect might take even longer, you can chain jobs for automatic resumption:

```bash
JOB1=$(sbatch --parsable scripts/run_combined_parallel.sh)
sbatch --dependency=afterany:$JOB1 scripts/run_combined_parallel.sh
```

Both scripts pass `--resume`, which loads existing results from the output JSON and skips already-processed sample IDs.

---

## CLI Reference

```bash
python main.py [OPTIONS]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset_name` | `PaperBananaBench` | Dataset name |
| `--task_name` | `diagram` | `diagram` or `plot` |
| `--split_name` | `test` | Dataset split |
| `--exp_mode` | `dev` | Pipeline mode (see below) |
| `--retrieval_setting` | `auto` | `auto`, `manual`, `random`, `none` |
| `--max_critic_rounds` | `3` | Max critique-refinement iterations |
| `--main_model_name` | *(from config)* | Main VLM model (e.g. `bedrock/anthropic.claude-opus-4-6-v1`) |
| `--image_gen_model_name` | *(from config)* | Image generation model (e.g. `glm-image`) |
| `--critic_b_model_name` | `""` | Second critic for parallel debate (e.g. `bedrock/qwen.qwen3-vl-235b-a22b`) |
| `--resume` | `false` | Skip already-processed samples |

### Experiment Modes

| Mode | Pipeline |
|------|----------|
| `vanilla` | Direct generation (no planning/refinement) |
| `dev_planner` | Retriever → Planner → Visualizer |
| `dev_planner_stylist` | Retriever → Planner → Stylist → Visualizer |
| `dev_planner_critic` | Retriever → Planner → Visualizer → Critic loop |
| `dev_full` | Full pipeline (Retriever → Planner → Stylist → Visualizer → Critic loop) |
| **`dev_parallel_debate`** | **Full pipeline + Parallel Debate (two critics → synthesizer → visualizer loop)** |

---

## Project Structure

```
PaperBanana/
├── agents/
│   ├── base_agent.py
│   ├── retriever_agent.py
│   ├── planner_agent.py
│   ├── stylist_agent.py
│   ├── visualizer_agent.py
│   ├── critic_agent.py
│   ├── parallel_critic_agent.py       # NEW
│   ├── synthesizer_agent.py           # NEW
│   ├── vanilla_agent.py
│   └── polish_agent.py
├── utils/
│   ├── config.py
│   ├── paperviz_processor.py          # Pipeline orchestration
│   ├── eval_toolkits.py               # VLM-as-Judge evaluation
│   ├── generation_utils.py            # Unified model router (Bedrock, Gemini, etc.)
│   └── image_utils.py
├── configs/
│   └── model_config.template.yaml
├── scripts/
│   ├── run_combined_baseline.sh       # Slurm: single-critic baseline
│   ├── run_combined_parallel.sh       # Slurm: parallel debate
│   ├── eval_round.py                  # Post-hoc eval of intermediate rounds
│   └── ablation_table.py              # Aggregate evals into ablation table
├── data/
│   └── PaperBananaBench/
├── results/
├── main.py                            # Entry point
├── requirements.txt
└── README.md
```

---

## Results

Results are saved to `results/PaperBananaBench_diagram/<timestamp>_<mode>_test.json`.

Each result JSON contains per-sample data including:
- Generated images (base64) at **every** critic round (not just the final one)
- Critic suggestions and revised descriptions per round
- Synthesis reasoning (parallel debate only)
- VLM-as-Judge evaluation scores — but **only for the final round's image**

### Post-Hoc Evaluation (Mini-Ablation)

The main pipeline evaluates only the final image (t=3). To build the full ablation table, you need to evaluate intermediate rounds (t=0, t=1) separately. This does **not** require a GPU — only Bedrock API calls.

**Step 1:** After both generation runs complete, run post-hoc evaluations:

```bash
# Let BASELINE=results/.../timestamp_dev_full_test.json
# Let PARALLEL=results/.../timestamp_dev_parallel_debate_test.json

# t=0 (shared initial image, same for both pipelines)
python scripts/eval_round.py --input $BASELINE --round 0 --output results/t0_eval.json

# Solo Critic t=1
python scripts/eval_round.py --input $BASELINE --round 1 --output results/solo_t1_eval.json

# Parallel Debate t=1
python scripts/eval_round.py --input $PARALLEL --round 1 --output results/debate_t1_eval.json
```

(Solo t=3 and Parallel t=3 are already evaluated by the main pipeline.)

**Step 2:** Aggregate into a single table:

```bash
python scripts/ablation_table.py \
    --t0        results/t0_eval.json \
    --solo_t1   results/solo_t1_eval.json \
    --solo_t3   $BASELINE \
    --debate_t1 results/debate_t1_eval.json \
    --debate_t3 $PARALLEL \
    --csv       results/ablation_table.csv
```

Output:
```
=============================================================================
Condition                 | Faith         | Conci         | Reada         | Aesth         | Overa         |   N
-----------------------------------------------------------------------------
t=0 (shared)              | 35.2/ 30.1/ 34.7 | ...           | ...           | ...           | ...           | 292
Solo Critic t=1           | 40.3/ 28.5/ 31.2 | ...           | ...           | ...           | ...           | 292
Solo Critic t=3           | 43.1/ 25.0/ 31.9 | ...           | ...           | ...           | ...           | 292
Parallel Debate t=1       | 45.5/ 27.0/ 27.5 | ...           | ...           | ...           | ...           | 292
Parallel Debate t=3       | 48.2/ 24.1/ 27.7 | ...           | ...           | ...           | ...           | 292
=============================================================================
Format: Model% / Tie% / Human%
```

### Visualize Results

```bash
# View pipeline evolution (click through critic rounds)
streamlit run visualize/show_pipeline_evolution.py

# View evaluation scorecard
streamlit run visualize/show_referenced_eval.py
```

---

## Acknowledgments

Built on [PaperBanana](https://github.com/dwzhu-pku/PaperBanana) (Zhu et al., 2026). Uses [GLM-Image](https://github.com/THUDM/GLM-Image) served via [SGLang](https://github.com/sgl-project/sglang), and [AWS Bedrock](https://aws.amazon.com/bedrock/) for Claude and Qwen VLM access.
