# Faithful Diagram Generation via Multi-Agent VLM Consensus

> **10-623 Course Project** — Extends [PaperBanana](https://github.com/dwzhu-pku/PaperBanana) with a *Parallel Debate* architecture: two diverse VLM critics independently evaluate generated diagrams, then a synthesizer reconciles their feedback into a unified refinement prompt. This targets the "fine-grained faithfulness gap" (missed spatial errors, hallucinated connections) identified in the original paper.

## Architecture

```
Planner → Visualizer ─┬─→ Critic A (Llama 4 Maverick)  ─→ Synthesizer (Llama 4 Maverick) → Visualizer → …
                       └─→ Critic B (Claude Sonnet 4.6)  ─↗
```

| Role | Model | Provider |
|------|-------|----------|
| Planner, Synthesizer, Critic A | Llama 4 Maverick 17B | AWS Bedrock |
| Critic B | Claude Sonnet 4.6 | AWS Bedrock |
| Visualizer (image gen) | FLUX-2-pro | Replicate API |
| Evaluation Judge | Llama 4 Maverick 17B | AWS Bedrock |

> **Note:** The Retriever and Stylist agents from the original PaperBanana pipeline are **omitted** in our experiments. Retriever was budget-prohibitive, and Stylist showed minimal impact on faithfulness per the paper's own ablation. Their code remains in the codebase for backward compatibility.

**Key files:**
- `agents/parallel_critic_agent.py` — Runs two critics in parallel via `asyncio.gather`
- `agents/synthesizer_agent.py` — Reconciles two critiques into unified refinement
- `agents/critic_agent.py` — Single critic (supports `output_prefix` for namespaced keys)
- `utils/paperviz_processor.py` — `dev_parallel_debate` pipeline mode
- `utils/generation_utils.py` — Unified model router (Bedrock, Gemini, Anthropic, OpenAI, OpenRouter)

---

## Setup

### Option A: Local Setup (Replicate API)

For running experiments locally without a GPU cluster, using FLUX-2-pro via Replicate for image generation. See [SETUP_FLUX_API.md](SETUP_FLUX_API.md) for details.

```bash
conda create -n paperbanana python=3.12 -y
conda activate paperbanana
pip install -r requirements.txt

export REPLICATE_API_TOKEN="..."
export AWS_BEARER_TOKEN_BEDROCK="..."
```

### Option B: PSC Bridges-2 Setup (Self-Hosted FLUX.2-dev)

For running experiments on PSC with a self-hosted FLUX.2-dev image generation server.

#### Prerequisites

- **PSC Bridges-2 account** with GPU allocation (V100-32GB for baseline, H100-80GB for full pipeline)
- **AWS Bedrock access** with an ABSK bearer token for Llama 4 Maverick and Claude Sonnet 4.6 in `us-east-1`
- **Hugging Face account** with access to gated model `black-forest-labs/FLUX.2-dev`
- Python 3.12

#### 1. PSC Storage Configuration

Add to your `~/.bashrc` (critical due to 25 GB home directory quota):

```bash
export HF_HOME=$PROJECT/.cache/huggingface
export TRANSFORMERS_CACHE=$PROJECT/.cache/huggingface/transformers
export HF_DATASETS_CACHE=$PROJECT/.cache/huggingface/datasets
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export CONDA_PKGS_DIRS=$PROJECT/.conda/pkgs
export CONDA_ENVS_DIRS=$PROJECT/.conda/envs
export PIP_CACHE_DIR=$PROJECT/.cache/pip
export TMPDIR=$PROJECT/tmp
```

#### 2. Clone and Install

```bash
cd $PROJECT
git clone https://github.com/KenYe23/10623-project.git PaperBanana
module load anaconda3
conda create -n paperbanana python=3.12 -y
conda activate paperbanana

cd PaperBanana
pip install -r requirements.txt
pip install hf_transfer
```

#### 3. Hugging Face Login (for FLUX.2-dev)

`FLUX.2-dev` is gated — accept model terms at https://huggingface.co/black-forest-labs/FLUX.2-dev, then:

```bash
hf auth login
```

#### 4. Pre-download FLUX.2-dev Weights

For the **quantized** server (V100-compatible):
```bash
hf download diffusers/FLUX.2-dev-bnb-4bit --local-dir $PROJECT/models/FLUX.2-dev-bnb-4bit
```

For the **full-precision** server (H100 only):
```bash
hf download black-forest-labs/FLUX.2-dev --local-dir $PROJECT/models/FLUX.2-dev
```

#### 5. Download the Dataset

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
│   ├── test.json          # 292 test cases (full)
│   ├── test_8_4kb.json    # 50 test cases (methodology <8.4 KB)
│   └── ref.json           # reference examples for Retriever
└── plot/
```

---

## Running Experiments

### Environment Variables

```bash
# Required for all Bedrock model calls
export AWS_BEARER_TOKEN_BEDROCK="ABSKQmVkcm9ja0FQSUtleS..."
export AWS_BEDROCK_REGION="us-east-1"

# For Replicate API (local runs)
export REPLICATE_API_TOKEN="..."

# For self-hosted FLUX server (PSC runs)
export FLUX2_SERVER_URL="http://localhost:30000"
```

### Local Run (Replicate API)

```bash
# Baseline: single-critic
python main.py \
    --dataset_name PaperBananaBench \
    --task_name diagram \
    --split_name test_8_4kb \
    --exp_mode dev_full \
    --max_critic_rounds 3 \
    --main_model_name "bedrock/us.meta.llama4-maverick-17b-instruct-v1:0" \
    --retrieval_setting none \
    --image_gen_model_name "flux-2-pro" \
    --max_concurrent 1

# Parallel Debate
python main.py \
    --dataset_name PaperBananaBench \
    --task_name diagram \
    --split_name test_8_4kb \
    --exp_mode dev_parallel_debate \
    --max_critic_rounds 3 \
    --main_model_name "bedrock/us.meta.llama4-maverick-17b-instruct-v1:0" \
    --critic_b_model_name "bedrock/global.anthropic.claude-sonnet-4-6" \
    --retrieval_setting none \
    --image_gen_model_name "flux-2-pro" \
    --max_concurrent 1
```

### PSC Slurm Runs

Both scripts start the FLUX.2-dev server as a background process, wait for readiness, then run the pipeline with `--resume` to skip already-processed samples.

```bash
cd $PROJECT/PaperBanana
mkdir -p logs

# Baseline: single-critic (V100, ~9 hours)
sbatch scripts/run_combined_baseline.sh

# Parallel Debate (H100, ~48 hours)
MAIN_MODEL_NAME="bedrock/us.meta.llama4-maverick-17b-instruct-v1:0" \
sbatch scripts/run_combined_parallel.sh

# Sharded run (process samples 0-49)
START_IDX=0 END_IDX=50 sbatch scripts/run_combined_parallel.sh

# Merge shard outputs after all jobs finish
python scripts/merge_shard_results.py \
    --inputs results/PaperBananaBench_diagram/*dev_parallel_debate_test*.json \
    --output results/PaperBananaBench_diagram/merged_parallel.json
```

---

## CLI Reference

```bash
python main.py [OPTIONS]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset_name` | `PaperBananaBench` | Dataset name |
| `--task_name` | `diagram` | `diagram` or `plot` |
| `--split_name` | `test` | Dataset split (e.g. `test`, `test_8_4kb`) |
| `--exp_mode` | `dev` | Pipeline mode (see below) |
| `--retrieval_setting` | `auto` | `auto`, `manual`, `random`, `none` |
| `--max_critic_rounds` | `3` | Max critique-refinement iterations |
| `--max_samples` | `0` | Limit number of samples (`0` = all) |
| `--start_idx` | `0` | Start index (inclusive) for shard processing |
| `--end_idx` | `-1` | End index (exclusive) for shard processing (`-1` = end) |
| `--max_concurrent` | `10` | Max concurrent samples |
| `--main_model_name` | *(from config)* | Main VLM (e.g. `bedrock/us.meta.llama4-maverick-17b-instruct-v1:0`) |
| `--image_gen_model_name` | *(from config)* | Image gen model (e.g. `flux-2-pro`, `flux2-dev`) |
| `--critic_b_model_name` | `""` | Second critic for parallel debate |
| `--resume` | `false` | Skip already-processed samples |

### Experiment Modes

| Mode | Pipeline |
|------|----------|
| `vanilla` | Direct generation (no planning/refinement) |
| `dev_planner` | Retriever → Planner → Visualizer |
| `dev_planner_stylist` | Retriever → Planner → Stylist → Visualizer |
| `dev_planner_critic` | Retriever → Planner → Visualizer → Critic loop |
| `dev_full` | Planner → Visualizer → Critic loop (baseline) |
| **`dev_parallel_debate`** | **Planner → Visualizer → [Critic A ∥ Critic B] → Synthesizer → Visualizer loop** |

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
│   ├── parallel_critic_agent.py       # NEW — parallel critique orchestrator
│   ├── synthesizer_agent.py           # NEW — critique reconciliation
│   ├── vanilla_agent.py
│   └── polish_agent.py
├── utils/
│   ├── config.py                      # ExpConfig dataclass
│   ├── paperviz_processor.py          # Pipeline orchestration
│   ├── eval_toolkits.py               # VLM-as-Judge evaluation
│   ├── generation_utils.py            # Unified model router (Bedrock, Gemini, etc.)
│   └── image_utils.py
├── prompts/                           # System prompt templates
├── configs/
│   └── model_config.template.yaml
├── scripts/
│   ├── run_combined_baseline.sh       # Slurm: single-critic baseline
│   ├── run_combined_parallel.sh       # Slurm: parallel debate
│   ├── flux2_http_server.py           # Full-precision FLUX.2-dev server (H100)
│   ├── flux2_quantized_server.py      # 4-bit quantized FLUX.2-dev server (V100)
│   ├── eval_round.py                  # Post-hoc eval of intermediate rounds
│   ├── ablation_table.py              # Aggregate evals into ablation table
│   ├── merge_shard_results.py         # Merge sharded result JSONs
│   ├── create_test_subset.py          # Create filtered test subsets
│   └── create_qwen_compatible_subset.py
├── visualize/
│   ├── show_pipeline_evolution.py     # Streamlit: per-sample pipeline viewer
│   └── show_referenced_eval.py        # Streamlit: evaluation scorecard
├── data/
│   └── PaperBananaBench/
├── results/
├── main.py                            # Entry point
├── SETUP_FLUX_API.md                  # Replicate API setup guide
├── requirements.txt
└── README.md
```

---

## Results

Results are saved to `results/PaperBananaBench_diagram/<timestamp>_<mode>_<split>.json`.

Each result JSON contains per-sample data including:
- Generated images (base64) at **every** critic round (not just the final one)
- Critic suggestions and revised descriptions per round
- Synthesis reasoning (parallel debate only)
- VLM-as-Judge evaluation scores for the **final round's image**

### Post-Hoc Evaluation (Ablation Table)

The main pipeline evaluates only the final image (t=3). To build the full ablation table, evaluate intermediate rounds separately. This does **not** require a GPU — only Bedrock API calls.

**Step 1:** After both generation runs complete, run post-hoc evaluations:

```bash
# Let BASELINE=results/.../timestamp_dev_full_test.json
# Let PARALLEL=results/.../timestamp_dev_parallel_debate_test.json

# t=0 (shared initial image, same for both pipelines)
python scripts/eval_round.py --input $BASELINE --round 0 --output results/t0_eval.json \
    --eval_model_name "bedrock/us.meta.llama4-maverick-17b-instruct-v1:0"

# Solo Critic t=1
python scripts/eval_round.py --input $BASELINE --round 1 --output results/solo_t1_eval.json \
    --eval_model_name "bedrock/us.meta.llama4-maverick-17b-instruct-v1:0"

# Parallel Debate t=1
python scripts/eval_round.py --input $PARALLEL --round 1 --output results/debate_t1_eval.json \
    --eval_model_name "bedrock/us.meta.llama4-maverick-17b-instruct-v1:0"
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

### Experimental Results (n=50, Llama 4 Maverick Judge)

Format: Model% / Tie% / Human%

| Condition | Faithfulness | Conciseness | Readability | Aesthetics | Overall |
|---|---|---|---|---|---|
| t=0 (shared) | 0/82/18 | 18/56/26 | 2/92/6 | 12/60/28 | 20/34/46 |
| Solo Critic t=1 | 0/80/20 | 22/44/34 | 4/94/2 | 16/62/22 | 26/34/40 |
| Solo Critic t=3 | 0/82/18 | 18/56/26 | 2/92/6 | 12/60/28 | 20/34/46 |
| Parallel Debate t=1 | 0/76/24 | 16/48/36 | 2/90/8 | 12/56/32 | 16/36/48 |
| **Parallel Debate t=3** | **0/88/12** | **18/52/30** | **2/94/4** | **16/68/16** | **22/36/42** |

### Visualize Results

```bash
# Pipeline evolution viewer
streamlit run visualize/show_pipeline_evolution.py

# Evaluation scorecard
streamlit run visualize/show_referenced_eval.py
```

If running on PSC, create an SSH tunnel from your local machine:

```bash
ssh -N -L 8501:<compute-node>:8501 <username>@bridges2.psc.edu
# Then open http://localhost:8501
```

---

## Acknowledgments

Built on [PaperBanana](https://github.com/dwzhu-pku/PaperBanana) (Zhu et al., 2026). Uses [FLUX.2-dev](https://github.com/black-forest-labs/flux2) for self-hosted image generation, [Replicate](https://replicate.com/) for FLUX-2-pro cloud API, and [AWS Bedrock](https://aws.amazon.com/bedrock/) for Llama 4 Maverick and Claude Sonnet VLM access.
