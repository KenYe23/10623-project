# Replicate API Setup

Use this guide to run the project locally with `flux-2-pro` served through Replicate.

## 1. Create the environment

```bash
conda create -n paperbanana python=3.12 -y
conda activate paperbanana
pip install -r requirements.txt
```

## 2. Export required credentials

You need both of the following:

- `REPLICATE_API_TOKEN` for image generation with Replicate / FLUX
- `AWS_BEARER_TOKEN_BEDROCK` for the Bedrock text models used by the agents

```bash
export REPLICATE_API_TOKEN=...
export AWS_BEARER_TOKEN_BEDROCK=...
```

If you want these variables to persist across terminal sessions, add them to your shell profile such as `~/.zshrc`.

## 3. Run the baseline experiment

```bash
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
```

## 4. Run parallel debate

```bash
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

## Notes

- The commands above use the `test_8_4kb` split, which is intentionally tracked for testing, which include 50 test cases with methodology less than 8.4kb.
