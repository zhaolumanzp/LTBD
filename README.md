# LTBD

Official implementation of **Learnable Trust-Boundary Delimiters (LTBD)** for prompt injection defense.

## Setup

* Install environment dependencies via [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/zhaolumanzp/LTBD.git
cd LTBD

uv venv ltbd --python 3.13
source ltbd/bin/activate

uv pip install -r requirements.txt
```

* Download the base models used in our experiments from Hugging Face:

  * [Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
  * [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
  * [Falcon3-7B-Instruct](https://huggingface.co/tiiuae/Falcon3-7B-Instruct)
  * [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

For example, the model can be downloaded with `huggingface-cli`:

```bash
huggingface-cli download \
    meta-llama/Llama-3.1-8B-Instruct \
    --local-dir models/Llama-3.1-8B-Instruct
```

* Download the [Cleaned Alpaca](https://github.com/gururise/AlpacaDataCleaned) instruction-tuning dataset used for defense training.

```bash
git clone https://github.com/gururise/AlpacaDataCleaned.git
```

## Construct Defense Training Data

* **Step 1: Generate self-labeled responses.**

```bash
python self_label.py \
    --model "$MODEL" \
    --output data/self_labeled.jsonl \
    --batch_size 192 \
    --max_samples 51760 \
    --seed 0
```

The original base model is used to generate reference responses on the Cleaned Alpaca dataset.

* **Step 2: Construct the fixed prompt-injected dataset \(D'\).**

```bash
python train_defensive_tokens.py \
    --model "$MODEL" \
    --self_labeled data/self_labeled.jsonl \
    --output_dir outputs/<MODEL_NAME>_delimiter_model_pref_lambda0.3 \
    --export_dprime data/dprime_chosen.jsonl \
    --max_length 1024 \
    --max_samples 51760 \
    --seed 0
```

The constructed dataset contains clean and prompt-injected samples, including `ignore`, `completion`, and `delimiter_spoof` variants.

* **Step 3: Generate rejected responses for preference training.**

```bash
python gen_rejected.py \
    --model "$MODEL" \
    --dprime data/dprime_chosen.jsonl \
    --output data/pref_dprime.jsonl \
    --max_new_tokens 1024 \
    --temperature 0.8 \
    --seed 0 \
    --batch_size 96
```

For each sample in \(D'\), the desired response is used as the `chosen` response, while the original base model generates the corresponding `rejected` response using the plain prompt without LTBD delimiters.

## Train Learnable Trust-Boundary Delimiters

LTBD introduces four learnable special tokens:

```text
<INST_BEGIN>
<INST_END>
<DATA_BEGIN>
<DATA_END>
```

During training, all original LLM parameters are frozen and only the embeddings of these four delimiter tokens are optimized.

* **Step 4: Train LTBD with the defense and preference objectives.**

```bash
python train_defensive_tokens.py \
    --model "$MODEL" \
    --self_labeled data/self_labeled.jsonl \
    --pref_data data/pref_dprime.jsonl \
    --output_dir outputs/<MODEL_NAME>_delimiter_model_pref_lambda0.3 \
    --export_json outputs/<MODEL_NAME>_delimiter_model_pref_lambda0.3/delimiter_embeddings.json \
    --lr 0.1 \
    --pref_lambda 0.3 \
    --pref_beta 0.1 \
    --epochs 1 \
    --max_length 1024 \
    --batch_size 1 \
    --grad_accum 16 \
    --seed 0
```

The complete training pipeline can also be executed using the provided shell script after configuring the model and data paths:

```bash
bash run_train.sh
```

If `self_labeled.jsonl`, `dprime_chosen.jsonl`, and `pref_dprime.jsonl` have already been generated, Steps 1–3 can be skipped.

## Evaluation

* Reproduce LTBD test results using [Meta_SecAlign](https://github.com/facebookresearch/Meta_SecAlign).

* Specify the trained LTBD model with:

```bash
-m model_path
```

For example:

```bash
python test.py \
    -a ignore completion completion_ignore \
    -d none \
    --test_data data/davinci_003_outputs.json \
    -m /path/to/LTBD_model
```

* Use `"role": "system"` for the trusted instruction and `"role": "user"` for the untrusted data.

Meta_SecAlign originally uses `"role": "user"` for the trusted instruction and `"role": "input"` for the untrusted data. Modify the corresponding prompt construction in [utils.py](https://github.com/facebookresearch/Meta_SecAlign/blob/main/utils.py#L250).

* Enable LTBD when calling `tokenizer.apply_chat_template`:

```python
tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    add_delimiter_tokens=True,
)
```

With LTBD enabled, the model input follows the structure:

```text
[system]
<INST_BEGIN> trusted instruction <INST_END>

[user]
<DATA_BEGIN> untrusted data <DATA_END>
```

## Acknowledgements

Our implementation and evaluation build upon [DefensiveToken](https://github.com/Sizhe-Chen/DefensiveToken) and [Meta_SecAlign](https://github.com/facebookresearch/Meta_SecAlign). We thank the authors for releasing their code and evaluation framework.
