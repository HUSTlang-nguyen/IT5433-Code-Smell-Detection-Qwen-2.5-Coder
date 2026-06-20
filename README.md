# IT5433 Code Smell Detection

Kaggle-first implementation of LLM-based Java code smell detection on DeepLearningSmells raw source code.

## Files

- `notebooks/code_smell_llm_qwen.ipynb`: self-contained Kaggle notebook.
- `report/main.tex`: English LaTeX report template.
- `report/references.bib`: bibliography.

No separate Python module is required on Kaggle.

## Kaggle Setup

1. Upload `notebooks/code_smell_llm_qwen.ipynb` to Kaggle.
2. Attach a Kaggle Dataset containing Java raw archives under `training_data_java/` or `dataset_java/`.
3. Use GPU `T4 x2`.
4. Run a smoke test first with `FAST_RUN=True`.

The notebook samples and extracts only the selected raw source files instead of unpacking the full archives.

## Safe Full Run

For full Qwen 3B training, use `train_only` first so checkpoints can be backed up before validation/test:

```python
FAST_RUN = False

cfg = Config(
    fast_run=FAST_RUN,
    work_dir=Path("/kaggle/working/code_smell_llm_full"),
    run_baseline=False,
    run_qwen_3b=True,
    run_qwen_7b=False,
    qwen_run_mode="train_only",
    checkpoint_backup_enabled=True,
    checkpoint_dataset_slug="YOUR_USERNAME/code-smell-qwen-checkpoints",
    checkpoint_backup_keep_last=1,
    save_steps=200,
    save_total_limit=4,
    eval_batch_size=1,
    resume_from_checkpoint=True,
)
```

Create `YOUR_USERNAME/code-smell-qwen-checkpoints` as a private Kaggle Dataset before enabling checkpoint backup.

After training finishes, or after a timeout with checkpoint backup available, run evaluation separately:

```python
cfg = Config(
    fast_run=False,
    work_dir=Path("/kaggle/working/code_smell_llm_full"),
    run_baseline=False,
    run_qwen_3b=True,
    run_qwen_7b=False,
    qwen_run_mode="eval_only",
    checkpoint_dataset_slug="YOUR_USERNAME/code-smell-qwen-checkpoints",
    restore_checkpoint_from_dataset=True,
    eval_batch_size=1,
)
```

`train_then_eval` is intended for smoke tests only.

## Evaluation Strategy

The notebook keeps Hugging Face Trainer validation disabled:

```python
evaluation_strategy="no"
```

This avoids validation-loss OOM on Kaggle T4. Checkpoint selection is done after training by YES/NO inference on the validation split, selecting the best checkpoint by MCC with F1 as a tie-breaker.

## Outputs

Main outputs:

```text
/kaggle/working/code_smell_llm_full/outputs/dataset_index.csv
/kaggle/working/code_smell_llm_full/outputs/metrics.csv
/kaggle/working/code_smell_llm_full/outputs/predictions.csv
/kaggle/working/code_smell_llm_full/outputs/checkpoint_validation_metrics.csv
/kaggle/working/code_smell_llm_full/outputs/best_checkpoints.csv
/kaggle/working/code_smell_llm_full/outputs/training_log_*.csv
/kaggle/working/code_smell_llm_full/outputs/training_log_live_*.csv
/kaggle/working/code_smell_llm_full/outputs/confusion_matrices/
/kaggle/working/code_smell_llm_full/outputs/report_tables/
```

Checkpoint backup dataset contains:

```text
checkpoint_bundle_Qwen2_5_Coder_3B_Instruct.zip
manifest.json
training_log_live_*.csv
```

## Models

- Baseline: `TF-IDF char n-gram + LinearSVC`
- Main: `Qwen/Qwen2.5-Coder-3B-Instruct + QLoRA`
- Optional size comparison: `Qwen/Qwen2.5-Coder-7B-Instruct + QLoRA`

Run 3B first. Run 7B only after the 3B workflow is stable.

## OOM Fallback

If CUDA OOM happens, reduce:

```python
max_seq_length=1024
code_char_limit=6000
per_device_train_batch_size=1
gradient_accumulation_steps=16
run_qwen_7b=False
```
