# en-ar-translation

Notebook-first workflow for building a bidirectional English <-> Arabic translation dataset and training pipeline (Transformer encoder-decoder).

## Current Project Status

- Dataset discovery notebook is implemented: `notebooks/01_dataset_discovery.ipynb`.
- Data cleaning and normalization pipeline is implemented in the notebook.
- Full cleaned combined dataset is exported for training reuse.
- EDA artifacts are exported under `artifacts/eda/`.
- Model/training notebook (`notebooks/02_model_training.ipynb`) is the next phase.

## Data Sources (with credit)

These are the five datasets currently used in the combined dataset, in variable order:

1. `local_25k` (Kaggle):
https://www.kaggle.com/datasets/tahaalselwii/the-arabic-english-sentence-bank-25k?resource=download

2. `ds2` (Hugging Face):
https://huggingface.co/datasets/Arabic-Clip-Archive/ImageCaptions-7M-Translations-Arabic/viewer/default/train?p=1132

3. `ds3` (Hugging Face):
https://huggingface.co/datasets/salehalmansour/english-to-arabic-translate

4. `ds4` (Hugging Face):
https://huggingface.co/datasets/ammagra/english-arabic-speech-translation

5. `ds5` (Kaggle):
https://www.kaggle.com/datasets/yumnagamal/translation-english-arabic?select=merge_df.csv

## Dataset Building Pipeline (Notebook 01)

The notebook currently performs the following steps:

1. Loads datasets from local files and Hugging Face.
2. Inspects source schemas to choose English/Arabic columns.
3. Standardizes all sources into two columns: `en`, `ar`.
4. Merges all sources.
5. Removes exact duplicate (`en`, `ar`) pairs.
6. Detects likely swapped-language rows (Arabic in `en`, English in `ar`) and swaps them back.
7. Cleans boundary newline artifacts (literal `\n` and real line breaks).
8. Removes Arabic diacritics from the Arabic column.
9. Re-deduplicates after cleaning steps when needed.
10. Runs EDA statistics (nulls, uniqueness, word/char lengths, script anomalies, top frequencies, sample preview, truncation estimate).

## Exported Outputs

### Full cleaned dataset export

- `artifacts/eda/final_cleaned_combined_dataset.parquet` (primary for training)
- `artifacts/eda/final_cleaned_combined_dataset.csv` (portable copy)
- `artifacts/eda/final_cleaned_combined_dataset_metadata.json`

### EDA exports

- `artifacts/eda/eda_summary.json`
- `artifacts/eda/word_length_stats.csv`
- `artifacts/eda/char_length_stats.csv`
- `artifacts/eda/anomaly_summary.csv`
- `artifacts/eda/truncation_estimate.csv`
- `artifacts/eda/top_en.csv`
- `artifacts/eda/top_ar.csv`
- `artifacts/eda/sample_preview.csv`

## Notes

- `MAX_SEQ_LEN` reference in EDA is currently set to `128`.
- Truncation estimate in Notebook 01 is word-level proxy; final truncation should be validated with the actual tokenizer in training notebook.
- Current final row count depends on latest notebook run and cleaning outputs (see `artifacts/eda/eda_summary.json`).

## Model + Training Progress (Notebook 02)

This section documents what has already been implemented and trained so far in `notebooks/02_model_training.ipynb`.

### 1) Training Data Pipeline (Implemented)

1. Loads training source from:
- `artifacts/eda/final_cleaned_combined_dataset.parquet`

2. Applies row-level cleanup before split:
- Drops null `en/ar`
- Trims whitespace
- Removes empty-string rows

3. Deterministic split (leakage-safe):
- Hash-based split on exact `(en, ar)` pair
- Ratios: `train=0.90`, `val=0.05`, `test=0.05`
- Leakage guard: asserts that no exact pair appears in more than one split

4. Bidirectional expansion:
- For EN -> AR: source is prefixed with `<2ar>`
- For AR -> EN: source is prefixed with `<2en>`
- This doubles examples by direction

5. Split sizes used:
- Cleaned base pairs: `827,546`
- Base split: `train=744,926`, `val=41,445`, `test=41,175`
- Bidirectional split: `train=1,489,852`, `val=82,890`, `test=82,350`

### 2) Tokenizer (Implemented)

1. Tokenizer family:
- Shared ByteLevel BPE tokenizer (trained from scratch)

2. Training data for tokenizer:
- Train split only (both `source_text` and `target_text`)

3. Tokenizer settings:
- `vocab_size=32,000`
- Special tokens: `<pad>`, `<s>`, `</s>`, `<unk>`, `<2ar>`, `<2en>`

4. Artifacts:
- Raw tokenizer JSON under `artifacts/tokenizer/`
- HF-compatible tokenizer under `artifacts/tokenizer/hf_tokenizer/`

5. Tokenization behavior for model input:
- `max_seq_len=128`
- Truncation enabled
- No static padding at tokenize stage (`padding=False`)

### 3) Model Architecture (Implemented)

Model is a BART-style encoder-decoder trained from random initialization (`BartForConditionalGeneration`).

Configuration used:
- `d_model=512`
- `encoder_layers=6`
- `decoder_layers=6`
- `encoder_attention_heads=8`
- `decoder_attention_heads=8`
- `encoder_ffn_dim=2048`
- `decoder_ffn_dim=2048`
- `dropout=0.1`
- `attention_dropout=0.1`
- `activation_dropout=0.0`
- `max_position_embeddings=130` (`max_seq_len + 2`)
- Token IDs sourced from HF tokenizer (`pad/bos/eos`)
- `decoder_start_token_id=bos_token_id`

### 4) Training System Setup (Implemented)

1. Padding/batching:
- Dynamic padding via `DataCollatorForSeq2Seq`
- Label pad token id: `-100`

2. Precision and memory:
- Mixed precision enabled (`FP16`) with `torch.amp.autocast` + `GradScaler`
- Gradient checkpointing disabled in the completed run (`False`)

3. Optimizer:
- `AdamW`
- `learning_rate=3e-4`
- `weight_decay=0.01`
- `betas=(0.9, 0.98)`
- `eps=1e-8`

4. LR scheduler:
- `get_cosine_with_min_lr_schedule_with_warmup`
- `warmup_ratio=0.015`
- `num_warmup_steps=480` (for `max_steps=32,000`)
- `min_lr_rate=0.10` (minimum LR is 10% of initial LR)

### 5) Completed Full Training Run So Far

Run ID:
- `20260218_003516`

Core run hyperparameters (from `artifacts/runs/20260218_003516/run_config.json`):
- `max_steps=32,000`
- `per_device_train_batch_size=16`
- `gradient_accumulation_steps=8`
- Effective batch size per optimizer update: `128` examples
- Logging/eval/checkpoint:
- `log_every_steps=40`
- `eval_every_steps=400`
- `save_every_steps=500`
- `keep_last_n_checkpoints=10`
- Validation loss loop cap: `max_val_batches=300`
- In-training text metric decode:
- `text_metric_num_beams=1`
- `max_text_metric_batches_log=4`
- `max_text_metric_batches_eval=32`
- `enable_comet=false` in this completed run (COMET column is `NaN` for that run)

Approximate token throughput context from this run:
- Mean tokens/example (source + target) ~= `48.4`
- Effective tokens/update ~= `6,200` (derived from logs and effective batch size 128)

### 6) Metrics Logged During Training

Train-side CSV:
- `artifacts/runs/20260218_003516/train_metrics.csv`
- Includes: step, train loss, train PPL, LR, grad norm, timing, memory, sampled BLEU/chrF/COMET, direction-wise BLEU/chrF

Eval-side CSV:
- `artifacts/runs/20260218_003516/eval_metrics.csv`
- Includes: step, val loss, val PPL, LR, timing, sampled BLEU/chrF/COMET, direction-wise BLEU/chrF

### 7) Best/Final Snapshot from Completed Run

From `artifacts/runs/20260218_003516/eval_metrics.csv`:

1. Best validation loss:
- Step `29,200`
- `val_loss=2.2382`
- `val_ppl=9.3769`
- `bleu=1.7312`
- `chrf=22.4940`

2. Best BLEU:
- Step `25,600`
- `bleu=1.7903`

3. Best chrF:
- Step `24,000`
- `chrf=23.8769`

4. Final step (`32,000`) eval:
- `val_loss=2.2853`
- `val_ppl=9.8289`
- `bleu=1.4676`
- `chrf=21.6023`

Checkpoint output:
- Best checkpoint directory: `checkpoints/20260218_003516/best_model`
- Saved step checkpoints: `checkpoints/20260218_003516/step_*`

### 8) Added Post-Run Evaluation Cells (Notebook 02)

Two additional cells were added for full test evaluation from `best_model`:

1. `Micro-step 22a`
- Loads `best_model`
- Loads BLEU/chrF/COMET metric objects
- Sets beam search for evaluation

2. `Micro-step 22b`
- Runs full test evaluation on `test_df`
- Saves:
- `artifacts/runs/<run_id>/test_metrics_best_model_full.json`
- `artifacts/runs/<run_id>/test_metrics_best_model_full.csv`
