# en-ar-translation

Notebook-first project for bidirectional English <-> Arabic translation with a Transformer encoder-decoder pipeline.

## Project Overview

This repository contains:
- Dataset discovery, cleaning, and consolidation (`notebooks/01_dataset_discovery.ipynb`)
- Tokenizer training + model training/resume workflows (`notebooks/02_model_training.ipynb`)
- Training artifacts, checkpoints, and metric logs under `artifacts/` and `checkpoints/`

## Current Status

- Dataset pipeline is complete and exported.
- Initial full training phase completed to `32,000` steps.
- Additional resumed training phase completed to `37,000` steps (stopped manually).
- Best-so-far validation metrics have improved after resumed training.
- Full test-set evaluation section is prepared in notebook and pending final run completion.

## Transformer Architecture

<!-- TODO: Replace with your final architecture image path -->
![Transformer Architecture](docs/images/transformer-architecture.webp)

Reference: *[Attention Is All You Need](https://arxiv.org/abs/1706.03762)* (Vaswani et al., 2017).  
Architecture note: this implementation follows the same high-level Transformer design, with `pre-norm` blocks (LayerNorm before each attention/FFN sublayer) instead of the original paper's post-sublayer normalization, plus an additional final normalization layer at the end of the decoder.

Image tip: place your image in `docs/images/` (for example `docs/images/transformer-architecture.png`) and keep this Markdown line format.

## Training Snapshot

Run line tracked in this project:
- Primary run ID: `20260218_003516`
- Training hardware: single GPU `NVIDIA RTX 4060 (8GB VRAM)`

Training phases:

| Phase | Step Range | Batch Size | Grad Accum | Effective Batch | Time |
|---|---:|---:|---:|---:|---:|
| Initial training | `0 -> 32,000` | `16` | `8` | `128` | `5.34 h` |
| Resumed training | `32,000 -> 37,000` | `24` | `32` | `768` | `6.13 h` |

Total cumulative training time so far:
- `11.47 h` (`5.34 + 6.13`)

## Best Validation Metrics So Far

Source: `artifacts/runs/20260218_003516/eval_metrics.csv`

- Best `val_loss`: `1.9739` at step `36,400` (`val_ppl=7.1987`)
- Best overall `BLEU`: `2.0214` at step `36,000`
- Best overall `chrF`: `26.0659` at step `36,800`
- Best `BLEU (EN->AR)`: `1.5286` at step `36,000`
- Best `BLEU (AR->EN)`: `2.4680` at step `31,600`
- Best `chrF (EN->AR)`: `28.6933` at step `36,000`
- Best `chrF (AR->EN)`: `24.3222` at step `36,800`

Latest recorded eval point:
- Step `36,800`: `val_loss=1.9931`, `val_ppl=7.3386`, `BLEU=1.8721`, `chrF=26.0659`

## Training Configuration (Implemented)

Model/training setup in `notebooks/02_model_training.ipynb` includes:
- BART-style seq2seq model from scratch (`d_model=512`, `6e/6d`, `8` heads, FFN `2048`)
- Max sequence length: `128`
- Shared ByteLevel BPE tokenizer (`vocab=32k`)
- Direction tokens: `<2ar>`, `<2en>`
- Mixed precision (`FP16`) with `torch.amp.autocast` + `torch.amp.GradScaler`
- Dynamic padding with `DataCollatorForSeq2Seq`
- Optimizer: `AdamW` (`lr=3e-4`, `weight_decay=0.01`, `betas=(0.9, 0.98)`, `eps=1e-8`)
- Scheduler: `get_cosine_with_min_lr_schedule_with_warmup` (`warmup_ratio=0.015`, `min_lr_rate=0.10`)

## Pending Sections

### Test Results (To Fill After Final Test Run)

- Overall BLEU:
- Overall chrF:
- Overall COMET:
- BLEU EN->AR:
- BLEU AR->EN:
- chrF EN->AR:
- chrF AR->EN:
- Evaluation runtime:

### Visual Plots (To Add)

- Training loss vs step
- Validation loss vs step
- BLEU vs step
- chrF vs step
- LR schedule vs step
- Direction-specific BLEU/chrF trends

## Project Structure

```text
en-ar-translation/
|- notebooks/
|  |- 01_dataset_discovery.ipynb
|  |- 02_model_training.ipynb
|- src/
|  |- deep_agents_from_scratch/
|- dataset/
|- artifacts/
|  |- eda/
|  |- tokenizer/
|  |- runs/
|- checkpoints/
|- README.md
```

## Dataset Documentation

### Data Sources (Credits)

1. `local_25k` (Kaggle)  
   https://www.kaggle.com/datasets/tahaalselwii/the-arabic-english-sentence-bank-25k?resource=download
2. `ds2` (Hugging Face)  
   https://huggingface.co/datasets/Arabic-Clip-Archive/ImageCaptions-7M-Translations-Arabic/viewer/default/train?p=1132
3. `ds3` (Hugging Face)  
   https://huggingface.co/datasets/salehalmansour/english-to-arabic-translate
4. `ds4` (Hugging Face)  
   https://huggingface.co/datasets/ammagra/english-arabic-speech-translation
5. `ds5` (Kaggle)  
   https://www.kaggle.com/datasets/yumnagamal/translation-english-arabic?select=merge_df.csv

### Dataset Processing Summary

Pipeline in `notebooks/01_dataset_discovery.ipynb`:
- Load and inspect source schemas
- Standardize to `en/ar`
- Merge sources
- Remove exact duplicate pairs
- Detect and swap language-flipped rows
- Clean boundary newline artifacts
- Remove Arabic diacritics
- Export cleaned dataset and EDA outputs

Final split basis used in training notebook:
- Cleaned base pairs: `827,546`
- Base split: `744,926` train / `41,445` val / `41,175` test
- Bidirectional expansion: `1,489,852` train / `82,890` val / `82,350` test

## Notes

- Truncation in EDA is a proxy; actual model truncation is token-based in notebook training.
- COMET can be enabled/disabled by evaluation configuration; earlier run logs include phases with COMET disabled.
