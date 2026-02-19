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
- Full test-set evaluation completed for the best checkpoint (`best_model`).

## Transformer Architecture

<!-- TODO: Replace with your final architecture image path -->
<img src="docs/images/transformer-architecture.webp" width="350" alt="Transformer Architecture">

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

## Best Validation Metrics

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

### Test Results (Best Checkpoint on Test Split)

Source: `artifacts/runs/20260218_003516/test_metrics_best_model_full.json`

- Split: `test`
- Checkpoint: `best_model`
- Num beams: `3`
- Evaluated samples: `9,600`
- Evaluated batches: `150`
- Overall BLEU: `1.8862`
- Overall chrF: `25.0110`
- Overall COMET: `0.3116`
- BLEU EN->AR: `1.4024`
- BLEU AR->EN: `2.3359`
- chrF EN->AR: `27.9461`
- chrF AR->EN: `23.2507`
- Evaluation runtime: `1.10 h`

### Generated Samples
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>direction</th>
      <th>source_text</th>
      <th>reference_text</th>
      <th>generated_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>ar_to_en</td>
      <td>&lt;2en&gt; "عربة حصان مجرة تقود في شارع منحرف في كندا"</td>
      <td>a horse drawn carriage driving down a cobbled street in canada</td>
      <td>a galactic horse carriage driving down a cliff street in canada royalty illustration stock illustration on a picture of a bird that is being driven by a forested</td>
    </tr>
    <tr>
      <th>2</th>
      <td>en_to_ar</td>
      <td>&lt;2ar&gt; I think optimism is the key to overcoming obstacles.</td>
      <td>أظن أن التفاؤل هو أساس التغلب على العقبات.</td>
      <td>أظن أن التفاؤل هو مفتاح التغلب على العقبات. ‏. ‎.‎.أعتقد أن التفاؤل. ‬.القرار.”.'التغلب على</td>
    </tr>
    <tr>
      <th>3</th>
      <td>en_to_ar</td>
      <td>&lt;2ar&gt; i do hope youre not vegan</td>
      <td>آمل حقا أنك غير نباتية</td>
      <td>أتمنى ألا تكون (فيغان) ؟ !</td>
    </tr>
    <tr>
      <th>4</th>
      <td>en_to_ar</td>
      <td>&lt;2ar&gt; help help me</td>
      <td>النجدة النجدة</td>
      <td>النجدة، النجدة! ساعدوني</td>
    </tr>
    <tr>
      <th>5</th>
      <td>en_to_ar</td>
      <td>&lt;2ar&gt; a police case a friend who needs your help</td>
      <td>قضية شرطة صديق يحتاج لمساعدتك</td>
      <td>حالة شرطة، صديق يحتاج لمساعدتكم. ‏ ‎. ‬ ‫من يريد المساعدة. ‏‏‏‏"مساعدتك_BAR_</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ar_to_en</td>
      <td>&lt;2en&gt; اللوحة من قبل شخص</td>
      <td>the painting by person</td>
      <td>the painting by person person royalty illustration royalty illustration stock illustration royalty image png clipart clipart clipart print displayed on a picture of the painting is shown with a woman in</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ar_to_en</td>
      <td>&lt;2en&gt; أين تعيش والدتك ؟</td>
      <td>where does your mother live</td>
      <td>where do you live your mother and i dont know what it was like to do with her father in the world where is she going to be alive. k</td>
    </tr>
    <tr>
      <th>8</th>
      <td>en_to_ar</td>
      <td>&lt;2ar&gt; thats it</td>
      <td>! إنتهى الأمر</td>
      <td>أهذا كل شيء؟ -أجل . ! ـ نعم! ـ هذا هو... ـ ـ ـ هل أنت بخير؟ ـ ـ شكرا جزيلا لك ؟ ـ أجل.</td>
    </tr>
    <tr>
      <th>9</th>
      <td>ar_to_en</td>
      <td>&lt;2en&gt; ستكون على ما يرام</td>
      <td>youll be fine</td>
      <td>shes gonna be all right  yeah i dont know what you want to do with this guy in the world and then shes going to make a lot of room</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ar_to_en</td>
      <td>&lt;2en&gt; تحتاج أمتنا منكم التصويت اليوم</td>
      <td>our nation needs you to vote today</td>
      <td>our mother of you needs to vote today. kmtlvdwahu mnfi.gh. nb251</td>
    </tr>
    <tr>
      <th>11</th>
      <td>en_to_ar</td>
      <td>&lt;2ar&gt; He who seeks excellence never stops.</td>
      <td>من يسعى للتميز لا يتوقف أبدا.</td>
      <td>من يسعى للتميز لا يتوقف أبدا. ‏ ‎. ‬.”.’.من يسعى إلى التميز. [الأشخاص الذين لا يتوقفون.</td>
    </tr>
  </tbody>
</table>
</div>

### Visual Plots (To Add)

<img src="artifacts\runs\20260218_003516\plots\key_curves_overlay.png" width="350" alt="Evaluation Metrics Plots ">

<img src="artifacts\runs\20260218_003516\plots\eval_metrics_overview.png" width="350" alt="Evaluation Metrics Plots ">

<img src="artifacts\runs\20260218_003516\plots\bleu_chrf.png" width="350" alt="BLEU and chrF plots">

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
