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
