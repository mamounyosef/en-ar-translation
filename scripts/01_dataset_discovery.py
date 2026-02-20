from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from datasets import DatasetDict, load_dataset

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

REQUIRED_COLUMNS = ["en", "ar"]
MAX_SEQ_LEN = 128

# Manual column picks from notebook inspection.
DS2_EN_COL = "caption"
DS2_AR_COL = "caption_multi"
DS3_EN_COL = "en"
DS3_AR_COL = "ar"
DS4_EN_COL = "sentence"
DS4_AR_COL = "translation"
DS5_EN_COL = "english"
DS5_AR_COL = "arabic"

# Row caps.
DS2_MAX_ROWS = None
DS3_MAX_ROWS = 412_000
DS4_MAX_ROWS = None
DS5_MAX_ROWS = None

EN_CANDIDATES = [
    "en",
    "english",
    "eng",
    "source",
    "src",
    "text_en",
    "sentence_en",
    "input",
    "caption",
]
AR_CANDIDATES = [
    "ar",
    "arabic",
    "target",
    "tgt",
    "text_ar",
    "sentence_ar",
    "output",
    "caption_multi",
    "translation",
]


def resolve_project_root() -> Path:
    candidate_roots = [Path.cwd(), Path.cwd().parent]
    return next((r for r in candidate_roots if (r / "dataset").exists()), Path.cwd())


def pick_column(
    column_names: Iterable[str],
    preferred: str | None,
    candidates: list[str],
    side_name: str,
) -> str:
    cols = list(column_names)
    if preferred is not None:
        if preferred not in cols:
            raise ValueError(f"{side_name} column `{preferred}` not found. Available: {cols}")
        return preferred
    for cand in candidates:
        if cand in cols:
            return cand
    raise ValueError(f"Could not infer {side_name} column. Available: {cols}.")


def dataset_dict_to_en_ar(
    ds: DatasetDict,
    preferred_en: str | None,
    preferred_ar: str | None,
    max_rows: int | None,
    name: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    used = 0
    print(f"\n{name} splits: {list(ds.keys())}")

    for split_name, split_ds in ds.items():
        chosen_en = pick_column(split_ds.column_names, preferred_en, EN_CANDIDATES, f"{name} EN")
        chosen_ar = pick_column(split_ds.column_names, preferred_ar, AR_CANDIDATES, f"{name} AR")

        split_df = split_ds.to_pandas()[[chosen_en, chosen_ar]].rename(
            columns={chosen_en: "en", chosen_ar: "ar"}
        )

        if max_rows is not None:
            remaining = max_rows - used
            if remaining <= 0:
                break
            split_df = split_df.head(remaining).copy()

        used += len(split_df)
        frames.append(split_df)
        print(
            f"- {name}[{split_name}] rows={len(split_df):,} "
            f"(en_col={chosen_en}, ar_col={chosen_ar})"
        )

    if not frames:
        raise ValueError(f"No rows collected from {name}.")

    out = pd.concat(frames, ignore_index=True)
    print(f"{name} total rows used: {len(out):,}")
    return out


def dataframe_to_en_ar(
    df_in: pd.DataFrame,
    preferred_en: str | None,
    preferred_ar: str | None,
    max_rows: int | None,
    name: str,
) -> pd.DataFrame:
    chosen_en = pick_column(df_in.columns, preferred_en, EN_CANDIDATES, f"{name} EN")
    chosen_ar = pick_column(df_in.columns, preferred_ar, AR_CANDIDATES, f"{name} AR")
    df_out = df_in[[chosen_en, chosen_ar]].rename(columns={chosen_en: "en", chosen_ar: "ar"}).copy()
    if max_rows is not None:
        df_out = df_out.head(max_rows).copy()
    print(f"{name} rows={len(df_out):,} (en_col={chosen_en}, ar_col={chosen_ar})")
    return df_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert notebook dataset discovery pipeline to script.")
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=MAX_SEQ_LEN,
        help="Reference max sequence length used in truncation proxy reporting.",
    )
    args = parser.parse_args()

    project_root = resolve_project_root()
    local_25k_csv_path = (
        project_root / "dataset" / "The Arabic-English Sentence Bank 25k" / "arabic_english_sentences.csv"
    )
    local_ds5_csv_path = project_root / "dataset" / "translation-english-arabic.csv"
    eda_output_dir = project_root / "artifacts" / "eda"

    print(f"Project root: {project_root}")
    print(f"Local 25k CSV path: {local_25k_csv_path}")
    print(f"Local ds5 CSV path: {local_ds5_csv_path}")
    print(f"EDA output dir: {eda_output_dir}")

    if not local_25k_csv_path.exists():
        raise FileNotFoundError(f"Local 25k CSV not found: {local_25k_csv_path}")
    if not local_ds5_csv_path.exists():
        raise FileNotFoundError(f"Local ds5 CSV not found: {local_ds5_csv_path}")

    print("\nLoading datasets...")
    ds2 = load_dataset("Arabic-Clip-Archive/ImageCaptions-7M-Translations-Arabic")
    ds3 = load_dataset("salehalmansour/english-to-arabic-translate")
    ds4 = load_dataset("ammagra/english-arabic-speech-translation")
    ds5 = pd.read_csv(local_ds5_csv_path, encoding="utf-8")

    print(f"Loaded ds2 splits: {list(ds2.keys())}")
    print(f"Loaded ds3 splits: {list(ds3.keys())}")
    print(f"Loaded ds4 splits: {list(ds4.keys())}")
    print(f"Loaded ds5 rows: {len(ds5):,}")

    print("\nBuilding standardized en/ar frames...")
    df_local_25k = pd.read_csv(local_25k_csv_path, encoding="utf-8")
    df_local_25k = df_local_25k.rename(columns={"English": "en", "Arabic": "ar"})[["en", "ar"]]
    print(f"local_25k rows: {len(df_local_25k):,}")

    df_ds2 = dataset_dict_to_en_ar(ds2, DS2_EN_COL, DS2_AR_COL, DS2_MAX_ROWS, "ds2")
    df_ds3 = dataset_dict_to_en_ar(ds3, DS3_EN_COL, DS3_AR_COL, DS3_MAX_ROWS, "ds3")
    df_ds4 = dataset_dict_to_en_ar(ds4, DS4_EN_COL, DS4_AR_COL, DS4_MAX_ROWS, "ds4")
    df_ds5 = dataframe_to_en_ar(ds5, DS5_EN_COL, DS5_AR_COL, DS5_MAX_ROWS, "ds5")

    source_frames = [
        ("local_25k", df_local_25k),
        ("ds2", df_ds2),
        ("ds3", df_ds3),
        ("ds4", df_ds4),
        ("ds5", df_ds5),
    ]

    for name, frame in source_frames:
        missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"{name} is missing columns: {missing}")

    print("\nMerging and initial dedup...")
    df_merged = pd.concat([f for _, f in source_frames], ignore_index=True)
    rows_before_dedup = len(df_merged)
    unique_pairs_before = df_merged.drop_duplicates(subset=REQUIRED_COLUMNS).shape[0]
    duplicate_pairs_before = rows_before_dedup - unique_pairs_before
    duplicate_ratio_before = duplicate_pairs_before / rows_before_dedup if rows_before_dedup else 0.0

    df = df_merged.drop_duplicates(subset=REQUIRED_COLUMNS, keep="first").reset_index(drop=True)

    print("Rows by source:")
    for name, frame in source_frames:
        print(f"- {name}: {len(frame):,}")
    print(f"Merged rows (before dedup): {rows_before_dedup:,}")
    print(f"Duplicate pairs before dedup: {duplicate_pairs_before:,} ({duplicate_ratio_before:.4%})")
    print(f"Rows after dedup: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    print("\nBasic shape and memory...")
    row_count, col_count = df.shape
    memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
    print(f"Shape: {df.shape}")
    print(f"Row count: {row_count:,}")
    print(f"Column count: {col_count}")
    print(f"Approx memory usage: {memory_mb:,.2f} MB")

    print("\nDetecting swapped-language rows...")
    en_text = df["en"].fillna("").astype(str)
    ar_text = df["ar"].fillna("").astype(str)

    en_has_arabic = en_text.str.contains(r"[\u0600-\u06FF]", regex=True)
    en_has_latin = en_text.str.contains(r"[A-Za-z]", regex=True)
    ar_has_arabic = ar_text.str.contains(r"[\u0600-\u06FF]", regex=True)
    ar_has_latin = ar_text.str.contains(r"[A-Za-z]", regex=True)

    swap_mask = en_has_arabic & ar_has_latin & (~en_has_latin | ~ar_has_arabic)
    swap_count = int(swap_mask.sum())

    if swap_count > 0:
        df.loc[swap_mask, ["en", "ar"]] = df.loc[swap_mask, ["ar", "en"]].values

    rows_before_rededup = len(df)
    df = df.drop_duplicates(subset=REQUIRED_COLUMNS, keep="first").reset_index(drop=True)
    rededup_removed_after_swap = rows_before_rededup - len(df)

    print(f"Detected likely swapped rows: {swap_count:,}")
    print(f"Rows removed after swap re-dedup: {rededup_removed_after_swap:,}")

    print('\nCleaning boundary newline markers and "-"...')
    newline_clean_stats: dict[str, int] = {}
    for col in REQUIRED_COLUMNS:
        before_col = df[col].fillna("").astype(str)
        after_col = (
            before_col.str.replace(r"^(?:\s*\\n\s*|\s*-\s*)+", "", regex=True)
            .str.replace(r"(?:\s*\\n\s*|\s*-\s*)+$", "", regex=True)
            .str.replace(r"^[\r\n\-]+", "", regex=True)
            .str.replace(r"[\r\n\-]+$", "", regex=True)
            .str.strip()
        )
        changed_rows = int((before_col != after_col).sum())
        newline_clean_stats[col] = changed_rows
        df[col] = after_col
    for col, count in newline_clean_stats.items():
        print(f"- {col}: {count:,} rows updated")

    print("\nRemoving Arabic diacritics...")
    arabic_diacritics_pattern = r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"
    before_ar = df["ar"].fillna("").astype(str)
    diacritics_removed_chars = int(before_ar.str.count(arabic_diacritics_pattern).sum())
    after_ar = before_ar.str.replace(arabic_diacritics_pattern, "", regex=True)
    diacritics_changed_rows = int((before_ar != after_ar).sum())
    df["ar"] = after_ar.str.replace(r"\s+", " ", regex=True).str.strip()

    rows_before_rededup = len(df)
    df = df.drop_duplicates(subset=REQUIRED_COLUMNS, keep="first").reset_index(drop=True)
    rededup_removed_after_diacritics = rows_before_rededup - len(df)

    print(f"Rows with Arabic diacritics removed: {diacritics_changed_rows:,}")
    print(f"Total diacritic characters removed: {diacritics_removed_chars:,}")
    print(f"Rows removed after diacritics re-dedup: {rededup_removed_after_diacritics:,}")

    print("\nComputing EDA tables...")
    null_counts = df[REQUIRED_COLUMNS].isna().sum()
    empty_counts = {
        col: int(df[col].fillna("").astype(str).str.strip().eq("").sum()) for col in REQUIRED_COLUMNS
    }
    quality_df = pd.DataFrame(
        {
            "null_count": null_counts,
            "empty_or_whitespace_count": pd.Series(empty_counts),
        }
    )
    print("\nNull/empty checks:")
    print(quality_df)

    pair_count = len(df)
    unique_pair_count = df.drop_duplicates(subset=REQUIRED_COLUMNS).shape[0]
    duplicate_pair_count = pair_count - unique_pair_count
    duplicate_ratio = duplicate_pair_count / pair_count if pair_count else 0.0
    print("\nExact EN-AR duplicate ratio after dedup:")
    print(f"- total pairs: {pair_count:,}")
    print(f"- unique pairs: {unique_pair_count:,}")
    print(f"- duplicate pairs: {duplicate_pair_count:,}")
    print(f"- duplicate ratio: {duplicate_ratio:.4%}")

    unique_en_count = int(df["en"].nunique(dropna=True))
    unique_ar_count = int(df["ar"].nunique(dropna=True))
    print("\nUnique sentence counts:")
    print(f"- Unique EN: {unique_en_count:,} ({unique_en_count / pair_count:.4%})")
    print(f"- Unique AR: {unique_ar_count:,} ({unique_ar_count / pair_count:.4%})")

    en_word_lengths = df["en"].fillna("").astype(str).str.split().str.len()
    ar_word_lengths = df["ar"].fillna("").astype(str).str.split().str.len()
    word_length_stats = pd.DataFrame(
        {
            "en": [
                en_word_lengths.mean(),
                en_word_lengths.quantile(0.50),
                en_word_lengths.quantile(0.90),
                en_word_lengths.quantile(0.95),
                en_word_lengths.quantile(0.99),
            ],
            "ar": [
                ar_word_lengths.mean(),
                ar_word_lengths.quantile(0.50),
                ar_word_lengths.quantile(0.90),
                ar_word_lengths.quantile(0.95),
                ar_word_lengths.quantile(0.99),
            ],
        },
        index=["mean", "p50", "p90", "p95", "p99"],
    ).round(2)

    en_char_lengths = df["en"].fillna("").astype(str).str.len()
    ar_char_lengths = df["ar"].fillna("").astype(str).str.len()
    char_length_stats = pd.DataFrame(
        {
            "en": [
                en_char_lengths.mean(),
                en_char_lengths.quantile(0.50),
                en_char_lengths.quantile(0.90),
                en_char_lengths.quantile(0.95),
                en_char_lengths.quantile(0.99),
                en_char_lengths.max(),
            ],
            "ar": [
                ar_char_lengths.mean(),
                ar_char_lengths.quantile(0.50),
                ar_char_lengths.quantile(0.90),
                ar_char_lengths.quantile(0.95),
                ar_char_lengths.quantile(0.99),
                ar_char_lengths.max(),
            ],
        },
        index=["mean", "p50", "p90", "p95", "p99", "max"],
    ).round(2)

    max_en_idx = en_char_lengths.idxmax()
    max_ar_idx = ar_char_lengths.idxmax()
    outlier_preview = pd.DataFrame(
        [
            {
                "side": "en",
                "index": int(max_en_idx),
                "char_len": int(en_char_lengths.loc[max_en_idx]),
                "text": str(df.loc[max_en_idx, "en"])[:200],
            },
            {
                "side": "ar",
                "index": int(max_ar_idx),
                "char_len": int(ar_char_lengths.loc[max_ar_idx]),
                "text": str(df.loc[max_ar_idx, "ar"])[:200],
            },
        ]
    )

    en_contains_arabic = df["en"].fillna("").astype(str).str.contains(r"[\u0600-\u06FF]", regex=True)
    ar_contains_latin = df["ar"].fillna("").astype(str).str.contains(r"[A-Za-z]", regex=True)
    anomaly_summary = pd.DataFrame(
        {
            "count": [int(en_contains_arabic.sum()), int(ar_contains_latin.sum())],
            "ratio": [float(en_contains_arabic.mean()), float(ar_contains_latin.mean())],
        },
        index=["en_contains_arabic", "ar_contains_latin"],
    )
    anomaly_summary["ratio"] = (anomaly_summary["ratio"] * 100).round(4).astype(str) + "%"

    en_token_proxy = df["en"].fillna("").astype(str).str.split().str.len()
    ar_token_proxy = df["ar"].fillna("").astype(str).str.split().str.len()
    en_trunc_count = int((en_token_proxy > args.max_seq_len).sum())
    ar_trunc_count = int((ar_token_proxy > args.max_seq_len).sum())
    either_trunc_count = int(((en_token_proxy > args.max_seq_len) | (ar_token_proxy > args.max_seq_len)).sum())
    total_rows = len(df)
    truncation_estimate = pd.DataFrame(
        {
            "count": [en_trunc_count, ar_trunc_count, either_trunc_count],
            "ratio": [
                en_trunc_count / total_rows if total_rows else 0.0,
                ar_trunc_count / total_rows if total_rows else 0.0,
                either_trunc_count / total_rows if total_rows else 0.0,
            ],
        },
        index=["en_gt_max_len", "ar_gt_max_len", "either_side_gt_max_len"],
    )
    truncation_estimate["ratio"] = (truncation_estimate["ratio"] * 100).round(4).astype(str) + "%"

    top_en = df["en"].fillna("").astype(str).value_counts().head(10).rename_axis("en").reset_index(name="count")
    top_ar = df["ar"].fillna("").astype(str).value_counts().head(10).rename_axis("ar").reset_index(name="count")
    sample_preview = df.sample(n=10, random_state=RANDOM_SEED)[["en", "ar"]].reset_index(drop=True)

    print("\nWord length stats:")
    print(word_length_stats)
    print("\nCharacter length stats:")
    print(char_length_stats)
    print("\nMax-length outlier preview:")
    print(outlier_preview)
    print("\nScript anomaly summary:")
    print(anomaly_summary)
    print(f"\nTruncation proxy (MAX_SEQ_LEN={args.max_seq_len}):")
    print(truncation_estimate)

    print("\nExporting EDA artifacts...")
    eda_output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "max_seq_len_reference": int(args.max_seq_len),
        "duplicate_pairs_before_dedup": int(duplicate_pairs_before),
        "duplicate_ratio_before_dedup": float(duplicate_ratio_before),
        "swap_count": int(swap_count),
        "rededup_removed_after_swap": int(rededup_removed_after_swap),
        "diacritics_removed_chars": int(diacritics_removed_chars),
    }
    pd.Series(summary).to_json(eda_output_dir / "eda_summary.json", force_ascii=False, indent=2)

    word_length_stats.to_csv(eda_output_dir / "word_length_stats.csv", encoding="utf-8")
    char_length_stats.to_csv(eda_output_dir / "char_length_stats.csv", encoding="utf-8")
    anomaly_summary.to_csv(eda_output_dir / "anomaly_summary.csv", encoding="utf-8")
    truncation_estimate.to_csv(eda_output_dir / "truncation_estimate.csv", encoding="utf-8")
    top_en.to_csv(eda_output_dir / "top_en.csv", index=False, encoding="utf-8")
    top_ar.to_csv(eda_output_dir / "top_ar.csv", index=False, encoding="utf-8")
    sample_preview.to_csv(eda_output_dir / "sample_preview.csv", index=False, encoding="utf-8")
    print(f"Saved compact EDA artifacts to: {eda_output_dir}")

    full_dataset_parquet_path = eda_output_dir / "final_cleaned_combined_dataset.parquet"
    full_dataset_csv_path = eda_output_dir / "final_cleaned_combined_dataset.csv"
    metadata_path = eda_output_dir / "final_cleaned_combined_dataset_metadata.json"

    df.to_parquet(full_dataset_parquet_path, index=False)
    df.to_csv(full_dataset_csv_path, index=False, encoding="utf-8")

    metadata = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "parquet_path": str(full_dataset_parquet_path),
        "csv_path": str(full_dataset_csv_path),
        "max_seq_len_reference": int(args.max_seq_len),
    }
    pd.Series(metadata).to_json(metadata_path, force_ascii=False, indent=2)

    print(f"\nSaved parquet: {full_dataset_parquet_path}")
    print(f"Saved csv: {full_dataset_csv_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Final cleaned rows: {len(df):,}")


if __name__ == "__main__":
    main()
