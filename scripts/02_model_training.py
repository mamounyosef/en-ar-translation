from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizerFast,
    get_cosine_with_min_lr_schedule_with_warmup,
)


SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<unk>", "<2ar>", "<2en>"]
REQUIRED_COLUMNS = ["en", "ar"]


@dataclass
class Config:
    # Data/tokenization
    random_seed: int = 42
    max_seq_len: int = 128
    vocab_size: int = 32_000
    train_ratio: float = 0.90
    val_ratio: float = 0.05
    test_ratio: float = 0.05

    # Training (fixed as requested)
    max_steps: int = 32_000
    per_device_train_batch_size: int = 24
    gradient_accumulation_steps: int = 32

    # Optimizer/scheduler
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    adam_betas: tuple[float, float] = (0.9, 0.98)
    adam_eps: float = 1e-8
    warmup_ratio: float = 0.015
    min_lr_rate: float = 0.10

    # Runtime/logging/checkpoints
    use_gradient_checkpointing: bool = False
    use_torch_compile: bool = False
    train_num_workers: int = 2
    val_num_workers: int = 2
    pin_memory: bool = True
    log_every_steps: int = 40
    eval_every_steps: int = 400
    save_every_steps: int = 500
    keep_last_n_checkpoints: int | None = 10
    max_val_batches: int | None = 300
    text_metric_num_beams: int = 1
    max_text_metric_batches_log: int | None = 4
    max_text_metric_batches_eval: int | None = 32
    enable_comet: bool = False


def resolve_project_root() -> Path:
    roots = [Path.cwd(), Path.cwd().parent]
    return next((r for r in roots if (r / "artifacts").exists()), Path.cwd())


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loss_to_ppl(loss: float) -> float:
    if loss is None or not np.isfinite(loss):
        return float("nan")
    return float(math.exp(min(float(loss), 20.0)))


def load_text_metrics(enable_comet: bool = False):
    try:
        import evaluate
    except ImportError as e:
        raise ImportError("Install metrics deps: pip install evaluate sacrebleu") from e

    bleu_metric = evaluate.load("sacrebleu")
    chrf_metric = evaluate.load("chrf")
    comet_metric = None
    if enable_comet:
        try:
            comet_metric = evaluate.load("comet")
        except Exception as ex:  # noqa: BLE001
            print(f"[warn] COMET unavailable, continuing without it. reason={ex}")
    return bleu_metric, chrf_metric, comet_metric


def eval_val_loss(
    model: BartForConditionalGeneration,
    val_loader: DataLoader,
    device: torch.device,
    use_fp16: bool,
    max_batches: int | None = None,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if max_batches is not None and i >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            if use_fp16:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(**batch)
            else:
                out = model(**batch)
            losses.append(float(out.loss.detach().cpu().item()))
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def eval_text_metrics(
    model: BartForConditionalGeneration,
    tokenizer: PreTrainedTokenizerFast,
    eval_df: pd.DataFrame,
    device: torch.device,
    batch_size: int,
    max_batches: int | None,
    num_beams: int,
    max_seq_len: int,
    seed: int,
    bleu_metric,
    chrf_metric,
    comet_metric=None,
) -> dict[str, float | int]:
    empty = {
        "num_samples": 0,
        "num_batches": 0,
        "bleu": float("nan"),
        "chrf": float("nan"),
        "comet": float("nan"),
        "bleu_en_to_ar": float("nan"),
        "bleu_ar_to_en": float("nan"),
        "chrf_en_to_ar": float("nan"),
        "chrf_ar_to_en": float("nan"),
    }
    if eval_df is None or len(eval_df) == 0:
        return empty

    if max_batches is None:
        sample_df = eval_df.reset_index(drop=True)
    else:
        max_samples = min(int(max_batches * batch_size), len(eval_df))
        if max_samples <= 0:
            return empty
        sample_df = eval_df.sample(n=max_samples, random_state=seed).reset_index(drop=True)

    model.eval()
    preds, refs, srcs, dirs = [], [], [], []
    total_batches = math.ceil(len(sample_df) / batch_size)
    with torch.no_grad():
        for start in tqdm(range(0, len(sample_df), batch_size), total=total_batches, desc="Generating", unit="batch"):
            chunk = sample_df.iloc[start : start + batch_size]
            src_batch = chunk["source_text"].astype(str).tolist()
            ref_batch = chunk["target_text"].astype(str).tolist()
            dir_batch = chunk["direction"].astype(str).tolist()

            enc = tokenizer(
                src_batch,
                truncation=True,
                max_length=max_seq_len,
                padding=True,
                return_token_type_ids=False,
                return_tensors="pt",
            )
            enc.pop("token_type_ids", None)
            enc = {k: v.to(device) for k, v in enc.items()}

            gen_ids = model.generate(
                **enc,
                max_new_tokens=max_seq_len,
                num_beams=num_beams,
                do_sample=False,
            )
            pred_batch = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            preds.extend([p.strip() for p in pred_batch])
            refs.extend([r.strip() for r in ref_batch])
            srcs.extend(src_batch)
            dirs.extend(dir_batch)
    model.train()

    if len(preds) == 0:
        return empty

    def score_subset(_preds: list[str], _refs: list[str]) -> tuple[float, float]:
        if len(_preds) == 0:
            return float("nan"), float("nan")
        bleu = float(bleu_metric.compute(predictions=_preds, references=[[r] for r in _refs])["score"])
        chrf = float(chrf_metric.compute(predictions=_preds, references=_refs)["score"])
        return bleu, chrf

    bleu_all, chrf_all = score_subset(preds, refs)
    idx_en_to_ar = [i for i, d in enumerate(dirs) if d == "en_to_ar"]
    idx_ar_to_en = [i for i, d in enumerate(dirs) if d == "ar_to_en"]
    bleu_en_to_ar, chrf_en_to_ar = score_subset([preds[i] for i in idx_en_to_ar], [refs[i] for i in idx_en_to_ar])
    bleu_ar_to_en, chrf_ar_to_en = score_subset([preds[i] for i in idx_ar_to_en], [refs[i] for i in idx_ar_to_en])

    comet = float("nan")
    if comet_metric is not None:
        try:
            out = comet_metric.compute(predictions=preds, references=refs, sources=srcs)
            comet = float(out.get("mean_score", out.get("score", float("nan"))))
        except Exception as ex:  # noqa: BLE001
            print(f"[warn] COMET failed; continuing. reason={ex}")

    return {
        "num_samples": int(len(preds)),
        "num_batches": int(math.ceil(len(preds) / batch_size)),
        "bleu": bleu_all,
        "chrf": chrf_all,
        "comet": comet,
        "bleu_en_to_ar": bleu_en_to_ar,
        "bleu_ar_to_en": bleu_ar_to_en,
        "chrf_en_to_ar": chrf_en_to_ar,
        "chrf_ar_to_en": chrf_ar_to_en,
    }


def save_checkpoint(
    model: BartForConditionalGeneration,
    tokenizer: PreTrainedTokenizerFast,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    step: int,
    val_loss: float | None,
    target_dir: Path,
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(target_dir))
    tokenizer.save_pretrained(str(target_dir / "tokenizer"))
    torch.save(
        {
            "step": int(step),
            "val_loss": float(val_loss) if val_loss is not None else None,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
        },
        target_dir / "trainer_state.pt",
    )


def main() -> None:
    cfg = Config()
    set_seed(cfg.random_seed)

    project_root = resolve_project_root()
    data_path = project_root / "artifacts" / "eda" / "final_cleaned_combined_dataset.parquet"
    tokenizer_dir = project_root / "artifacts" / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = tokenizer_dir / "en_ar_bpe_tokenizer.json"
    hf_tokenizer_dir = tokenizer_dir / "hf_tokenizer"
    hf_tokenizer_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project root: {project_root}")
    print(f"Dataset path: {data_path}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # 1) Load cleaned dataset and basic row cleaning.
    if not data_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {data_path}")
    df = pd.read_parquet(data_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    rows_before = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS).copy()
    df["en"] = df["en"].astype(str).str.strip()
    df["ar"] = df["ar"].astype(str).str.strip()
    df = df[(df["en"] != "") & (df["ar"] != "")].reset_index(drop=True)
    print(f"Rows before cleaning: {rows_before:,} | after: {len(df):,}")

    # 2) Deterministic split (90/5/5) with leakage guard.
    if not math.isclose(cfg.train_ratio + cfg.val_ratio + cfg.test_ratio, 1.0, abs_tol=1e-9):
        raise ValueError("Split ratios must sum to 1.0")
    pair_hash = pd.util.hash_pandas_object(df[["en", "ar"]], index=False).astype("uint64")
    u = pair_hash / np.float64(2**64)
    train_cut = cfg.train_ratio
    val_cut = cfg.train_ratio + cfg.val_ratio
    df["split"] = np.where(u < train_cut, "train", np.where(u < val_cut, "val", "test"))
    leak_count = int((df.groupby(["en", "ar"])["split"].nunique() > 1).sum())
    if leak_count != 0:
        raise ValueError(f"Leakage detected across splits for {leak_count} pairs")

    # 3) Bidirectional rows with direction tokens.
    df_bi = pd.concat(
        [
            pd.DataFrame(
                {
                    "source_text": "<2ar> " + df["en"],
                    "target_text": df["ar"],
                    "direction": "en_to_ar",
                    "split": df["split"],
                }
            ),
            pd.DataFrame(
                {
                    "source_text": "<2en> " + df["ar"],
                    "target_text": df["en"],
                    "direction": "ar_to_en",
                    "split": df["split"],
                }
            ),
        ],
        ignore_index=True,
    )
    train_df = df_bi[df_bi["split"] == "train"].reset_index(drop=True)
    val_df = df_bi[df_bi["split"] == "val"].reset_index(drop=True)
    test_df = df_bi[df_bi["split"] == "test"].reset_index(drop=True)
    print(f"Bidirectional splits => train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")

    # 4) Train ByteLevel BPE tokenizer (train split only).
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=cfg.vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(
        (text for row in train_df.itertuples(index=False) for text in (row.source_text, row.target_text)),
        trainer=trainer,
    )
    tokenizer.save(str(tokenizer_path))

    # 5) Build Hugging Face fast tokenizer.
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path),
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        additional_special_tokens=["<2ar>", "<2en>"],
    )
    hf_tokenizer.save_pretrained(str(hf_tokenizer_dir))

    # 6) Tokenize splits (truncation only, no static padding).
    def tokenize_batch(batch):
        src = hf_tokenizer(batch["source_text"], truncation=True, max_length=cfg.max_seq_len, padding=False)
        tgt = hf_tokenizer(batch["target_text"], truncation=True, max_length=cfg.max_seq_len, padding=False)
        src["labels"] = tgt["input_ids"]
        return src

    train_ds = Dataset.from_pandas(train_df[["source_text", "target_text", "direction", "split"]], preserve_index=False)
    val_ds = Dataset.from_pandas(val_df[["source_text", "target_text", "direction", "split"]], preserve_index=False)
    test_ds = Dataset.from_pandas(test_df[["source_text", "target_text", "direction", "split"]], preserve_index=False)
    train_tok = train_ds.map(tokenize_batch, batched=True, desc="Tokenizing train")
    val_tok = val_ds.map(tokenize_batch, batched=True, desc="Tokenizing val")
    test_tok = test_ds.map(tokenize_batch, batched=True, desc="Tokenizing test")
    _ = test_tok  # kept for parity; not used in full-training loop.

    model_input_cols = ["input_ids", "attention_mask", "labels"]
    train_tok_model = train_tok.remove_columns([c for c in train_tok.column_names if c not in model_input_cols])
    val_tok_model = val_tok.remove_columns([c for c in val_tok.column_names if c not in model_input_cols])
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=hf_tokenizer,
        model=None,
        padding="longest",
        label_pad_token_id=-100,
    )

    # 7) Build model and run one forward sanity batch.
    bart_config = BartConfig(
        vocab_size=hf_tokenizer.vocab_size,
        max_position_embeddings=cfg.max_seq_len + 2,
        d_model=512,
        encoder_layers=6,
        decoder_layers=6,
        encoder_attention_heads=8,
        decoder_attention_heads=8,
        encoder_ffn_dim=2048,
        decoder_ffn_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.0,
        pad_token_id=hf_tokenizer.pad_token_id,
        bos_token_id=hf_tokenizer.bos_token_id,
        eos_token_id=hf_tokenizer.eos_token_id,
        decoder_start_token_id=hf_tokenizer.bos_token_id,
    )
    model = BartForConditionalGeneration(bart_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    sanity_n = min(8, len(train_tok_model))
    sanity_ds = train_tok_model.shuffle(seed=cfg.random_seed).select(range(sanity_n))
    sanity_batch = data_collator([sanity_ds[i] for i in range(len(sanity_ds))])
    sanity_batch = {k: v.to(device) for k, v in sanity_batch.items()}
    sanity_out = model(**sanity_batch)
    sanity_loss = float(sanity_out.loss.detach().cpu().item())
    if not np.isfinite(sanity_loss):
        raise ValueError(f"Non-finite loss in forward sanity check: {sanity_loss}")
    print(f"Forward sanity OK | loss={sanity_loss:.6f}")

    # 8) Optimizer/scheduler/runtime config.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=cfg.adam_betas,
        eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
    )
    num_warmup_steps = int(cfg.max_steps * cfg.warmup_ratio)
    lr_scheduler = get_cosine_with_min_lr_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=cfg.max_steps,
        min_lr_rate=cfg.min_lr_rate,
    )
    if cfg.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()
    use_fp16 = torch.cuda.is_available()
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    if torch.cuda.is_available() and hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if cfg.use_torch_compile:
        try:
            model = torch.compile(model, backend="eager", dynamic=True)
            print("torch.compile enabled")
        except Exception as ex:  # noqa: BLE001
            print(f"torch.compile failed, continuing without it. reason={ex}")

    # 9) Dataloaders.
    train_loader = DataLoader(
        train_tok_model,
        batch_size=cfg.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=cfg.train_num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=(cfg.train_num_workers > 0),
    )
    val_loader = DataLoader(
        val_tok_model,
        batch_size=cfg.per_device_train_batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=cfg.val_num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=(cfg.val_num_workers > 0),
    )

    # -------------------------------------------------------------
    # Optional smoke run (kept commented out exactly as requested).
    # -------------------------------------------------------------
    # smoke_steps = 50
    # ... run a short loop before full training ...

    # 10) Run directories and config.
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = project_root / "artifacts" / "runs" / run_id
    ckpt_root = project_root / "checkpoints" / run_id
    best_dir = ckpt_root / "best_model"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)
    train_metrics_path = run_dir / "train_metrics.csv"
    eval_metrics_path = run_dir / "eval_metrics.csv"
    run_config_path = run_dir / "run_config.json"

    run_config = {
        **asdict(cfg),
        "run_id": run_id,
        "data_path": str(data_path),
        "train_rows_bidir": int(len(train_df)),
        "val_rows_bidir": int(len(val_df)),
        "test_rows_bidir": int(len(test_df)),
        "num_warmup_steps": int(num_warmup_steps),
        "effective_batch_size": int(cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps),
        "use_fp16": bool(use_fp16),
    }
    run_config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    bleu_metric, chrf_metric, comet_metric = load_text_metrics(enable_comet=cfg.enable_comet)

    # 11) Full training loop.
    model.train()
    optimizer.zero_grad(set_to_none=True)
    train_metrics_log = []
    eval_metrics_log = []
    best_val_loss = float("inf")
    optimizer_steps_done = 0
    micro_step = 0
    interval_start = time.time()
    start_time = interval_start
    interval_raw_loss_sum = 0.0
    interval_micro_steps = 0
    interval_opt_steps = 0
    saved_ckpts: list[Path] = []

    print(f"Starting full training run: {run_id}")
    while optimizer_steps_done < cfg.max_steps:
        for batch in train_loader:
            if optimizer_steps_done >= cfg.max_steps:
                break

            micro_step += 1
            batch = {k: v.to(device) for k, v in batch.items()}

            if use_fp16:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(**batch)
                    scaled_loss = out.loss / cfg.gradient_accumulation_steps
                grad_scaler.scale(scaled_loss).backward()
            else:
                out = model(**batch)
                scaled_loss = out.loss / cfg.gradient_accumulation_steps
                scaled_loss.backward()

            raw_loss = float(out.loss.detach().cpu().item())
            interval_raw_loss_sum += raw_loss
            interval_micro_steps += 1

            if micro_step % cfg.gradient_accumulation_steps == 0:
                if use_fp16:
                    grad_scaler.unscale_(optimizer)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).detach().cpu().item())
                if use_fp16:
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                optimizer_steps_done += 1
                interval_opt_steps += 1

                current_lr = float(optimizer.param_groups[0]["lr"])
                examples_seen = int(optimizer_steps_done * cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps)
                approx_epochs = examples_seen / len(train_df) if len(train_df) else float("nan")

                if optimizer_steps_done % cfg.log_every_steps == 0 or optimizer_steps_done == 1:
                    interval_elapsed = time.time() - interval_start
                    avg_interval_loss = interval_raw_loss_sum / max(1, interval_micro_steps)
                    train_ppl = loss_to_ppl(avg_interval_loss)
                    steps_per_sec = interval_opt_steps / interval_elapsed if interval_elapsed > 0 else 0.0
                    sec_per_step = 1.0 / steps_per_sec if steps_per_sec > 0 else float("inf")

                    mem_alloc_gb, mem_peak_gb = None, None
                    if torch.cuda.is_available():
                        mem_alloc_gb = torch.cuda.memory_allocated(device) / (1024**3)
                        mem_peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

                    text_t0 = time.time()
                    text_metrics_log = eval_text_metrics(
                        model=model,
                        tokenizer=hf_tokenizer,
                        eval_df=val_df,
                        device=device,
                        batch_size=cfg.per_device_train_batch_size,
                        max_batches=cfg.max_text_metric_batches_log,
                        num_beams=cfg.text_metric_num_beams,
                        max_seq_len=cfg.max_seq_len,
                        seed=cfg.random_seed + optimizer_steps_done,
                        bleu_metric=bleu_metric,
                        chrf_metric=chrf_metric,
                        comet_metric=comet_metric,
                    )
                    text_sec = time.time() - text_t0

                    print(
                        f"step {optimizer_steps_done:>6}/{cfg.max_steps} | avg_loss={avg_interval_loss:.4f} | "
                        f"train_ppl={train_ppl:.4f} | lr={current_lr:.6g} | grad_norm={grad_norm:.4f}"
                    )
                    print(
                        f"interval_sec={interval_elapsed:.2f} | steps/sec={steps_per_sec:.4f} | "
                        f"sec/step={sec_per_step:.4f} | examples_seen={examples_seen:,} | approx_epochs={approx_epochs:.4f}"
                    )
                    print(
                        f"[text@log] samples={text_metrics_log['num_samples']} | bleu={text_metrics_log['bleu']:.3f} | "
                        f"chrf={text_metrics_log['chrf']:.3f} | comet={text_metrics_log['comet']:.3f} | "
                        f"text_eval_sec={text_sec:.2f}"
                    )

                    train_metrics_log.append(
                        {
                            "step": int(optimizer_steps_done),
                            "avg_interval_loss": float(avg_interval_loss),
                            "train_ppl": float(train_ppl),
                            "lr": current_lr,
                            "grad_norm": grad_norm,
                            "interval_sec": float(interval_elapsed),
                            "steps_per_sec": float(steps_per_sec),
                            "sec_per_step": float(sec_per_step),
                            "examples_seen": int(examples_seen),
                            "approx_epochs": float(approx_epochs),
                            "batch_input_len": int(batch["input_ids"].shape[1]),
                            "batch_label_len": int(batch["labels"].shape[1]),
                            "cuda_mem_alloc_gb": float(mem_alloc_gb) if mem_alloc_gb is not None else None,
                            "cuda_mem_peak_gb": float(mem_peak_gb) if mem_peak_gb is not None else None,
                            "text_eval_sec": float(text_sec),
                            "text_samples": int(text_metrics_log["num_samples"]),
                            "text_batches": int(text_metrics_log["num_batches"]),
                            "bleu": float(text_metrics_log["bleu"]),
                            "chrf": float(text_metrics_log["chrf"]),
                            "comet": float(text_metrics_log["comet"]),
                            "bleu_en_to_ar": float(text_metrics_log["bleu_en_to_ar"]),
                            "bleu_ar_to_en": float(text_metrics_log["bleu_ar_to_en"]),
                            "chrf_en_to_ar": float(text_metrics_log["chrf_en_to_ar"]),
                            "chrf_ar_to_en": float(text_metrics_log["chrf_ar_to_en"]),
                        }
                    )
                    pd.DataFrame(train_metrics_log).to_csv(train_metrics_path, index=False)
                    interval_start = time.time()
                    interval_raw_loss_sum = 0.0
                    interval_micro_steps = 0
                    interval_opt_steps = 0

                if optimizer_steps_done % cfg.eval_every_steps == 0 or optimizer_steps_done == cfg.max_steps:
                    val_loss = eval_val_loss(model, val_loader, device, use_fp16, max_batches=cfg.max_val_batches)
                    val_ppl = loss_to_ppl(val_loss)
                    text_eval_t0 = time.time()
                    text_metrics_eval = eval_text_metrics(
                        model=model,
                        tokenizer=hf_tokenizer,
                        eval_df=val_df,
                        device=device,
                        batch_size=cfg.per_device_train_batch_size,
                        max_batches=cfg.max_text_metric_batches_eval,
                        num_beams=cfg.text_metric_num_beams,
                        max_seq_len=cfg.max_seq_len,
                        seed=cfg.random_seed + 10_000 + optimizer_steps_done,
                        bleu_metric=bleu_metric,
                        chrf_metric=chrf_metric,
                        comet_metric=comet_metric,
                    )
                    text_eval_sec = time.time() - text_eval_t0
                    print(
                        f"[eval] step={optimizer_steps_done} val_loss={val_loss:.4f} | val_ppl={val_ppl:.4f} | "
                        f"bleu={text_metrics_eval['bleu']:.3f} | chrf={text_metrics_eval['chrf']:.3f} | "
                        f"comet={text_metrics_eval['comet']:.3f} | text_eval_sec={text_eval_sec:.2f}"
                    )
                    eval_metrics_log.append(
                        {
                            "step": int(optimizer_steps_done),
                            "val_loss": float(val_loss),
                            "val_ppl": float(val_ppl),
                            "lr": current_lr,
                            "text_eval_sec": float(text_eval_sec),
                            "text_samples": int(text_metrics_eval["num_samples"]),
                            "text_batches": int(text_metrics_eval["num_batches"]),
                            "bleu": float(text_metrics_eval["bleu"]),
                            "chrf": float(text_metrics_eval["chrf"]),
                            "comet": float(text_metrics_eval["comet"]),
                            "bleu_en_to_ar": float(text_metrics_eval["bleu_en_to_ar"]),
                            "bleu_ar_to_en": float(text_metrics_eval["bleu_ar_to_en"]),
                            "chrf_en_to_ar": float(text_metrics_eval["chrf_en_to_ar"]),
                            "chrf_ar_to_en": float(text_metrics_eval["chrf_ar_to_en"]),
                        }
                    )
                    pd.DataFrame(eval_metrics_log).to_csv(eval_metrics_path, index=False)

                    if np.isfinite(val_loss) and val_loss < best_val_loss:
                        best_val_loss = float(val_loss)
                        save_checkpoint(
                            model=model,
                            tokenizer=hf_tokenizer,
                            optimizer=optimizer,
                            scheduler=lr_scheduler,
                            scaler=grad_scaler,
                            step=optimizer_steps_done,
                            val_loss=val_loss,
                            target_dir=best_dir,
                        )
                        print(f"[best] new best val_loss={best_val_loss:.4f} saved to {best_dir}")

                if optimizer_steps_done % cfg.save_every_steps == 0 or optimizer_steps_done == cfg.max_steps:
                    ckpt_dir = ckpt_root / f"step_{optimizer_steps_done:06d}"
                    save_checkpoint(
                        model=model,
                        tokenizer=hf_tokenizer,
                        optimizer=optimizer,
                        scheduler=lr_scheduler,
                        scaler=grad_scaler,
                        step=optimizer_steps_done,
                        val_loss=None,
                        target_dir=ckpt_dir,
                    )
                    saved_ckpts.append(ckpt_dir)
                    print(f"[ckpt] saved: {ckpt_dir}")
                    if cfg.keep_last_n_checkpoints is not None and cfg.keep_last_n_checkpoints > 0:
                        while len(saved_ckpts) > cfg.keep_last_n_checkpoints:
                            old = saved_ckpts.pop(0)
                            if old.exists():
                                shutil.rmtree(old)
                                print(f"[ckpt] removed old checkpoint: {old}")

    total_elapsed = time.time() - start_time
    print("=" * 100)
    print(f"Training finished at step {optimizer_steps_done}/{cfg.max_steps}")
    print(f"Total elapsed sec: {total_elapsed:.2f}")
    print(f"Best val loss: {best_val_loss if np.isfinite(best_val_loss) else None}")
    print(f"Train metrics CSV: {train_metrics_path}")
    print(f"Eval metrics CSV: {eval_metrics_path}")
    print(f"Best model dir: {best_dir}")


if __name__ == "__main__":
    main()
