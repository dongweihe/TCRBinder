#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import json
from typing import List, Optional, Tuple
from transformers.utils import logging
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, BertTokenizer


INPUT_CSV = "./PHT.csv"

ALPHA_COL = "TCR_Alpha"
BETA_COL  = "TCR_Beta"
ID_COL    = "ID"

ENCODE_ALPHA = True
ENCODE_BETA  = True

BETA_DIR  = "./Result_beta/checkpoints/Pretrain/0518_125864"
ALPHA_DIR = "./Result_alpha/checkpoints/Pretrain/0518_029163"

TOKENIZER_DIR = BETA_DIR

OUT_PREFIX = "./Standalone_TCR_Embeddings/tcr_embeddings"

MAX_LENGTH = 120
BATCH_SIZE = 64
POOLING = "cls"
DROP_DUPLICATES_BETA = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FP16 = True

SAVE_PAIRED = True


def ensure_dir(path_prefix: str):

    parent = os.path.dirname(os.path.abspath(path_prefix))
    if parent and (not os.path.exists(parent)):
        os.makedirs(parent, exist_ok=True)


def normalize_seq(x) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if (not s) or (s.lower() == "nan"):
        return None
    return s


def build_tokenizer(tokenizer_dir: str) -> BertTokenizer:
    tok = BertTokenizer.from_pretrained(
        tokenizer_dir,
        do_lower_case=False,
        do_basic_tokenize=True,
        tokenize_chinese_chars=False,
        pad_token="$",
        mask_token=".",
        unk_token="?",
        sep_token="|",
        cls_token="*",
        padding_side="right",
    )
    return tok


def load_encoder(model_dir: str, device: torch.device) -> AutoModel:
    model = AutoModel.from_pretrained(model_dir, return_dict=True)
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def encode_batch(
    seqs: List[str],
    tokenizer: BertTokenizer,
    model: AutoModel,
    device: torch.device,
    max_length: int,
    pooling: str,
    fp16: bool
) -> np.ndarray:
    inputs = tokenizer(
        seqs,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    if (device.type == "cuda") and fp16:
        with torch.cuda.amp.autocast(dtype=torch.float16):
            out = model(**inputs)
    else:
        out = model(**inputs)

    hs = out.last_hidden_state  # (B, L, D)

    if pooling == "cls":
        return hs[:, 0, :].detach().cpu().numpy()

    # mean pooling
    attn = inputs["attention_mask"].unsqueeze(-1).type_as(hs)
    summed = (hs * attn).sum(dim=1)
    denom = attn.sum(dim=1).clamp(min=1.0)
    return (summed / denom).detach().cpu().numpy()


def encode_column(
    df: pd.DataFrame,
    col: str,
    tokenizer: BertTokenizer,
    model: AutoModel,
    device: torch.device,
    max_length: int,
    batch_size: int,
    pooling: str,
    fp16: bool
) -> Tuple[np.ndarray, List[Optional[str]]]:
    seqs_raw = df[col].tolist()
    seqs = [normalize_seq(s) for s in seqs_raw]

    keep_idx = [i for i, s in enumerate(seqs) if s is not None]
    if len(keep_idx) == 0:
        raise ValueError(f"{col} no seq")

    keep_seqs = [seqs[i] for i in keep_idx]

    embs = []
    for st in range(0, len(keep_seqs), batch_size):
        chunk = keep_seqs[st:st + batch_size]
        embs.append(encode_batch(chunk, tokenizer, model, device, max_length, pooling, fp16))

    keep_emb = np.vstack(embs).astype(np.float32)
    D = keep_emb.shape[1]

    full = np.zeros((len(df), D), dtype=np.float32)
    full[keep_idx, :] = keep_emb
    return full, seqs


def main():
    if POOLING not in ("cls", "mean"):
        raise ValueError("POOLING only 'cls' or 'mean'")
    if (not ENCODE_ALPHA) and (not ENCODE_BETA):
        raise ValueError("ENCODE_ALPHA and ENCODE_BETA Cant simultaneously be False")
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input file not found：{INPUT_CSV}")

    ensure_dir(OUT_PREFIX)
    device = torch.device(DEVICE)

    df = pd.read_csv(INPUT_CSV)

    ids = df[ID_COL].astype(str).tolist() if (ID_COL in df.columns) else [str(i) for i in range(len(df))]
    meta = pd.DataFrame({"row_index": np.arange(len(df)), "ID": ids})

    tokenizer = build_tokenizer(TOKENIZER_DIR)

    alpha_emb = None
    beta_emb = None

    if ENCODE_ALPHA:
        if ALPHA_COL not in df.columns:
            raise ValueError(f"missing column：{ALPHA_COL}")
        alpha_model = load_encoder(ALPHA_DIR, device)
        alpha_emb, alpha_seqs = encode_column(df, ALPHA_COL, tokenizer, alpha_model, device, MAX_LENGTH, BATCH_SIZE, POOLING, FP16)
        np.save(OUT_PREFIX + ".alpha.npy", alpha_emb)
        meta[ALPHA_COL] = [s if s is not None else "" for s in alpha_seqs]

    if ENCODE_BETA:
        if BETA_COL not in df.columns:
            raise ValueError(f"missing column：{BETA_COL}")
        beta_model = load_encoder(BETA_DIR, device)
        beta_emb, beta_seqs = encode_column(df, BETA_COL, tokenizer, beta_model, device, MAX_LENGTH, BATCH_SIZE, POOLING, FP16)
        np.save(OUT_PREFIX + ".beta.npy", beta_emb)
        meta[BETA_COL] = [s if s is not None else "" for s in beta_seqs]

    if SAVE_PAIRED:
        if (alpha_emb is None) or (beta_emb is None):
            raise ValueError("SAVE_PAIRED=True，Alpha and Beta must be encoded")
        paired = np.concatenate([alpha_emb, beta_emb], axis=1).astype(np.float32)
        np.save(OUT_PREFIX + ".paired.npy", paired)

    meta.to_csv(OUT_PREFIX + ".meta.csv", index=False)

    run_cfg = {
        "INPUT_CSV": INPUT_CSV,
        "OUT_PREFIX": OUT_PREFIX,
        "ALPHA_DIR": ALPHA_DIR,
        "BETA_DIR": BETA_DIR,
        "TOKENIZER_DIR": TOKENIZER_DIR,
        "ALPHA_COL": ALPHA_COL,
        "BETA_COL": BETA_COL,
        "ID_COL": ID_COL,
        "MAX_LENGTH": MAX_LENGTH,
        "BATCH_SIZE": BATCH_SIZE,
        "POOLING": POOLING,
        "DEVICE": str(device),
        "FP16": FP16,
        "DROP_DUPLICATES_BETA": DROP_DUPLICATES_BETA,
        "ENCODE_ALPHA": ENCODE_ALPHA,
        "ENCODE_BETA": ENCODE_BETA,
        "SAVE_PAIRED": SAVE_PAIRED,
        "n_rows_after_processing": int(len(df)),
    }
    with open(OUT_PREFIX + ".run.json", "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, ensure_ascii=False, indent=2)
    print("Saved:")
    if ENCODE_ALPHA:
        print(" ", OUT_PREFIX + ".alpha.npy")
    if ENCODE_BETA:
        print(" ", OUT_PREFIX + ".beta.npy")
    if SAVE_PAIRED:
        print(" ", OUT_PREFIX + ".paired.npy")
    print(" ", OUT_PREFIX + ".meta.csv")
    print(" ", OUT_PREFIX + ".run.json")


if __name__ == "__main__":
    main()