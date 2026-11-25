import os
import glob
import json
import random
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# 0. Configuration
# ============================================================

DEFAULT_RUN_DIR = "/root/vicuna/311_VS_gap"
DEFAULT_ENERGY_FILE = "/root/vicuna/311_VS_gap/energy/energy"
DEFAULT_MODEL_PATH = "/root/vicuna/models/vicuna-7b-v1.5"
DEFAULT_OUTPUT_ROOT = "/root/vicuna/311_VS_gap/outputs_vicuna_six_energy_gaps"

SEED = 42
SEQ_LEN = 16
TRAIN_RATIO = 0.15
BATCH_SIZE = 32
GRAD_ACCUM = 1

EPOCHS = 1000
LR = 2.0e-4
WEIGHT_DECAY = 1.0e-4
SMOOTH_L1_BETA = 0.5
PATIENCE = 200
MIN_DELTA = 1.0e-7

EXPECTED_ENERGY_DIM = 4
OUT_DIM = 6
UNIT = "eV"

GAP_PAIRS: List[Tuple[int, int]] = [
    (0, 1), (0, 2), (0, 3),
    (1, 2), (1, 3), (2, 3),
]
GAP_NAMES = [
    "Gap_0_1", "Gap_0_2", "Gap_0_3",
    "Gap_1_2", "Gap_1_3", "Gap_2_3",
]
GAP_LABELS = [
    "Gap 0-1 = |E1-E2|",
    "Gap 0-2 = |E1-E3|",
    "Gap 0-3 = |E1-E4|",
    "Gap 1-2 = |E2-E3|",
    "Gap 1-3 = |E2-E4|",
    "Gap 2-3 = |E3-E4|",
]


# ============================================================
# 1. General utilities
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_r2(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if len(y_true) < 2:
        return np.nan
    if np.allclose(y_true.max(), y_true.min()):
        return np.nan
    return float(r2_score(y_true, y_pred))


def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} != {y_pred.shape}")

    per_dim = []
    for d, name in enumerate(GAP_NAMES):
        yt, yp = y_true[:, d], y_pred[:, d]
        mae = float(np.mean(np.abs(yp - yt)))
        rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
        r2 = safe_r2(yt, yp)
        per_dim.append({"name": name, "mae": mae, "rmse": rmse, "r2": r2})

    valid_r2 = [m["r2"] for m in per_dim if np.isfinite(m["r2"])]
    return {
        "mae": float(np.mean([m["mae"] for m in per_dim])),
        "rmse": float(np.mean([m["rmse"] for m in per_dim])),
        "r2": float(np.mean(valid_r2)) if valid_r2 else np.nan,
        "per_dim": per_dim,
    }


def scaler_to_dict(scaler):
    return {
        "mean_": scaler.mean_.tolist(),
        "scale_": scaler.scale_.tolist(),
        "var_": scaler.var_.tolist(),
        "n_features_in_": int(scaler.n_features_in_),
    }


# ============================================================
# 2. Existing invariant cache only
# ============================================================

def find_existing_cache(run_dir, explicit=None):
    if explicit:
        p = os.path.abspath(explicit)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"指定 cache 不存在: {p}")
        return p

    candidates = sorted(set(
        glob.glob(os.path.join(run_dir, "atomwise_invariant_v3_Z*.pt"))
        + glob.glob(os.path.join(run_dir, "**", "atomwise_invariant_v3_Z*.pt"), recursive=True)
    ))

    candidates = [os.path.abspath(p) for p in candidates if os.path.isfile(p)]
    if not candidates:
        raise FileNotFoundError(
            f"在 {run_dir} 下没有找到已有 atomwise_invariant_v3_Z*.pt。\n"
            "本脚本不会重新构建特征。若 cache 在其他位置，请加：\n"
            "--cache-file /完整路径/atomwise_invariant_v3_Z16-42.pt"
        )

    if len(candidates) > 1:
        print("⚠️ 找到多个 cache：")
        for p in candidates:
            print("   ", p)
        print("默认使用:", candidates[0])

    return candidates[0]


def load_feature_cache(cache_file):
    print("直接加载已有结构 cache:", cache_file)
    cache = torch.load(cache_file, map_location="cpu", weights_only=False)

    required = ["atom_features", "atom_masks", "global_features", "frame_indices"]
    missing = [k for k in required if k not in cache]
    if missing:
        raise KeyError(f"cache 缺少字段 {missing}; keys={list(cache.keys())}")

    atom = np.asarray(cache["atom_features"], dtype=np.float32)
    mask = np.asarray(cache["atom_masks"], dtype=bool)
    glob = np.asarray(cache["global_features"], dtype=np.float32)
    indices = np.asarray(cache["frame_indices"], dtype=np.int64)
    state = cache.get("descriptor_state", {})

    if atom.ndim != 3:
        raise ValueError(f"atom_features 应为 [frame,atom,F], 实际 {atom.shape}")
    if mask.shape != atom.shape[:2]:
        raise ValueError(f"atom_masks shape 不匹配: {mask.shape} vs {atom.shape[:2]}")
    if glob.ndim != 2 or len(glob) != len(atom):
        raise ValueError(f"global_features shape 不匹配: {glob.shape}")
    if len(indices) != len(atom):
        raise ValueError("frame_indices 长度与 atom_features 不一致")

    print("cache 校验通过")
    print("  atom_features   :", atom.shape)
    print("  atom_masks      :", mask.shape)
    print("  global_features :", glob.shape)
    print("  frame_indices   :", indices.shape, f"{indices.min()}~{indices.max()}")
    return atom, mask, glob, indices, state


# ============================================================
# 3. Four energies -> six pairwise gaps
# ============================================================

def _read_numeric_rows(filepath):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"energy 文件不存在: {filepath}")

    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("!"):
                continue
            try:
                values = [float(x) for x in line.split()]
            except ValueError:
                continue
            if values:
                rows.append((line_no, values))

    if not rows:
        raise ValueError(f"energy 文件没有有效数字数据: {filepath}")
    return rows


def six_gaps(energies):
    e = np.asarray(energies, dtype=np.float64).reshape(-1)
    if len(e) != 4:
        raise ValueError(f"必须是4个 energy，实际 {len(e)}")
    return np.asarray([abs(e[i] - e[j]) for i, j in GAP_PAIRS], dtype=np.float64)


def read_energy_and_gap_targets(filepath, frame_indices):
    """
    Supports:
      E1 E2 E3 E4
    or
      index E1 E2 E3 E4
    """
    rows = _read_numeric_rows(filepath)
    frame_indices = [int(x) for x in frame_indices]
    frame_set = set(frame_indices)

    first_as_idx = []
    for _, values in rows:
        if len(values) >= 2 and abs(values[0] - round(values[0])) < 1e-10:
            first_as_idx.append(int(round(values[0])))
        else:
            first_as_idx.append(None)

    n_integer = sum(x is not None for x in first_as_idx)
    n_overlap = sum(x in frame_set for x in first_as_idx if x is not None)

    explicit_index = (
        n_integer >= max(1, int(0.8 * len(rows)))
        and n_overlap >= max(1, int(0.5 * min(len(rows), len(frame_set))))
    )

    energy_dict: Dict[int, np.ndarray] = {}
    gap_dict: Dict[int, np.ndarray] = {}

    if explicit_index:
        print("Energy 格式: index E1 E2 E3 E4")
        for (line_no, values), idx in zip(rows, first_as_idx):
            if idx is None:
                continue
            if len(values) != 5:
                raise ValueError(
                    f"第 {line_no} 行应为 index+4 energy 共5列，实际 {len(values)} 列"
                )
            e = np.asarray(values[1:5], dtype=np.float64)
            if not np.all(np.isfinite(e)):
                raise ValueError(f"第 {line_no} 行存在非有限 energy")
            energy_dict[int(idx)] = e
            gap_dict[int(idx)] = six_gaps(e)
    else:
        print("Energy 格式: E1 E2 E3 E4")
        n = min(len(rows), len(frame_indices))
        if len(rows) != len(frame_indices):
            print(f"⚠️ energy 行数={len(rows)}, frame数={len(frame_indices)}; 对齐前 {n} 个")
        for pos in range(n):
            line_no, values = rows[pos]
            if len(values) != 4:
                raise ValueError(
                    f"第 {line_no} 行必须正好4列 E1 E2 E3 E4，实际 {len(values)} 列"
                )
            e = np.asarray(values, dtype=np.float64)
            if not np.all(np.isfinite(e)):
                raise ValueError(f"第 {line_no} 行存在非有限 energy")
            idx = int(frame_indices[pos])
            energy_dict[idx] = e
            gap_dict[idx] = six_gaps(e)

    if not gap_dict:
        raise ValueError("没有成功建立 gap target")

    mat = np.stack(list(gap_dict.values()), axis=0)
    print(f"目标建立成功: {len(gap_dict)} 帧, target shape={mat.shape}")
    for d, name in enumerate(GAP_LABELS):
        print(f"  Dim{d+1} {name}: min={mat[:,d].min():.8e}, max={mat[:,d].max():.8e}")

    return energy_dict, gap_dict


# ============================================================
# 4. Temporal sample alignment
# ============================================================

def align_samples(frame_indices, energy_dict, gap_dict, seq_len=16):
    starts, targets, energies, target_indices = [], [], [], []
    skip_noncontinuous = 0
    skip_missing = 0

    for end in range(seq_len - 1, len(frame_indices)):
        start = end - seq_len + 1
        window_idx = frame_indices[start:end + 1]

        if not np.all(np.diff(window_idx) == 1):
            skip_noncontinuous += 1
            continue

        idx = int(frame_indices[end])
        if idx not in gap_dict or idx not in energy_dict:
            skip_missing += 1
            continue

        starts.append(start)
        targets.append(gap_dict[idx])
        energies.append(energy_dict[idx])
        target_indices.append(idx)

    starts = np.asarray(starts, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.float64)
    energies = np.asarray(energies, dtype=np.float64)
    target_indices = np.asarray(target_indices, dtype=np.int64)

    if len(starts) < 2:
        raise ValueError("有效 six-gap 样本少于2")

    print("=" * 92)
    print("Six-gap 时序样本对齐完成")
    print(f"有效样本      : {len(starts)}")
    print(f"targets       : {targets.shape}")
    print(f"energies      : {energies.shape}")
    print(f"不连续窗口跳过: {skip_noncontinuous}")
    print(f"缺失标签跳过  : {skip_missing}")
    print(f"target index  : {target_indices.min()}~{target_indices.max()}")
    print("=" * 92)

    return starts, targets, energies, target_indices


# ============================================================
# 5. Same uniform split
# ============================================================

def uniform_split_indices(total_num, train_ratio=0.15):
    if total_num < 2:
        raise ValueError("样本数必须>=2")

    train_num = int(round(total_num * train_ratio))
    train_num = max(1, min(train_num, total_num - 1))

    train_idx = np.unique(
        np.linspace(0, total_num - 1, train_num, dtype=int)
    )
    test_idx = np.setdiff1d(np.arange(total_num), train_idx)

    print(
        f"全局均匀划分 -> Train={len(train_idx)} ({len(train_idx)/total_num:.2%}), "
        f"Test={len(test_idx)} ({len(test_idx)/total_num:.2%})"
    )
    return train_idx, test_idx


# ============================================================
# 6. Train-only scaling
# ============================================================

def fit_feature_scalers(atom, mask, glob, starts, train_pos, seq_len):
    atom_scaler = StandardScaler()
    global_scaler = StandardScaler()

    for p in train_pos:
        s = int(starts[p])
        e = s + seq_len

        aw = atom[s:e]
        mw = mask[s:e]
        valid = aw[mw]
        if valid.ndim != 2 or len(valid) == 0:
            raise ValueError("训练窗口没有有效原子")

        atom_scaler.partial_fit(valid)
        global_scaler.partial_fit(glob[s:e])

    return atom_scaler, global_scaler


def transform_atom_frames(atom, mask, scaler):
    out = np.zeros_like(atom, dtype=np.float32)
    out[mask] = scaler.transform(atom[mask]).astype(np.float32)
    return out


def transform_global_frames(glob, scaler):
    return scaler.transform(glob).astype(np.float32)


# ============================================================
# 7. Dataset: dynamic 16-frame slicing
# ============================================================

class CachedSixGapDataset(Dataset):
    def __init__(
        self,
        atom_t,
        mask_t,
        global_t,
        starts,
        y_scaled,
        y_raw,
        energies,
        frame_indices,
        subset_pos,
        seq_len,
    ):
        self.atom_t = atom_t
        self.mask_t = mask_t
        self.global_t = global_t
        self.starts = np.asarray(starts, dtype=np.int64)
        self.y_scaled = torch.tensor(y_scaled, dtype=torch.float32)
        self.y_raw = torch.tensor(y_raw, dtype=torch.float32)
        self.energies = torch.tensor(energies, dtype=torch.float32)
        self.frame_indices = np.asarray(frame_indices, dtype=np.int64)
        self.subset_pos = np.asarray(subset_pos, dtype=np.int64)
        self.seq_len = int(seq_len)

    def __len__(self):
        return len(self.subset_pos)

    def __getitem__(self, idx):
        p = int(self.subset_pos[idx])
        s = int(self.starts[p])
        e = s + self.seq_len

        return (
            self.atom_t[s:e],
            self.mask_t[s:e],
            self.global_t[s:e],
            self.y_scaled[p],
            self.y_raw[p],
            self.energies[p],
            self.frame_indices[p],
        )


# ============================================================
# 8. Frozen Vicuna six-gap regressor
# ============================================================


class SelfCrossAttentionFCProjector(nn.Module):
    """
    Trainable structure projector:

        raw structural sequence [B,S,input_dim]
            -> LayerNorm
            -> Linear(input_dim -> attention_dim)
            -> temporal position embedding
            -> SELF-ATTENTION across S frames
            -> structure memory [B,S,attention_dim]

        trainable query tokens [B,S,attention_dim]
            -> CROSS-ATTENTION
               Q = trainable queries
               K = structure memory
               V = structure memory
            -> residual + LayerNorm

            -> FULLY CONNECTED / FFN
               attention_dim -> 4*attention_dim -> attention_dim
            -> residual + LayerNorm

            -> Linear(attention_dim -> Vicuna hidden size)
            -> structure pseudo-tokens [B,S,Vicuna_hidden_size]

    The number of trainable query tokens is equal to seq_len, so the
    downstream Vicuna input still receives the same number of structure
    tokens as in the original code.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seq_len: int,
        attention_dim: int = 256,
        num_heads: int = 8,
        num_self_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        if attention_dim % num_heads != 0:
            raise ValueError(
                f"attention_dim={attention_dim} 必须能被 "
                f"num_heads={num_heads} 整除"
            )

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.seq_len = int(seq_len)
        self.attention_dim = int(attention_dim)
        self.num_heads = int(num_heads)
        self.num_self_layers = int(num_self_layers)

        # ----------------------------------------------------
        # A. Input projection
        # ----------------------------------------------------
        self.input_norm = nn.LayerNorm(
            self.input_dim
        )

        self.input_proj = nn.Linear(
            self.input_dim,
            self.attention_dim,
        )

        # Learned temporal position embedding for the structural sequence.
        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                self.seq_len,
                self.attention_dim,
            )
        )

        # ----------------------------------------------------
        # B. SELF-ATTENTION across structural frames
        # ----------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.attention_dim,
            nhead=self.num_heads,
            dim_feedforward=self.attention_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.self_attention = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_self_layers,
        )

        self.memory_norm = nn.LayerNorm(
            self.attention_dim
        )

        # ----------------------------------------------------
        # C. Trainable queries + CROSS-ATTENTION
        # ----------------------------------------------------
        self.query_tokens = nn.Parameter(
            torch.zeros(
                1,
                self.seq_len,
                self.attention_dim,
            )
        )

        self.query_position_embedding = nn.Parameter(
            torch.zeros(
                1,
                self.seq_len,
                self.attention_dim,
            )
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.attention_dim,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.cross_attn_dropout = nn.Dropout(
            dropout
        )

        self.cross_attn_norm = nn.LayerNorm(
            self.attention_dim
        )

        # ----------------------------------------------------
        # D. FULLY CONNECTED / FFN
        # ----------------------------------------------------
        self.fully_connected = nn.Sequential(
            nn.Linear(
                self.attention_dim,
                self.attention_dim * 4,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                self.attention_dim * 4,
                self.attention_dim,
            ),
            nn.GELU(),
        )

        self.fc_dropout = nn.Dropout(
            dropout
        )

        self.fc_norm = nn.LayerNorm(
            self.attention_dim
        )

        # ----------------------------------------------------
        # E. Final projection to Vicuna hidden size
        # ----------------------------------------------------
        self.output_norm = nn.LayerNorm(
            self.attention_dim
        )

        self.output_proj = nn.Linear(
            self.attention_dim,
            self.output_dim,
        )

        # ----------------------------------------------------
        # Initialization
        # ----------------------------------------------------
        nn.init.normal_(
            self.input_proj.weight,
            mean=0.0,
            std=0.02,
        )
        nn.init.zeros_(
            self.input_proj.bias
        )

        nn.init.normal_(
            self.position_embedding,
            mean=0.0,
            std=0.02,
        )

        nn.init.normal_(
            self.query_tokens,
            mean=0.0,
            std=0.02,
        )

        nn.init.normal_(
            self.query_position_embedding,
            mean=0.0,
            std=0.02,
        )

        nn.init.normal_(
            self.output_proj.weight,
            mean=0.0,
            std=0.02,
        )
        nn.init.zeros_(
            self.output_proj.bias
        )

    def forward(self, x):
        # x: [B,S,input_dim]
        if x.ndim != 3:
            raise ValueError(
                f"Projector 输入必须为 [B,S,D]，"
                f"实际 {tuple(x.shape)}"
            )

        B, S, _ = x.shape

        if S != self.seq_len:
            raise ValueError(
                f"Projector seq_len={self.seq_len}，"
                f"当前输入 S={S}"
            )

        # ----------------------------------------------------
        # 1. Input projection
        # ----------------------------------------------------
        h = self.input_proj(
            self.input_norm(
                x
            )
        )

        h = (
            h
            + self.position_embedding[
                :,
                :S,
                :,
            ]
        )

        # ----------------------------------------------------
        # 2. SELF-ATTENTION
        # ----------------------------------------------------
        memory = self.self_attention(
            h
        )

        memory = self.memory_norm(
            memory
        )

        # ----------------------------------------------------
        # 3. CROSS-ATTENTION
        #    Q = trainable queries
        #    K,V = self-attended structure memory
        # ----------------------------------------------------
        queries = (
            self.query_tokens[
                :,
                :S,
                :,
            ]
            + self.query_position_embedding[
                :,
                :S,
                :,
            ]
        )

        queries = queries.expand(
            B,
            -1,
            -1,
        )

        cross_out, _ = self.cross_attention(
            query=queries,
            key=memory,
            value=memory,
            need_weights=False,
        )

        h = self.cross_attn_norm(
            queries
            + self.cross_attn_dropout(
                cross_out
            )
        )

        # ----------------------------------------------------
        # 4. FULLY CONNECTED / FFN
        # ----------------------------------------------------
        fc_out = self.fully_connected(
            h
        )

        h = self.fc_norm(
            h
            + self.fc_dropout(
                fc_out
            )
        )

        # ----------------------------------------------------
        # 5. Map to Vicuna embedding dimension
        # ----------------------------------------------------
        h = self.output_proj(
            self.output_norm(
                h
            )
        )

        return h


class FrozenVicunaSixGap(nn.Module):
    def __init__(
        self,
        model_path,
        atom_dim,
        global_dim,
        prompt,
        seq_len=SEQ_LEN,
        use_gradient_checkpointing=False,
    ):
        super().__init__()

        self.seq_len = int(
            seq_len
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )

        print("加载 Vicuna:", model_path)
        print("LLM dtype:", self.llm_dtype)

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.llm_dtype,
            low_cpu_mem_usage=True,
        )
        self.llm.config.use_cache = False

        for p in self.llm.parameters():
            p.requires_grad = False
        self.llm.eval()

        if use_gradient_checkpointing:
            self.llm.gradient_checkpointing_enable()
            print("gradient checkpointing: ON")
        else:
            print("gradient checkpointing: OFF")

        self.hidden_size = int(self.llm.config.hidden_size)
        raw_frame_dim = int(atom_dim + global_dim)

        # Structure projector:
        # Self-Attention -> Cross-Attention -> Fully Connected
        # -> Vicuna hidden size.
        self.structure_projector = SelfCrossAttentionFCProjector(
            input_dim=raw_frame_dim,
            output_dim=self.hidden_size,
            seq_len=self.seq_len,
            attention_dim=256,
            num_heads=8,
            num_self_layers=2,
            dropout=0.1,
        )

        # Same head depth as the previous four-state Energy model;
        # only final output dimension changes from 4 to 6.
        self.gap_head = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, 512),
            nn.GELU(),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, OUT_DIM),
        )

        prompt_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        )["input_ids"]
        self.register_buffer("prompt_ids", prompt_ids, persistent=True)

        print("Vicuna hidden_size :", self.hidden_size)
        print("Prompt tokens      :", prompt_ids.shape[1])
        print("Raw frame dim      :", raw_frame_dim)
        print(
            "Projector          : "
            "Self-Attention + Cross-Attention + Fully Connected "
            "(dim=256, heads=8, self_layers=2)"
        )
        print("Structure tokens   :", self.seq_len)
        print("Output dim         :", OUT_DIM)

    def train(self, mode=True):
        super().train(mode)
        self.llm.eval()
        return self

    def trainable_parameters(self):
        for module in [
            self.structure_projector,
            self.gap_head,
        ]:
            yield from module.parameters()

    def trainable_state_dict(self):
        return {
            "structure_projector":
                self.structure_projector.state_dict(),
            "gap_head":
                self.gap_head.state_dict(),
        }

    def load_trainable_state_dict(self, state):
        self.structure_projector.load_state_dict(
            state["structure_projector"]
        )
        self.gap_head.load_state_dict(
            state["gap_head"]
        )

    def forward(self, atom_x, atom_mask, global_x):
        # [B,S,A,F], [B,S,A], [B,S,G]
        B = atom_x.shape[0]

        mask_f = atom_mask.unsqueeze(-1).to(atom_x.dtype)
        atom_mean = (atom_x * mask_f).sum(dim=2) / mask_f.sum(dim=2).clamp_min(1.0)

        raw_frame = torch.cat(
            [
                atom_mean,
                global_x,
            ],
            dim=-1,
        )

        struct_tokens = self.structure_projector(
            raw_frame
        ).to(
            self.llm_dtype
        )

        ids = self.prompt_ids.expand(B, -1)
        text_emb = self.llm.get_input_embeddings()(ids)

        bos_id = self.tokenizer.bos_token_id
        has_bos = (
            ids.shape[1] > 0
            and bos_id is not None
            and bool(torch.all(ids[:, 0] == bos_id))
        )

        if has_bos:
            input_embeds = torch.cat(
                [text_emb[:, :1], struct_tokens, text_emb[:, 1:]],
                dim=1,
            )
        else:
            input_embeds = torch.cat([struct_tokens, text_emb], dim=1)

        attention_mask = torch.ones(
            input_embeds.shape[:2],
            dtype=torch.long,
            device=input_embeds.device,
        )

        # No torch.no_grad() here during training:
        # gradient must pass through frozen Vicuna to the Self-Attention + Cross-Attention + FC projector.
        out = self.llm.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )

        h = out.last_hidden_state[:, -1, :].float()
        return self.gap_head(h)


# ============================================================
# 9. Loss / epoch / inference
# ============================================================

class GapLoss(nn.Module):
    def __init__(self, beta=0.5):
        super().__init__()
        self.beta = float(beta)

    def forward(self, pred, target):
        return F.smooth_l1_loss(
            pred,
            target,
            beta=self.beta,
            reduction="mean",
        )


def process_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    target_scaler,
    is_train=True,
    grad_accum=1,
):
    model.train(is_train)
    if is_train:
        optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    pred_all, true_all = [], []
    trainable = list(model.trainable_parameters())

    with torch.set_grad_enabled(is_train):
        for step, batch in enumerate(loader, start=1):
            atom_x, atom_mask, global_x, y_scaled, _, _, _ = batch

            atom_x = atom_x.to(device, non_blocking=True)
            atom_mask = atom_mask.to(device, non_blocking=True)
            global_x = global_x.to(device, non_blocking=True)
            y_scaled = y_scaled.to(device, non_blocking=True)

            pred = model(atom_x, atom_mask, global_x)
            loss = criterion(pred, y_scaled)

            if is_train:
                (loss / grad_accum).backward()

                if step % grad_accum == 0 or step == len(loader):
                    torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            bs = atom_x.size(0)
            total_loss += float(loss.detach()) * bs
            pred_all.append(pred.detach().float().cpu().numpy())
            true_all.append(y_scaled.detach().float().cpu().numpy())

    pred_scaled = np.concatenate(pred_all, axis=0)
    true_scaled = np.concatenate(true_all, axis=0)

    pred_real = target_scaler.inverse_transform(pred_scaled)
    true_real = target_scaler.inverse_transform(true_scaled)
    metrics = calc_metrics(true_real, pred_real)

    return {
        "loss": total_loss / len(loader.dataset),
        **metrics,
    }


@torch.no_grad()
def predict_loader(model, loader, device, target_scaler):
    model.eval()
    pred_all, true_all, energies_all, idx_all = [], [], [], []

    for batch in loader:
        atom_x, atom_mask, global_x, y_scaled, _, energies, idx = batch

        pred = model(
            atom_x.to(device, non_blocking=True),
            atom_mask.to(device, non_blocking=True),
            global_x.to(device, non_blocking=True),
        )

        pred_all.append(pred.float().cpu().numpy())
        true_all.append(y_scaled.numpy())
        energies_all.append(energies.numpy())
        idx_all.append(np.asarray(idx))

    pred_real = target_scaler.inverse_transform(np.concatenate(pred_all, axis=0))
    true_real = target_scaler.inverse_transform(np.concatenate(true_all, axis=0))

    return (
        true_real,
        pred_real,
        np.concatenate(energies_all, axis=0),
        np.concatenate(idx_all, axis=0),
    )


# ============================================================
# 10. Plots / CSV
# ============================================================

def plot_history(history, filename):
    epochs = history["epoch"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, key, title in [
        (axes[0, 0], "loss", "Normalized SmoothL1 Loss"),
        (axes[0, 1], "mae", "Macro MAE"),
        (axes[1, 0], "r2", "Macro R2"),
        (axes[1, 1], "rmse", "Macro RMSE"),
    ]:
        ax.plot(epochs, history[f"train_{key}"], label="Train")
        ax.plot(epochs, history[f"test_{key}"], label="Test", linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()

    plt.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)


def plot_comparison(y_true, y_pred, split_name, filename):
    fig, axes = plt.subplots(OUT_DIM, 2, figsize=(15, 5 * OUT_DIM), squeeze=False)

    for d, label in enumerate(GAP_LABELS):
        yt, yp = y_true[:, d], y_pred[:, d]
        m = calc_metrics(
            y_true[:, [d]],
            y_pred[:, [d]],
        ) if False else None

        mae = float(np.mean(np.abs(yp - yt)))
        rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
        r2 = safe_r2(yt, yp)

        order = np.argsort(yt)
        x = np.arange(1, len(yt) + 1)

        axes[d, 0].plot(x, yt[order], label="True", linewidth=1.5)
        axes[d, 0].plot(x, yp[order], label="Pred", linestyle="--", linewidth=1.5)
        axes[d, 0].set_title(f"[{split_name}] {label}")
        axes[d, 0].set_xlabel("Samples sorted by true gap")
        axes[d, 0].set_ylabel(f"Gap ({UNIT})")
        axes[d, 0].grid(True, linestyle=":", alpha=0.6)
        axes[d, 0].legend()

        axes[d, 1].scatter(yt, yp, alpha=0.65)
        lo = float(min(yt.min(), yp.min()))
        hi = float(max(yt.max(), yp.max()))
        if np.isclose(lo, hi):
            hi = lo + 1e-12
        axes[d, 1].plot([lo, hi], [lo, hi], "k--", linewidth=2)
        axes[d, 1].set_title(f"[{split_name}] Parity {label}")
        axes[d, 1].set_xlabel(f"True ({UNIT})")
        axes[d, 1].set_ylabel(f"Pred ({UNIT})")
        axes[d, 1].grid(True, linestyle=":", alpha=0.6)
        axes[d, 1].text(
            0.95, 0.05,
            f"R2={r2:.6f}\nMAE={mae:.6e}\nRMSE={rmse:.6e}",
            transform=axes[d, 1].transAxes,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round", alpha=0.8),
        )

    plt.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)


def plot_full_evolution(
    all_true,
    train_pos,
    train_pred,
    test_pos,
    test_pred,
    frame_indices,
    filename,
):
    fig, axes = plt.subplots(3, 2, figsize=(16, 15), squeeze=False)
    axes = axes.flatten()

    for d, label in enumerate(GAP_LABELS):
        ax = axes[d]
        ax.plot(frame_indices, all_true[:, d], linewidth=1.2, label="True")
        ax.scatter(frame_indices[train_pos], train_pred[:, d], s=12, label="Train Pred")
        ax.scatter(frame_indices[test_pos], test_pred[:, d], s=16, label="Test Pred")
        ax.set_title(label)
        ax.set_xlabel("MD Step / Target Frame Index")
        ax.set_ylabel(f"Gap ({UNIT})")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend()

    plt.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)


def save_predictions(true, pred, energies, indices, filename):
    data = {"Frame_Index": indices}

    for i in range(4):
        data[f"E{i+1}"] = energies[:, i]

    for d, name in enumerate(GAP_NAMES):
        data[f"True_{name}"] = true[:, d]
        data[f"Pred_{name}"] = pred[:, d]
        data[f"Error_{name}"] = pred[:, d] - true[:, d]
        data[f"AbsError_{name}"] = np.abs(pred[:, d] - true[:, d])

    pd.DataFrame(data).sort_values("Frame_Index").to_csv(
        filename,
        index=False,
        float_format="%.12e",
    )


def save_metrics(true, pred, split_name, filename):
    m = calc_metrics(true, pred)
    rows = [
        {
            "Split": split_name,
            "Gap": x["name"],
            "MAE": x["mae"],
            "RMSE": x["rmse"],
            "R2": x["r2"],
        }
        for x in m["per_dim"]
    ]
    pd.DataFrame(rows).to_csv(filename, index=False)


def save_full_csv(
    true,
    energies,
    train_pos,
    train_pred,
    test_pos,
    test_pred,
    indices,
    filename,
):
    n = len(true)
    data = {
        "Dataset_Position": np.arange(n),
        "Frame_Index": indices,
    }

    for i in range(4):
        data[f"E{i+1}"] = energies[:, i]

    for d, name in enumerate(GAP_NAMES):
        tr = np.full(n, np.nan)
        te = np.full(n, np.nan)
        comb = np.full(n, np.nan)

        tr[train_pos] = train_pred[:, d]
        te[test_pos] = test_pred[:, d]
        comb[train_pos] = train_pred[:, d]
        comb[test_pos] = test_pred[:, d]

        data[f"True_{name}"] = true[:, d]
        data[f"Train_Pred_{name}"] = tr
        data[f"Test_Pred_{name}"] = te
        data[f"Combined_Pred_{name}"] = comb
        data[f"Combined_AbsError_{name}"] = np.abs(comb - true[:, d])

    pd.DataFrame(data).sort_values("Frame_Index").to_csv(
        filename,
        index=False,
        float_format="%.12e",
    )


# ============================================================
# 11. CLI / main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    p.add_argument("--cache-file", default=None)
    p.add_argument("--energy-file", default=DEFAULT_ENERGY_FILE)
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)

    p.add_argument("--seq-len", type=int, default=SEQ_LEN)
    p.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)

    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)

    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--patience", type=int, default=PATIENCE)
    p.add_argument("--num-workers", type=int, default=0)

    p.add_argument("--gradient-checkpointing", action="store_true")

    p.add_argument(
        "--prompt",
        default=(
            "Predict all six pairwise electronic-state energy gaps at the "
            "target frame from the 16-frame invariant structural sequence. "
            "The four electronic states are E1, E2, E3, and E4. "
            "Each target is an absolute pairwise energy difference."
        ),
    )

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA GPU 运行 Vicuna-7B")

    run_dir = os.path.abspath(args.run_dir)
    cache_file = find_existing_cache(run_dir, args.cache_file)
    energy_file = os.path.abspath(args.energy_file)
    output_root = os.path.abspath(args.output_root)

    model_dir = os.path.join(output_root, "models")
    log_dir = os.path.join(output_root, "logs")
    plot_dir = os.path.join(output_root, "plots")
    result_dir = os.path.join(output_root, "results")

    for d in [model_dir, log_dir, plot_dir, result_dir]:
        os.makedirs(d, exist_ok=True)

    device = torch.device("cuda")

    print("=" * 100)
    print("Existing invariant cache + Frozen Vicuna-7B + 6 pairwise Energy Gaps")
    print("=" * 100)
    print("Run dir      :", run_dir)
    print("Cache        :", cache_file)
    print("Energy       :", energy_file)
    print("Vicuna       :", args.model_path)
    print("Output       :", output_root)
    print("SEQ_LEN      :", args.seq_len)
    print("TRAIN_RATIO  :", args.train_ratio)
    print("Batch/accum  :", f"{args.batch_size}/{args.grad_accum}")
    print("GPU          :", torch.cuda.get_device_name(0))
    print("=" * 100)

    # 1) Existing cache
    atom, mask, glob, frame_indices, descriptor_state = load_feature_cache(cache_file)

    # 2) Four energies -> six gaps
    energy_dict, gap_dict = read_energy_and_gap_targets(
        energy_file,
        frame_indices.tolist(),
    )

    # 3) Temporal sample alignment
    starts, targets, energies, target_indices = align_samples(
        frame_indices,
        energy_dict,
        gap_dict,
        seq_len=args.seq_len,
    )

    # 4) Same uniform split as previous Energy model
    train_pos, test_pos = uniform_split_indices(
        len(starts),
        train_ratio=args.train_ratio,
    )

    # 5) Train-only structural scalers
    atom_scaler, global_scaler = fit_feature_scalers(
        atom, mask, glob, starts, train_pos, args.seq_len
    )
    atom_scaled = transform_atom_frames(atom, mask, atom_scaler)
    global_scaled = transform_global_frames(glob, global_scaler)

    # 6) Train-only six-gap target scaler
    target_scaler = StandardScaler()
    target_scaler.fit(targets[train_pos])
    targets_scaled = target_scaler.transform(targets).astype(np.float32)

    print("\nTarget train mean/std:")
    for d, name in enumerate(GAP_NAMES):
        print(
            f"  {name}: "
            f"{target_scaler.mean_[d]:.12e} / "
            f"{target_scaler.scale_[d]:.12e}"
        )

    # 7) Dataset / loaders
    atom_t = torch.from_numpy(atom_scaled)
    mask_t = torch.from_numpy(mask)
    global_t = torch.from_numpy(global_scaled)

    train_ds = CachedSixGapDataset(
        atom_t, mask_t, global_t,
        starts, targets_scaled, targets, energies, target_indices,
        train_pos, args.seq_len,
    )
    test_ds = CachedSixGapDataset(
        atom_t, mask_t, global_t,
        starts, targets_scaled, targets, energies, target_indices,
        test_pos, args.seq_len,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    eval_train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 8) Model
    model = FrozenVicunaSixGap(
        model_path=args.model_path,
        atom_dim=atom_scaled.shape[-1],
        global_dim=global_scaled.shape[-1],
        prompt=args.prompt,
        seq_len=args.seq_len,
        use_gradient_checkpointing=args.gradient_checkpointing,
    ).to(device)

    trainable = list(model.trainable_parameters())

    print(
        f"LLM params frozen : "
        f"{sum(p.numel() for p in model.llm.parameters()):,}"
    )
    print(
        f"Trainable params  : "
        f"{sum(p.numel() for p in trainable):,}"
    )

    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=35,
        min_lr=1e-6,
    )
    criterion = GapLoss(beta=SMOOTH_L1_BETA).to(device)

    np.savez(
        os.path.join(model_dir, "scalers.npz"),
        atom_feature_mean=atom_scaler.mean_,
        atom_feature_scale=atom_scaler.scale_,
        atom_feature_var=atom_scaler.var_,
        global_feature_mean=global_scaler.mean_,
        global_feature_scale=global_scaler.scale_,
        global_feature_var=global_scaler.var_,
        target_mean=target_scaler.mean_,
        target_scale=target_scaler.scale_,
        target_var=target_scaler.var_,
    )

    history = {
        "epoch": [], "lr": [],
        "train_loss": [], "test_loss": [],
        "train_mae": [], "test_mae": [],
        "train_rmse": [], "test_rmse": [],
        "train_r2": [], "test_r2": [],
    }

    best_loss = float("inf")
    no_improve = 0
    best_model_path = os.path.join(
        model_dir,
        "best_frozen_vicuna_six_gaps.pth",
    )

    print("\n开始训练：Vicuna 完全冻结，只训练 Self-Attention + Cross-Attention + FC projector + six-gap head")

    for epoch in range(1, args.epochs + 1):
        tr = process_epoch(
            model, train_loader, optimizer, criterion,
            device, target_scaler,
            is_train=True,
            grad_accum=args.grad_accum,
        )
        te = process_epoch(
            model, test_loader, optimizer, criterion,
            device, target_scaler,
            is_train=False,
            grad_accum=1,
        )

        scheduler.step(te["loss"])
        lr_now = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["lr"].append(lr_now)
        for k in ["loss", "mae", "rmse", "r2"]:
            history[f"train_{k}"].append(tr[k])
            history[f"test_{k}"].append(te[k])

        print(
            f"Epoch {epoch:04d}/{args.epochs} | "
            f"Loss {tr['loss']:.5f}/{te['loss']:.5f} | "
            f"MAE {tr['mae']:.3e}/{te['mae']:.3e} | "
            f"RMSE {tr['rmse']:.3e}/{te['rmse']:.3e} | "
            f"R2 {tr['r2']:.5f}/{te['r2']:.5f} | "
            f"LR={lr_now:.2e}"
        )

        if epoch == 1 or epoch % 20 == 0:
            for m in te["per_dim"]:
                print(
                    f"   Test {m['name']}: "
                    f"MAE={m['mae']:.3e}, RMSE={m['rmse']:.3e}, R2={m['r2']:.6f}"
                )

        if te["loss"] < best_loss - MIN_DELTA:
            best_loss = te["loss"]
            no_improve = 0

            torch.save(
                {
                    "trainable_state_dict": model.trainable_state_dict(),
                    "model_path": args.model_path,
                    "hidden_size": model.hidden_size,
                    "atom_dim": int(atom_scaled.shape[-1]),
                    "global_dim": int(global_scaled.shape[-1]),
                    "out_dim": OUT_DIM,
                    "gap_pairs": GAP_PAIRS,
                    "gap_names": GAP_NAMES,
                    "seq_len": args.seq_len,
                    "train_ratio": args.train_ratio,
                    "batch_size": args.batch_size,
                    "grad_accum": args.grad_accum,
                    "best_epoch": epoch,
                    "best_test_loss": best_loss,
                    "prompt": args.prompt,
                    "cache_file": cache_file,
                    "energy_file": energy_file,
                    "descriptor_state": descriptor_state,
                    "atom_scaler": scaler_to_dict(atom_scaler),
                    "global_scaler": scaler_to_dict(global_scaler),
                    "target_scaler": scaler_to_dict(target_scaler),
                    "vicuna_frozen": True,
                },
                best_model_path,
            )
            print("   --> 保存最优 checkpoint:", best_model_path)
        else:
            no_improve += 1

        pd.DataFrame(history).to_csv(
            os.path.join(log_dir, "training_metrics_history.csv"),
            index=False,
        )

        if epoch == 1 or epoch % 10 == 0:
            plot_history(
                history,
                os.path.join(plot_dir, "training_curves.png"),
            )

        if no_improve >= args.patience:
            print(f"Early stopping at epoch={epoch}")
            break

    plot_history(
        history,
        os.path.join(plot_dir, "training_curves.png"),
    )

    # Final evaluation
    ckpt = torch.load(best_model_path, map_location="cpu", weights_only=False)
    model.load_trainable_state_dict(ckpt["trainable_state_dict"])
    model.eval()

    train_true, train_pred, train_energy, train_idx = predict_loader(
        model, eval_train_loader, device, target_scaler
    )
    test_true, test_pred, test_energy, test_idx = predict_loader(
        model, test_loader, device, target_scaler
    )

    train_m = calc_metrics(train_true, train_pred)
    test_m = calc_metrics(test_true, test_pred)

    print("\n" + "=" * 100)
    print(
        f"Train Macro: MAE={train_m['mae']:.8e} | "
        f"RMSE={train_m['rmse']:.8e} | R2={train_m['r2']:.8f}"
    )
    print(
        f"Test  Macro: MAE={test_m['mae']:.8e} | "
        f"RMSE={test_m['rmse']:.8e} | R2={test_m['r2']:.8f}"
    )
    for m in test_m["per_dim"]:
        print(
            f"Test {m['name']}: "
            f"MAE={m['mae']:.8e} | RMSE={m['rmse']:.8e} | R2={m['r2']:.8f}"
        )
    print("=" * 100)

    # Plots
    plot_comparison(
        train_true, train_pred, "Train",
        os.path.join(plot_dir, "train_six_gap_prediction.png"),
    )
    plot_comparison(
        test_true, test_pred, "Test",
        os.path.join(plot_dir, "test_six_gap_prediction.png"),
    )
    plot_full_evolution(
        targets, train_pos, train_pred, test_pos, test_pred, target_indices,
        os.path.join(plot_dir, "full_md_six_gap_evolution.png"),
    )

    # CSV
    save_predictions(
        train_true, train_pred, train_energy, train_idx,
        os.path.join(result_dir, "train_six_gap_predictions.csv"),
    )
    save_predictions(
        test_true, test_pred, test_energy, test_idx,
        os.path.join(result_dir, "test_six_gap_predictions.csv"),
    )
    save_metrics(
        train_true, train_pred, "Train",
        os.path.join(result_dir, "train_metrics_per_gap.csv"),
    )
    save_metrics(
        test_true, test_pred, "Test",
        os.path.join(result_dir, "test_metrics_per_gap.csv"),
    )
    save_full_csv(
        targets, energies,
        train_pos, train_pred,
        test_pos, test_pred,
        target_indices,
        os.path.join(result_dir, "full_md_six_gap_evolution.csv"),
    )

    summary = {
        "architecture": (
            "existing invariant cache -> masked atom mean + global -> "
            "Self-Attention + Cross-Attention + Fully Connected projector -> "
            "frozen Vicuna-7B -> six-gap head"
        ),
        "gap_pairs_0based": GAP_PAIRS,
        "gap_names": GAP_NAMES,
        "target_definition": GAP_LABELS,
        "cache_file": cache_file,
        "energy_file": energy_file,
        "vicuna_model_path": args.model_path,
        "vicuna_frozen": True,
        "seq_len": args.seq_len,
        "train_ratio": args.train_ratio,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "samples": int(len(starts)),
        "train_samples": int(len(train_pos)),
        "test_samples": int(len(test_pos)),
        "best_epoch": int(ckpt["best_epoch"]),
        "best_test_loss": float(ckpt["best_test_loss"]),
        "train_metrics": train_m,
        "test_metrics": test_m,
    }

    with open(
        os.path.join(result_dir, "run_summary.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n全部完成。")
    print("输出目录:", output_root)
    print("最优模型:", best_model_path)


if __name__ == "__main__":
    main()
 