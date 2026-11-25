import os
import json
import random
import argparse
from typing import Dict, List

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
# 0. Defaults
# ============================================================

DEFAULT_RUN_DIR = "/root/vicuna/311_VS_energy"

DEFAULT_CACHE_FILE = os.path.join(
    DEFAULT_RUN_DIR,
    "atomwise_invariant_v3_Z16-42.pt",
)

DEFAULT_ENERGY_FILE = os.path.join(
    DEFAULT_RUN_DIR,
    "energy",
    "energy",
)

DEFAULT_MODEL_PATH = "/root/vicuna/models/vicuna-7b-v1.5"

DEFAULT_OUTPUT_ROOT = os.path.join(
    DEFAULT_RUN_DIR,
    "outputs_vicuna_four_state_energy",
)

SEED = 42

SEQ_LEN = 16
TRAIN_RATIO = 0.15

# User's requested/previously used real physical batch.
BATCH_SIZE = 32
GRAD_ACCUM = 1

EPOCHS = 1000
LR = 2.0e-4
WEIGHT_DECAY = 1.0e-4

SMOOTH_L1_BETA = 0.5

PATIENCE = 200
MIN_DELTA = 1.0e-7

ENERGY_UNIT_LABEL = "eV"

EXPECTED_OUT_DIM = 4
ENERGY_NAMES = ["E1", "E2", "E3", "E4"]


# ============================================================
# 1. Reproducibility / metrics
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_r2(y_true, y_pred):
    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    ).reshape(-1)

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    ).reshape(-1)

    if len(y_true) < 2:
        return np.nan

    if np.allclose(
        np.max(y_true),
        np.min(y_true),
    ):
        return np.nan

    return float(
        r2_score(
            y_true,
            y_pred,
        )
    )


def calc_multidim_metrics(
    y_true,
    y_pred,
    names=None,
):
    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    if y_true.ndim == 1:
        y_true = y_true.reshape(
            -1,
            1,
        )

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(
            -1,
            1,
        )

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"metric shape mismatch: "
            f"{y_true.shape} != {y_pred.shape}"
        )

    if names is None:
        names = [
            f"Energy_{d+1}"
            for d in range(
                y_true.shape[1]
            )
        ]

    if len(names) != y_true.shape[1]:
        raise ValueError(
            "names 数量与 energy 输出维度不一致"
        )

    per_dim = []

    for d in range(
        y_true.shape[1]
    ):
        yt = y_true[:, d]
        yp = y_pred[:, d]

        mae = float(
            np.mean(
                np.abs(
                    yp - yt
                )
            )
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    (
                        yp - yt
                    ) ** 2
                )
            )
        )

        r2 = safe_r2(
            yt,
            yp,
        )

        per_dim.append({
            "dim": d + 1,
            "name": names[d],
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        })

    macro_mae = float(
        np.mean(
            [
                m["mae"]
                for m in per_dim
            ]
        )
    )

    macro_rmse = float(
        np.mean(
            [
                m["rmse"]
                for m in per_dim
            ]
        )
    )

    valid_r2 = [
        m["r2"]
        for m in per_dim
        if np.isfinite(
            m["r2"]
        )
    ]

    macro_r2 = (
        float(
            np.mean(valid_r2)
        )
        if valid_r2
        else np.nan
    )

    return {
        "mae": macro_mae,
        "rmse": macro_rmse,
        "r2": macro_r2,
        "per_dim": per_dim,
    }


# ============================================================
# 2. Directly load EXISTING invariant cache
# ============================================================

def load_feature_cache(
    cache_file: str,
):
    if not os.path.isfile(
        cache_file
    ):
        raise FileNotFoundError(
            f"feature cache 不存在: "
            f"{cache_file}"
        )

    print(
        f"直接加载已有 feature cache: "
        f"{cache_file}"
    )

    cache = torch.load(
        cache_file,
        map_location="cpu",
        weights_only=False,
    )

    required = [
        "atom_features",
        "atom_masks",
        "global_features",
        "frame_indices",
    ]

    missing = [
        k
        for k in required
        if k not in cache
    ]

    if missing:
        raise KeyError(
            f"cache 缺少字段: "
            f"{missing}; "
            f"实际 keys="
            f"{list(cache.keys())}"
        )

    atom_features = np.asarray(
        cache["atom_features"],
        dtype=np.float32,
    )

    atom_masks = np.asarray(
        cache["atom_masks"],
        dtype=bool,
    )

    global_features = np.asarray(
        cache["global_features"],
        dtype=np.float32,
    )

    frame_indices = np.asarray(
        cache["frame_indices"],
        dtype=np.int64,
    )

    descriptor_state = cache.get(
        "descriptor_state",
        {},
    )

    if atom_features.ndim != 3:
        raise ValueError(
            f"atom_features 应为 "
            f"[frames,atoms,F]，"
            f"得到 {atom_features.shape}"
        )

    if (
        atom_masks.shape
        != atom_features.shape[:2]
    ):
        raise ValueError(
            f"atom_masks shape 不匹配: "
            f"{atom_masks.shape} vs "
            f"{atom_features.shape[:2]}"
        )

    if (
        global_features.ndim != 2
        or len(global_features)
        != len(atom_features)
    ):
        raise ValueError(
            f"global_features shape "
            f"不匹配: "
            f"{global_features.shape}"
        )

    if (
        len(frame_indices)
        != len(atom_features)
    ):
        raise ValueError(
            "frame_indices 长度与 "
            "features 不一致"
        )

    print("cache 校验通过")
    print(
        f"  atom_features   : "
        f"{atom_features.shape}"
    )
    print(
        f"  atom_masks      : "
        f"{atom_masks.shape}"
    )
    print(
        f"  global_features : "
        f"{global_features.shape}"
    )
    print(
        f"  frame_indices   : "
        f"{frame_indices.shape}, "
        f"range="
        f"{frame_indices.min()}~"
        f"{frame_indices.max()}"
    )

    if descriptor_state:
        print(
            f"  cache_version   : "
            f"{descriptor_state.get('cache_version', 'unknown')}"
        )

    return (
        atom_features,
        atom_masks,
        global_features,
        frame_indices,
        descriptor_state,
    )


# ============================================================
# 3. Four-state Energy target loading
# ============================================================

def _read_numeric_energy_rows(
    filepath: str,
):
    if not os.path.isfile(
        filepath
    ):
        raise FileNotFoundError(
            f"energy 文件不存在: "
            f"{filepath}"
        )

    rows = []

    with open(
        filepath,
        "r",
        encoding="utf-8",
    ) as f:

        for line_no, line in enumerate(
            f,
            start=1,
        ):
            line = line.strip()

            if (
                not line
                or line.startswith("#")
                or line.startswith("!")
            ):
                continue

            parts = line.split()

            try:
                values = [
                    float(x)
                    for x in parts
                ]
            except ValueError:
                # allow header
                continue

            if values:
                rows.append(
                    (
                        line_no,
                        values,
                    )
                )

    if not rows:
        raise ValueError(
            "没有从 energy 文件读取到 "
            f"有效数字数据: {filepath}"
        )

    return rows


def read_four_state_energy_file(
    filepath: str,
    sorted_frame_indices: List[int],
):
    """
    Supports exactly these two forms:

      A) E1 E2 E3 E4

      B) index E1 E2 E3 E4

    Returns:
      energy_dict[idx] = np.ndarray([E1,E2,E3,E4])
      out_dim = 4
    """
    rows = _read_numeric_energy_rows(
        filepath
    )

    sorted_frame_indices = [
        int(x)
        for x in sorted_frame_indices
    ]

    frame_set = set(
        sorted_frame_indices
    )

    explicit_candidates = []

    for _, values in rows:
        if (
            len(values) >= 2
            and abs(
                values[0]
                - round(values[0])
            ) < 1.0e-10
        ):
            explicit_candidates.append(
                int(
                    round(
                        values[0]
                    )
                )
            )
        else:
            explicit_candidates.append(
                None
            )

    n_integer_first = sum(
        x is not None
        for x in explicit_candidates
    )

    n_overlap = sum(
        x in frame_set
        for x in explicit_candidates
        if x is not None
    )

    use_explicit_index = (
        len(rows) > 0
        and n_integer_first
        >= max(
            1,
            int(
                0.8
                * len(rows)
            ),
        )
        and n_overlap
        >= max(
            1,
            int(
                0.5
                * min(
                    len(rows),
                    len(frame_set),
                )
            ),
        )
    )

    result: Dict[
        int,
        np.ndarray,
    ] = {}

    if use_explicit_index:
        print(
            "Energy 格式识别: "
            "index E1 E2 E3 E4"
        )

        for (
            line_no,
            values,
        ), idx_candidate in zip(
            rows,
            explicit_candidates,
        ):

            if idx_candidate is None:
                continue

            if len(values) != 5:
                raise ValueError(
                    f"energy 第 {line_no} 行 "
                    f"应为 5 列 "
                    f"[index,E1,E2,E3,E4]，"
                    f"实际 {len(values)} 列: "
                    f"{values}"
                )

            idx = int(
                idx_candidate
            )

            target = np.asarray(
                values[1:5],
                dtype=np.float64,
            )

            if not np.all(
                np.isfinite(target)
            ):
                raise ValueError(
                    f"energy 第 {line_no} 行 "
                    "包含非有限值"
                )

            result[idx] = target

    else:
        print(
            "Energy 格式识别: "
            "E1 E2 E3 E4 "
            "(无显式 index)"
        )

        n = min(
            len(rows),
            len(sorted_frame_indices),
        )

        if (
            len(rows)
            != len(sorted_frame_indices)
        ):
            print(
                f"⚠️ energy 数值行数="
                f"{len(rows)}，"
                f"frame 数量="
                f"{len(sorted_frame_indices)}；"
                f"仅对齐前 {n} 个。"
            )

        for pos in range(n):
            line_no, values = (
                rows[pos]
            )

            if len(values) != 4:
                raise ValueError(
                    f"energy 第 {line_no} 行 "
                    f"必须正好是四个 state energy "
                    f"[E1,E2,E3,E4]，"
                    f"实际 {len(values)} 列: "
                    f"{values}"
                )

            idx = int(
                sorted_frame_indices[pos]
            )

            target = np.asarray(
                values[:4],
                dtype=np.float64,
            )

            if not np.all(
                np.isfinite(target)
            ):
                raise ValueError(
                    f"energy 第 {line_no} 行 "
                    "包含非有限值"
                )

            result[idx] = target

    if not result:
        raise ValueError(
            "没有成功建立 "
            "four-state energy 与 frame "
            "的对应关系"
        )

    matrix = np.stack(
        list(
            result.values()
        ),
        axis=0,
    )

    if (
        matrix.ndim != 2
        or matrix.shape[1]
        != EXPECTED_OUT_DIM
    ):
        raise ValueError(
            f"four-state target shape "
            f"异常: {matrix.shape}"
        )

    print(
        f"Four-state Energy 对齐成功: "
        f"{len(result)} 帧 | "
        f"out_dim="
        f"{EXPECTED_OUT_DIM}"
    )

    for d, name in enumerate(
        ENERGY_NAMES
    ):
        print(
            f"  {name}: "
            f"min="
            f"{matrix[:, d].min():.8e}, "
            f"max="
            f"{matrix[:, d].max():.8e}"
        )

    return (
        result,
        EXPECTED_OUT_DIM,
    )


# ============================================================
# 4. Lightweight temporal sample alignment
#    No duplicated structural dataset
# ============================================================

def align_energy_samples(
    frame_indices,
    energy_dict,
    seq_len=16,
):
    """
    Only build:
      sample_starts [N]
      targets [N,4]
      target_indices [N]

    The actual structural window [t-15,...,t]
    is sliced from the existing frame cache in Dataset.__getitem__.
    """
    sample_starts = []
    target_list = []
    target_indices = []

    skipped_noncontinuous = 0
    skipped_missing_energy = 0

    n = len(
        frame_indices
    )

    print(
        "\n开始对齐 four-state Energy "
        "监督样本..."
    )

    for end in range(
        seq_len - 1,
        n,
    ):
        start = (
            end
            - seq_len
            + 1
        )

        window_indices = (
            frame_indices[
                start:end + 1
            ]
        )

        if not np.all(
            np.diff(
                window_indices
            ) == 1
        ):
            skipped_noncontinuous += 1
            continue

        target_idx = int(
            frame_indices[end]
        )

        if (
            target_idx
            not in energy_dict
        ):
            skipped_missing_energy += 1
            continue

        target_value = np.asarray(
            energy_dict[
                target_idx
            ],
            dtype=np.float64,
        )

        if target_value.shape != (
            EXPECTED_OUT_DIM,
        ):
            raise ValueError(
                f"CIF/frame index="
                f"{target_idx} "
                f"energy target shape="
                f"{target_value.shape}, "
                f"预期 "
                f"{(EXPECTED_OUT_DIM,)}"
            )

        if not np.all(
            np.isfinite(
                target_value
            )
        ):
            raise ValueError(
                f"frame index="
                f"{target_idx} "
                f"energy target 非有限"
            )

        sample_starts.append(
            start
        )
        target_list.append(
            target_value
        )
        target_indices.append(
            target_idx
        )

    sample_starts = np.asarray(
        sample_starts,
        dtype=np.int64,
    )

    targets_arr = np.asarray(
        target_list,
        dtype=np.float64,
    )

    target_indices = np.asarray(
        target_indices,
        dtype=np.int64,
    )

    if len(
        sample_starts
    ) < 2:
        raise ValueError(
            "有效 four-state Energy "
            "时序样本少于 2"
        )

    print(
        "\n" + "=" * 88
    )
    print(
        "Four-state Energy 样本对齐完成"
    )
    print(
        "注意：没有重新计算结构特征，"
        "也没有复制 [N,16,A,F] 数据"
    )
    print(
        "=" * 88
    )
    print(
        f"Sample starts        : "
        f"{sample_starts.shape}"
    )
    print(
        f"Target shape         : "
        f"{targets_arr.shape}"
    )
    print(
        f"有效样本数           : "
        f"{len(sample_starts)}"
    )
    print(
        f"不连续窗口跳过       : "
        f"{skipped_noncontinuous}"
    )
    print(
        f"缺少 energy          : "
        f"{skipped_missing_energy}"
    )
    print(
        f"Target index         : "
        f"{target_indices.min()} ~ "
        f"{target_indices.max()}"
    )

    for d, name in enumerate(
        ENERGY_NAMES
    ):
        print(
            f"{name} min/max          : "
            f"{targets_arr[:, d].min():.8e} / "
            f"{targets_arr[:, d].max():.8e}"
        )

    print("=" * 88)

    return (
        sample_starts,
        targets_arr,
        target_indices,
    )


# ============================================================
# 5. Same uniform Train/Test split
# ============================================================

def uniform_split_indices(
    total_num,
    train_ratio=0.15,
):
    if total_num < 2:
        raise ValueError(
            "至少需要 2 个有效样本"
        )

    if not (
        0.0
        < train_ratio
        < 1.0
    ):
        raise ValueError(
            "train_ratio 必须在 (0,1)"
        )

    train_num = int(
        round(
            total_num
            * train_ratio
        )
    )

    train_num = max(
        1,
        min(
            train_num,
            total_num - 1,
        ),
    )

    train_idx = np.unique(
        np.linspace(
            0,
            total_num - 1,
            train_num,
            dtype=int,
        )
    )

    test_idx = np.setdiff1d(
        np.arange(
            total_num
        ),
        train_idx,
    )

    if len(test_idx) == 0:
        raise ValueError(
            "测试集为空"
        )

    print(
        "采用原代码全局均匀间隔采样 -> "
        f"Train={len(train_idx)} "
        f"({len(train_idx)/total_num:.2%}), "
        f"Test={len(test_idx)} "
        f"({len(test_idx)/total_num:.2%})"
    )

    print(
        "前 20 个训练样本位置:",
        train_idx[:20].tolist(),
    )

    return (
        train_idx,
        test_idx,
    )


# ============================================================
# 6. Train-only structural scaling without materializing windows
# ============================================================

def fit_frame_scalers_from_training_windows(
    frame_atom_features,
    frame_atom_masks,
    frame_global_features,
    sample_starts,
    train_positions,
    seq_len,
):
    """
    Reproduces the same train-only statistics as fitting on
    materialized training windows, but without storing [N,S,A,F].
    """
    atom_scaler = StandardScaler()
    global_scaler = StandardScaler()

    for sample_pos in train_positions:
        start = int(
            sample_starts[
                sample_pos
            ]
        )

        end = (
            start
            + seq_len
        )

        atom_window = (
            frame_atom_features[
                start:end
            ]
        )

        mask_window = (
            frame_atom_masks[
                start:end
            ]
        )

        valid_atoms = (
            atom_window[
                mask_window
            ]
        )

        if (
            valid_atoms.ndim != 2
            or len(valid_atoms) == 0
        ):
            raise ValueError(
                "训练窗口中没有有效原子"
            )

        atom_scaler.partial_fit(
            valid_atoms
        )

        global_scaler.partial_fit(
            frame_global_features[
                start:end
            ]
        )

    return (
        atom_scaler,
        global_scaler,
    )


def transform_frame_atom_features(
    frame_atom_features,
    frame_atom_masks,
    scaler,
):
    out = np.zeros_like(
        frame_atom_features,
        dtype=np.float32,
    )

    valid = (
        frame_atom_features[
            frame_atom_masks
        ]
    )

    out[
        frame_atom_masks
    ] = scaler.transform(
        valid
    ).astype(
        np.float32
    )

    return out


def transform_frame_global_features(
    frame_global_features,
    scaler,
):
    return scaler.transform(
        frame_global_features
    ).astype(
        np.float32
    )


def scaler_to_dict(
    scaler,
):
    return {
        "mean_":
            scaler.mean_.tolist(),
        "scale_":
            scaler.scale_.tolist(),
        "var_":
            scaler.var_.tolist(),
        "n_features_in_":
            int(
                scaler.n_features_in_
            ),
    }


# ============================================================
# 7. Index-based Dataset
# ============================================================

class CachedWindowFourEnergyDataset(
    Dataset
):
    """
    Frame cache is stored once.

    Each item slices one 16-frame window on demand:
        [start:start+16]
    """

    def __init__(
        self,
        frame_atom_X_tensor,
        frame_atom_mask_tensor,
        frame_global_X_tensor,
        sample_starts,
        targets_scaled,
        targets_raw,
        cif_indices,
        subset_positions,
        seq_len,
    ):
        self.frame_atom_X = (
            frame_atom_X_tensor
        )

        self.frame_atom_mask = (
            frame_atom_mask_tensor
        )

        self.frame_global_X = (
            frame_global_X_tensor
        )

        self.sample_starts = np.asarray(
            sample_starts,
            dtype=np.int64,
        )

        self.targets_scaled = torch.tensor(
            targets_scaled,
            dtype=torch.float32,
        )

        self.targets_raw = torch.tensor(
            targets_raw,
            dtype=torch.float32,
        )

        self.cif_indices = np.asarray(
            cif_indices,
            dtype=np.int64,
        )

        self.subset_positions = np.asarray(
            subset_positions,
            dtype=np.int64,
        )

        self.seq_len = int(
            seq_len
        )

    def __len__(self):
        return len(
            self.subset_positions
        )

    def __getitem__(
        self,
        idx,
    ):
        sample_pos = int(
            self.subset_positions[
                idx
            ]
        )

        start = int(
            self.sample_starts[
                sample_pos
            ]
        )

        end = (
            start
            + self.seq_len
        )

        return (
            self.frame_atom_X[
                start:end
            ],
            self.frame_atom_mask[
                start:end
            ],
            self.frame_global_X[
                start:end
            ],
            self.targets_scaled[
                sample_pos
            ],
            self.targets_raw[
                sample_pos
            ],
            self.cif_indices[
                sample_pos
            ],
        )


# ============================================================
# 8. Frozen Vicuna four-state Energy model
# ============================================================

class SelfAttentionProjector(
    nn.Module
):
    """
    Structure projector with:
        1) Self-Attention
        2) Cross-Attention
        3) Fully Connected Layer

    The rest of the model is unchanged.

    Input:
        raw_frame: [B, S, input_dim]

    Output:
        structure pseudo-tokens: [B, S, Vicuna_hidden_size]

    Logic:
        raw_frame
        -> LayerNorm
        -> Linear(input_dim -> attention_dim)
        -> learned temporal position embedding
        -> TransformerEncoder (SELF-ATTENTION across the S structural frames)
        -> structure memory

        trainable query tokens
        -> CROSS-ATTENTION(
               query = trainable queries,
               key   = structure memory,
               value = structure memory
           )
        -> residual + LayerNorm

        -> Fully Connected Layer / FFN
        -> residual + LayerNorm
        -> Linear(attention_dim -> Vicuna hidden size)

    Important:
        * Self-attention models interactions among the 16 structural frames.
        * Cross-attention lets trainable query tokens retrieve information
          from the self-attended structural memory.
        * The number of query tokens is kept equal to seq_len, therefore the
          downstream Vicuna input still receives S=16 structural pseudo-tokens.
        * Vicuna remains fully frozen.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seq_len: int,
        attention_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        if attention_dim % num_heads != 0:
            raise ValueError(
                f"attention_dim={attention_dim} 必须能被 "
                f"num_heads={num_heads} 整除"
            )

        self.input_dim = int(
            input_dim
        )

        self.output_dim = int(
            output_dim
        )

        self.seq_len = int(
            seq_len
        )

        self.attention_dim = int(
            attention_dim
        )

        self.num_heads = int(
            num_heads
        )

        self.num_layers = int(
            num_layers
        )

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

        # Temporal order for the 16-frame structural sequence.
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
            num_layers=self.num_layers,
        )

        self.memory_norm = nn.LayerNorm(
            self.attention_dim
        )

        # ----------------------------------------------------
        # C. Trainable queries + CROSS-ATTENTION
        # ----------------------------------------------------
        # Keep one query for each structural frame so the projector output
        # remains [B, S, D] and all downstream Vicuna logic stays unchanged.
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
        # D. Fully Connected Layer / Feed-Forward Network
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

    def forward(
        self,
        x,
    ):
        # x: [B, S, input_dim]
        if x.ndim != 3:
            raise ValueError(
                f"SelfAttentionProjector 输入必须为 [B,S,D]，"
                f"实际 {tuple(x.shape)}"
            )

        B, S, D = x.shape

        if S != self.seq_len:
            raise ValueError(
                f"SelfAttentionProjector seq_len={self.seq_len}，"
                f"但当前输入 S={S}"
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
        #    The 16 structural frames communicate with each other.
        # ----------------------------------------------------
        memory = self.self_attention(
            h
        )

        memory = self.memory_norm(
            memory
        )

        # ----------------------------------------------------
        # 3. CROSS-ATTENTION
        #    Q: trainable query tokens
        #    K,V: self-attended structural memory
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
        # 4. FULLY CONNECTED LAYER / FFN
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



class FrozenVicunaFourStateEnergy(
    nn.Module
):
    def __init__(
        self,
        model_path: str,
        atom_dim: int,
        global_dim: int,
        prompt: str,
        seq_len: int,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self.model_path = (
            model_path
        )

        self.atom_dim = int(
            atom_dim
        )

        self.global_dim = int(
            global_dim
        )

        self.out_dim = (
            EXPECTED_OUT_DIM
        )

        self.prompt = prompt

        self.seq_len = int(
            seq_len
        )

        self.tokenizer = (
            AutoTokenizer
            .from_pretrained(
                model_path,
                use_fast=False,
            )
        )

        if (
            self.tokenizer
            .pad_token_id
            is None
        ):
            self.tokenizer.pad_token = (
                self.tokenizer
                .eos_token
            )

        if (
            torch.cuda.is_available()
            and torch.cuda
            .is_bf16_supported()
        ):
            self.llm_dtype = (
                torch.bfloat16
            )
        else:
            self.llm_dtype = (
                torch.float16
            )

        print(
            f"加载 Vicuna: "
            f"{model_path}"
        )

        print(
            f"LLM dtype: "
            f"{self.llm_dtype}"
        )

        self.llm = (
            AutoModelForCausalLM
            .from_pretrained(
                model_path,
                torch_dtype=
                    self.llm_dtype,
                low_cpu_mem_usage=True,
            )
        )

        self.llm.config.use_cache = (
            False
        )

        # Freeze ALL Vicuna parameters.
        for p in (
            self.llm.parameters()
        ):
            p.requires_grad = False

        self.llm.eval()

        if use_gradient_checkpointing:
            try:
                self.llm.gradient_checkpointing_enable()
                print(
                    "gradient checkpointing: ON"
                )
            except Exception as e:
                print(
                    "⚠️ gradient checkpointing "
                    f"开启失败: {e}"
                )
        else:
            print(
                "gradient checkpointing: OFF"
            )

        self.hidden_size = int(
            self.llm.config
            .hidden_size
        )

        raw_frame_dim = (
            self.atom_dim
            + self.global_dim
        )

        # ONLY architecture change:
        # original frame-wise MLP/Linear projector
        # -> SELF-ATTENTION + CROSS-ATTENTION + Fully Connected projector.
        self.self_attention_projector = SelfAttentionProjector(
            input_dim=raw_frame_dim,
            output_dim=self.hidden_size,
            seq_len=self.seq_len,
            attention_dim=256,
            num_heads=8,
            num_layers=2,
            dropout=0.1,
        )

        # Fixed four-state output.
        self.energy_head = nn.Sequential(
            nn.LayerNorm(
                self.hidden_size
            ),
            nn.Linear(
                self.hidden_size,
                512,
            ),
            nn.GELU(),
            nn.Linear(
                512,
                128,
            ),
            nn.GELU(),
            nn.Linear(
                128,
                EXPECTED_OUT_DIM,
            ),
        )

        prompt_ids = (
            self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=True,
            )["input_ids"]
        )

        self.register_buffer(
            "prompt_ids",
            prompt_ids,
            persistent=True,
        )

        print(
            f"Vicuna hidden_size   : "
            f"{self.hidden_size}"
        )

        print(
            f"Prompt token count   : "
            f"{prompt_ids.shape[1]}"
        )

        print(
            f"Raw frame dim        : "
            f"atom({self.atom_dim}) + "
            f"global({self.global_dim}) = "
            f"{raw_frame_dim}"
        )

        print(
            f"Projector            : "
            f"Self-Attention + Cross-Attention + FC "
            f"(dim=256, heads=8, self_layers=2)"
        )

        print(
            f"Structure tokens     : "
            f"one token per frame, "
            f"seq_len={self.seq_len}"
        )

        print(
            "Energy head          : "
            "[E1,E2,E3,E4]"
        )

    def train(
        self,
        mode: bool = True,
    ):
        super().train(
            mode
        )

        # Frozen LLM always eval.
        self.llm.eval()

        return self

    def trainable_parameters(
        self,
    ):
        modules = [
            self.self_attention_projector,
            self.energy_head,
        ]

        for module in modules:
            for p in (
                module.parameters()
            ):
                yield p

    def trainable_state_dict(
        self,
    ):
        return {
            "self_attention_projector":
                self.self_attention_projector.state_dict(),
            "energy_head":
                self.energy_head.state_dict(),
        }

    def load_trainable_state_dict(
        self,
        state,
    ):
        self.self_attention_projector.load_state_dict(
            state["self_attention_projector"]
        )

        self.energy_head.load_state_dict(
            state["energy_head"]
        )

    def forward(
        self,
        atom_x,
        atom_mask,
        global_x,
    ):
        # atom_x    [B,S,A,F]
        # atom_mask [B,S,A]
        # global_x  [B,S,G]

        B, S, A, Fdim = (
            atom_x.shape
        )

        # Same non-trainable masked atom mean as the supplied Vicuna baseline.
        mask_f = (
            atom_mask
            .unsqueeze(-1)
            .to(
                atom_x.dtype
            )
        )

        atom_sum = (
            atom_x
            * mask_f
        ).sum(
            dim=2
        )

        atom_count = (
            mask_f.sum(
                dim=2
            )
            .clamp_min(
                1.0
            )
        )

        atom_mean = (
            atom_sum
            / atom_count
        )

        raw_frame = torch.cat(
            [
                atom_mean,
                global_x,
            ],
            dim=-1,
        )

        struct_tokens_fp32 = (
            self.self_attention_projector(
                raw_frame
            )
        )

        struct_tokens = (
            struct_tokens_fp32.to(
                dtype=
                    self.llm_dtype
            )
        )

        ids = (
            self.prompt_ids
            .expand(
                B,
                -1,
            )
        )

        text_emb = (
            self.llm
            .get_input_embeddings()(
                ids
            )
        )

        bos_id = (
            self.tokenizer
            .bos_token_id
        )

        has_bos = bool(
            ids.shape[1] > 0
            and bos_id is not None
            and torch.all(
                ids[:, 0]
                == bos_id
            )
        )

        if has_bos:
            input_embeds = (
                torch.cat(
                    [
                        text_emb[:, :1],
                        struct_tokens,
                        text_emb[:, 1:],
                    ],
                    dim=1,
                )
            )
        else:
            input_embeds = (
                torch.cat(
                    [
                        struct_tokens,
                        text_emb,
                    ],
                    dim=1,
                )
            )

        attention_mask = torch.ones(
            input_embeds.shape[:2],
            dtype=torch.long,
            device=
                input_embeds.device,
        )

        # IMPORTANT:
        # Do NOT use torch.no_grad() here during training.
        # Vicuna is frozen, but gradient must flow through it
        # back to the self-attention projector.
        outputs = self.llm.model(
            inputs_embeds=
                input_embeds,
            attention_mask=
                attention_mask,
            use_cache=False,
            return_dict=True,
        )

        h = (
            outputs
            .last_hidden_state[
                :, -1, :
            ]
            .float()
        )

        pred_scaled = (
            self.energy_head(
                h
            )
        )

        return pred_scaled


# ============================================================
# 9. Loss / train / inference
# ============================================================

class EnergyRegressionLoss(
    nn.Module
):
    def __init__(
        self,
        beta=0.5,
    ):
        super().__init__()
        self.beta = float(
            beta
        )

    def forward(
        self,
        pred_scaled,
        target_scaled,
    ):
        return F.smooth_l1_loss(
            pred_scaled,
            target_scaled,
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
    grad_accum_steps=1,
):
    if is_train:
        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )
    else:
        model.eval()

    total_loss = 0.0

    pred_scaled_all = []
    true_scaled_all = []

    trainable_params = list(
        model.trainable_parameters()
    )

    with torch.set_grad_enabled(
        is_train
    ):
        for step, batch in enumerate(
            loader,
            start=1,
        ):
            (
                atom_x,
                atom_mask,
                global_x,
                target_scaled,
                _,
                _,
            ) = batch

            atom_x = atom_x.to(
                device,
                non_blocking=True,
            )

            atom_mask = atom_mask.to(
                device,
                non_blocking=True,
            )

            global_x = global_x.to(
                device,
                non_blocking=True,
            )

            target_scaled = (
                target_scaled.to(
                    device,
                    non_blocking=True,
                )
            )

            pred_scaled = model(
                atom_x,
                atom_mask,
                global_x,
            )

            loss = criterion(
                pred_scaled,
                target_scaled,
            )

            if is_train:
                (
                    loss
                    / grad_accum_steps
                ).backward()

                do_step = (
                    step
                    % grad_accum_steps
                    == 0
                ) or (
                    step
                    == len(loader)
                )

                if do_step:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params,
                        max_norm=1.0,
                    )

                    optimizer.step()

                    optimizer.zero_grad(
                        set_to_none=True
                    )

            bs = atom_x.size(
                0
            )

            total_loss += (
                float(
                    loss
                    .detach()
                    .item()
                )
                * bs
            )

            pred_scaled_all.append(
                pred_scaled
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            true_scaled_all.append(
                target_scaled
                .detach()
                .float()
                .cpu()
                .numpy()
            )

    pred_scaled_all = np.concatenate(
        pred_scaled_all,
        axis=0,
    )

    true_scaled_all = np.concatenate(
        true_scaled_all,
        axis=0,
    )

    pred_real = (
        target_scaler
        .inverse_transform(
            pred_scaled_all
        )
    )

    true_real = (
        target_scaler
        .inverse_transform(
            true_scaled_all
        )
    )

    metrics = (
        calc_multidim_metrics(
            true_real,
            pred_real,
            ENERGY_NAMES,
        )
    )

    return {
        "loss":
            total_loss
            / len(loader.dataset),
        "mae":
            metrics["mae"],
        "rmse":
            metrics["rmse"],
        "r2":
            metrics["r2"],
        "per_dim":
            metrics["per_dim"],
    }


@torch.no_grad()
def predict_loader(
    model,
    loader,
    device,
    target_scaler,
):
    model.eval()

    true_scaled_all = []
    pred_scaled_all = []
    idx_all = []

    for (
        atom_x,
        atom_mask,
        global_x,
        target_scaled,
        _,
        idx,
    ) in loader:

        atom_x = atom_x.to(
            device,
            non_blocking=True,
        )

        atom_mask = atom_mask.to(
            device,
            non_blocking=True,
        )

        global_x = global_x.to(
            device,
            non_blocking=True,
        )

        pred_scaled = model(
            atom_x,
            atom_mask,
            global_x,
        )

        true_scaled_all.append(
            target_scaled.numpy()
        )

        pred_scaled_all.append(
            pred_scaled
            .float()
            .cpu()
            .numpy()
        )

        idx_all.append(
            np.asarray(idx)
        )

    true_scaled_all = np.concatenate(
        true_scaled_all,
        axis=0,
    )

    pred_scaled_all = np.concatenate(
        pred_scaled_all,
        axis=0,
    )

    return (
        target_scaler
        .inverse_transform(
            true_scaled_all
        ),
        target_scaler
        .inverse_transform(
            pred_scaled_all
        ),
        np.concatenate(
            idx_all,
            axis=0,
        ),
    )


# ============================================================
# 10. Visualization
# ============================================================

def plot_training_history(
    history,
    filename,
):
    epochs = np.asarray(
        history["epoch"]
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
    )

    axes[0, 0].plot(
        epochs,
        history["train_loss"],
        label="Train Loss",
    )

    axes[0, 0].plot(
        epochs,
        history["test_loss"],
        label="Test Loss",
        linestyle="--",
    )

    axes[0, 0].set_title(
        "Four-state Energy "
        "SmoothL1 Loss (normalized)"
    )

    axes[0, 1].plot(
        epochs,
        history["train_mae"],
        label="Train MAE",
    )

    axes[0, 1].plot(
        epochs,
        history["test_mae"],
        label="Test MAE",
        linestyle="--",
    )

    axes[0, 1].set_title(
        "Four-state Energy Macro MAE"
    )

    axes[1, 0].plot(
        epochs,
        history["train_r2"],
        label="Train R2",
    )

    axes[1, 0].plot(
        epochs,
        history["test_r2"],
        label="Test R2",
        linestyle="--",
    )

    axes[1, 0].set_title(
        "Four-state Energy Macro R2"
    )

    axes[1, 1].plot(
        epochs,
        history["train_rmse"],
        label="Train RMSE",
    )

    axes[1, 1].plot(
        epochs,
        history["test_rmse"],
        label="Test RMSE",
        linestyle="--",
    )

    axes[1, 1].set_title(
        "Four-state Energy Macro RMSE"
    )

    for ax in axes.flat:
        ax.set_xlabel(
            "Epoch"
        )

        ax.legend()

        ax.grid(
            True,
            linestyle=":",
            alpha=0.6,
        )

    plt.tight_layout()

    fig.savefig(
        filename,
        dpi=300,
    )

    plt.close(fig)


def plot_energy_comparison(
    targets,
    preds,
    title_prefix,
    filename,
):
    targets = np.asarray(
        targets,
        dtype=np.float64,
    )

    preds = np.asarray(
        preds,
        dtype=np.float64,
    )

    fig, axes = plt.subplots(
        EXPECTED_OUT_DIM,
        2,
        figsize=(
            15,
            5 * EXPECTED_OUT_DIM,
        ),
        squeeze=False,
    )

    for d, name in enumerate(
        ENERGY_NAMES
    ):
        yt = targets[:, d]
        yp = preds[:, d]

        mae = float(
            np.mean(
                np.abs(
                    yp - yt
                )
            )
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    (
                        yp - yt
                    ) ** 2
                )
            )
        )

        r2 = safe_r2(
            yt,
            yp,
        )

        order = np.argsort(
            yt
        )

        x = np.arange(
            1,
            len(yt) + 1,
        )

        axes[d, 0].plot(
            x,
            yt[order],
            label=f"True {name}",
            linewidth=1.5,
        )

        axes[d, 0].plot(
            x,
            yp[order],
            label=f"Pred {name}",
            linewidth=1.5,
            linestyle="--",
        )

        axes[d, 0].set_xlabel(
            f"Samples sorted by true {name}"
        )

        axes[d, 0].set_ylabel(
            f"{name} "
            f"({ENERGY_UNIT_LABEL})"
        )

        axes[d, 0].set_title(
            f"[{title_prefix}] "
            f"{name}: True vs Pred"
        )

        axes[d, 0].legend()

        axes[d, 0].grid(
            True,
            linestyle=":",
            alpha=0.6,
        )

        axes[d, 1].scatter(
            yt,
            yp,
            alpha=0.65,
        )

        lo = float(
            min(
                np.min(yt),
                np.min(yp),
            )
        )

        hi = float(
            max(
                np.max(yt),
                np.max(yp),
            )
        )

        if np.isclose(
            lo,
            hi,
        ):
            hi = (
                lo
                + 1.0e-12
            )

        axes[d, 1].plot(
            [lo, hi],
            [lo, hi],
            "k--",
            linewidth=2,
            label="y=x",
        )

        axes[d, 1].set_xlabel(
            f"True {name} "
            f"({ENERGY_UNIT_LABEL})"
        )

        axes[d, 1].set_ylabel(
            f"Pred {name} "
            f"({ENERGY_UNIT_LABEL})"
        )

        axes[d, 1].set_title(
            f"[{title_prefix}] "
            f"Parity {name}"
        )

        axes[d, 1].legend(
            loc="upper left"
        )

        axes[d, 1].grid(
            True,
            linestyle=":",
            alpha=0.6,
        )

        axes[d, 1].text(
            0.95,
            0.05,
            (
                f"R2 = {r2:.6f}\n"
                f"MAE = {mae:.6e}\n"
                f"RMSE = {rmse:.6e}"
            ),
            transform=
                axes[d, 1].transAxes,
            ha="right",
            va="bottom",
            bbox=dict(
                boxstyle="round",
                alpha=0.8,
            ),
        )

    plt.tight_layout()

    fig.savefig(
        filename,
        dpi=300,
    )

    plt.close(fig)


def plot_full_time_series(
    full_true,
    train_positions,
    train_pred,
    test_positions,
    test_pred,
    file_indices,
    filename,
):
    full_true = np.asarray(
        full_true,
        dtype=np.float64,
    )

    train_pred = np.asarray(
        train_pred,
        dtype=np.float64,
    )

    test_pred = np.asarray(
        test_pred,
        dtype=np.float64,
    )

    file_indices = np.asarray(
        file_indices,
        dtype=np.int64,
    ).reshape(-1)

    fig, axes = plt.subplots(
        EXPECTED_OUT_DIM,
        1,
        figsize=(
            16,
            4.8 * EXPECTED_OUT_DIM,
        ),
        squeeze=False,
    )

    axes = axes[:, 0]

    for d, name in enumerate(
        ENERGY_NAMES
    ):
        axes[d].plot(
            file_indices,
            full_true[:, d],
            linewidth=1.2,
            label=f"True {name}",
        )

        axes[d].scatter(
            file_indices[
                train_positions
            ],
            train_pred[:, d],
            s=12,
            alpha=0.8,
            label="Train Pred",
        )

        axes[d].scatter(
            file_indices[
                test_positions
            ],
            test_pred[:, d],
            s=16,
            alpha=0.8,
            label="Test Pred",
        )

        axes[d].set_xlabel(
            "MD Step / Target CIF Index"
        )

        axes[d].set_ylabel(
            f"{name} "
            f"({ENERGY_UNIT_LABEL})"
        )

        axes[d].set_title(
            f"Full MD Energy Evolution - "
            f"{name}"
        )

        axes[d].legend(
            loc="upper right"
        )

        axes[d].grid(
            True,
            linestyle=":",
            alpha=0.5,
        )

    plt.tight_layout()

    fig.savefig(
        filename,
        dpi=300,
    )

    plt.close(fig)


# ============================================================
# 11. CSV saving
# ============================================================

def save_predictions(
    true_energy,
    pred_energy,
    file_indices,
    filename,
):
    true_energy = np.asarray(
        true_energy,
        dtype=np.float64,
    )

    pred_energy = np.asarray(
        pred_energy,
        dtype=np.float64,
    )

    idx = np.asarray(
        file_indices,
        dtype=np.int64,
    ).reshape(-1)

    data = {
        "CIF_Index": idx,
    }

    for d, name in enumerate(
        ENERGY_NAMES
    ):
        data[
            f"True_{name}"
        ] = true_energy[:, d]

        data[
            f"Pred_{name}"
        ] = pred_energy[:, d]

        data[
            f"Error_{name}"
        ] = (
            pred_energy[:, d]
            - true_energy[:, d]
        )

        data[
            f"AbsError_{name}"
        ] = np.abs(
            pred_energy[:, d]
            - true_energy[:, d]
        )

    pd.DataFrame(
        data
    ).sort_values(
        "CIF_Index"
    ).to_csv(
        filename,
        index=False,
        float_format="%.12e",
    )

    print(
        f"结果已保存: {filename}"
    )


def save_full_time_series_csv(
    full_true,
    train_positions,
    train_pred,
    test_positions,
    test_pred,
    file_indices,
    filename,
):
    full_true = np.asarray(
        full_true,
        dtype=np.float64,
    )

    train_pred = np.asarray(
        train_pred,
        dtype=np.float64,
    )

    test_pred = np.asarray(
        test_pred,
        dtype=np.float64,
    )

    file_indices = np.asarray(
        file_indices,
        dtype=np.int64,
    ).reshape(-1)

    total_len = len(
        full_true
    )

    data = {
        "Dataset_Position":
            np.arange(
                total_len
            ),
        "CIF_Index":
            file_indices,
    }

    for d, name in enumerate(
        ENERGY_NAMES
    ):
        train_col = np.full(
            total_len,
            np.nan,
        )

        test_col = np.full(
            total_len,
            np.nan,
        )

        combined_col = np.full(
            total_len,
            np.nan,
        )

        train_col[
            train_positions
        ] = train_pred[:, d]

        test_col[
            test_positions
        ] = test_pred[:, d]

        combined_col[
            train_positions
        ] = train_pred[:, d]

        combined_col[
            test_positions
        ] = test_pred[:, d]

        data[
            f"True_{name}"
        ] = full_true[:, d]

        data[
            f"Train_Pred_{name}"
        ] = train_col

        data[
            f"Test_Pred_{name}"
        ] = test_col

        data[
            f"Combined_Pred_{name}"
        ] = combined_col

        data[
            f"Combined_AbsError_{name}"
        ] = np.abs(
            combined_col
            - full_true[:, d]
        )

    pd.DataFrame(
        data
    ).sort_values(
        "CIF_Index"
    ).to_csv(
        filename,
        index=False,
        float_format="%.12e",
    )

    print(
        f"全局 four-state Energy "
        f"evolution CSV 已保存: "
        f"{filename}"
    )


def save_metrics_per_state(
    true_energy,
    pred_energy,
    split_name,
    filename,
):
    metrics = (
        calc_multidim_metrics(
            true_energy,
            pred_energy,
            ENERGY_NAMES,
        )
    )

    rows = []

    for m in metrics[
        "per_dim"
    ]:
        rows.append({
            "Split":
                split_name,
            "State":
                m["name"],
            "MAE":
                m["mae"],
            "RMSE":
                m["rmse"],
            "R2":
                m["r2"],
        })

    pd.DataFrame(
        rows
    ).to_csv(
        filename,
        index=False,
    )


# ============================================================
# 12. CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--run-dir",
        default=DEFAULT_RUN_DIR,
    )

    p.add_argument(
        "--cache-file",
        default=None,
    )

    p.add_argument(
        "--energy-file",
        default=None,
    )

    p.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
    )

    p.add_argument(
        "--output-root",
        default=None,
    )

    p.add_argument(
        "--seq-len",
        type=int,
        default=SEQ_LEN,
    )

    p.add_argument(
        "--train-ratio",
        type=float,
        default=TRAIN_RATIO,
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )

    p.add_argument(
        "--grad-accum",
        type=int,
        default=GRAD_ACCUM,
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
    )

    p.add_argument(
        "--lr",
        type=float,
        default=LR,
    )

    p.add_argument(
        "--weight-decay",
        type=float,
        default=WEIGHT_DECAY,
    )

    p.add_argument(
        "--patience",
        type=int,
        default=PATIENCE,
    )

    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    p.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help=(
            "默认关闭。若 batch=32 OOM，"
            "可以开启以省显存，但速度会下降。"
        ),
    )

    p.add_argument(
        "--prompt",
        default=(
            "Predict the four electronic-state energies "
            "E1, E2, E3, and E4 at the target frame "
            "from the 16-frame invariant structural sequence."
        ),
    )

    return p.parse_args()


# ============================================================
# 13. Main
# ============================================================

def main():
    args = parse_args()

    set_seed(SEED)

    run_dir = (
        args.run_dir
    )

    cache_file = (
        args.cache_file
        or os.path.join(
            run_dir,
            "atomwise_invariant_v3_Z16-42.pt",
        )
    )

    energy_file = (
        args.energy_file
        or os.path.join(
            run_dir,
            "energy",
            "energy",
        )
    )

    output_root = (
        args.output_root
        or os.path.join(
            run_dir,
            "outputs_vicuna_four_state_energy",
        )
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "本脚本需要 CUDA GPU "
            "运行 Vicuna-7B"
        )

    device = torch.device(
        "cuda"
    )

    model_dir = os.path.join(
        output_root,
        "models",
    )

    log_dir = os.path.join(
        output_root,
        "logs",
    )

    plot_dir = os.path.join(
        output_root,
        "plots",
    )

    result_dir = os.path.join(
        output_root,
        "results",
    )

    for d in [
        model_dir,
        log_dir,
        plot_dir,
        result_dir,
    ]:
        os.makedirs(
            d,
            exist_ok=True,
        )

    print("=" * 108)
    print(
        "Existing invariant cache + "
        "frozen Vicuna-7B + "
        "FOUR-state Energy regression"
    )
    print("=" * 108)

    print(
        f"Run dir       : "
        f"{run_dir}"
    )

    print(
        f"Cache         : "
        f"{cache_file}"
    )

    print(
        f"Energy        : "
        f"{energy_file}"
    )

    print(
        f"Vicuna        : "
        f"{args.model_path}"
    )

    print(
        f"Output        : "
        f"{output_root}"
    )

    print(
        f"Targets       : "
        f"{ENERGY_NAMES}"
    )

    print(
        f"SEQ_LEN       : "
        f"{args.seq_len}"
    )

    print(
        f"TRAIN_RATIO   : "
        f"{args.train_ratio:.2%}"
    )

    print(
        f"Batch/accum   : "
        f"{args.batch_size}/"
        f"{args.grad_accum}"
    )

    print(
        f"GPU           : "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        "Structure     : "
        "DIRECT CACHE LOAD; "
        "NO CIF; "
        "NO descriptor recomputation"
    )

    print("=" * 108)

    # --------------------------------------------------------
    # 1. Direct load existing structural cache
    # --------------------------------------------------------
    (
        frame_atom_features,
        frame_atom_masks,
        frame_global_features,
        frame_indices,
        descriptor_state,
    ) = load_feature_cache(
        cache_file
    )

    # --------------------------------------------------------
    # 2. Read exactly E1,E2,E3,E4
    # --------------------------------------------------------
    (
        energy_dict,
        out_dim,
    ) = read_four_state_energy_file(
        energy_file,
        frame_indices.tolist(),
    )

    if (
        out_dim
        != EXPECTED_OUT_DIM
    ):
        raise ValueError(
            f"out_dim 必须是 4，"
            f"实际 {out_dim}"
        )

    # --------------------------------------------------------
    # 3. Lightweight 16-frame sample alignment
    # --------------------------------------------------------
    (
        sample_starts,
        targets_np,
        target_cif_indices,
    ) = align_energy_samples(
        frame_indices,
        energy_dict,
        seq_len=args.seq_len,
    )

    num_samples = len(
        sample_starts
    )

    # --------------------------------------------------------
    # 4. Exact same uniform split logic
    # --------------------------------------------------------
    (
        train_positions,
        test_positions,
    ) = uniform_split_indices(
        num_samples,
        train_ratio=
            args.train_ratio,
    )

    # --------------------------------------------------------
    # 5. Train-only structural scaling
    # --------------------------------------------------------
    (
        atom_scaler,
        global_scaler,
    ) = fit_frame_scalers_from_training_windows(
        frame_atom_features,
        frame_atom_masks,
        frame_global_features,
        sample_starts,
        train_positions,
        args.seq_len,
    )

    frame_atom_scaled = (
        transform_frame_atom_features(
            frame_atom_features,
            frame_atom_masks,
            atom_scaler,
        )
    )

    frame_global_scaled = (
        transform_frame_global_features(
            frame_global_features,
            global_scaler,
        )
    )

    # --------------------------------------------------------
    # 6. Train-only four-energy target scaling
    # --------------------------------------------------------
    target_scaler = (
        StandardScaler()
    )

    target_scaler.fit(
        targets_np[
            train_positions
        ]
    )

    targets_scaled_all = (
        target_scaler
        .transform(
            targets_np
        )
        .astype(
            np.float32
        )
    )

    print("\n训练集统计:")

    print(
        f"Atom feature dim    : "
        f"{frame_atom_scaled.shape[-1]}"
    )

    print(
        f"Global feature dim  : "
        f"{frame_global_scaled.shape[-1]}"
    )

    print(
        f"Max atoms           : "
        f"{frame_atom_scaled.shape[1]}"
    )

    print(
        f"Energy OUT_DIM      : "
        f"{EXPECTED_OUT_DIM}"
    )

    print(
        f"Train samples       : "
        f"{len(train_positions)}"
    )

    print(
        f"Test samples        : "
        f"{len(test_positions)}"
    )

    for d, name in enumerate(
        ENERGY_NAMES
    ):
        print(
            f"{name} train mean/std : "
            f"{target_scaler.mean_[d]:.12e} / "
            f"{target_scaler.scale_[d]:.12e}"
        )

    # --------------------------------------------------------
    # 7. Shared frame tensors stored ONCE
    # --------------------------------------------------------
    frame_atom_tensor = (
        torch.from_numpy(
            frame_atom_scaled
        )
    )

    frame_mask_tensor = (
        torch.from_numpy(
            frame_atom_masks
        )
    )

    frame_global_tensor = (
        torch.from_numpy(
            frame_global_scaled
        )
    )

    train_dataset = (
        CachedWindowFourEnergyDataset(
            frame_atom_tensor,
            frame_mask_tensor,
            frame_global_tensor,
            sample_starts,
            targets_scaled_all,
            targets_np,
            target_cif_indices,
            train_positions,
            args.seq_len,
        )
    )

    test_dataset = (
        CachedWindowFourEnergyDataset(
            frame_atom_tensor,
            frame_mask_tensor,
            frame_global_tensor,
            sample_starts,
            targets_scaled_all,
            targets_np,
            target_cif_indices,
            test_positions,
            args.seq_len,
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=
            args.batch_size,
        shuffle=True,
        num_workers=
            args.num_workers,
        pin_memory=True,
    )

    eval_train_loader = DataLoader(
        train_dataset,
        batch_size=
            args.batch_size,
        shuffle=False,
        num_workers=
            args.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=
            args.batch_size,
        shuffle=False,
        num_workers=
            args.num_workers,
        pin_memory=True,
    )

    # --------------------------------------------------------
    # 8. Frozen Vicuna four-output model
    # --------------------------------------------------------
    model = (
        FrozenVicunaFourStateEnergy(
            model_path=
                args.model_path,
            atom_dim=
                frame_atom_scaled
                .shape[-1],
            global_dim=
                frame_global_scaled
                .shape[-1],
            prompt=
                args.prompt,
            seq_len=
                args.seq_len,
            use_gradient_checkpointing=
                args.gradient_checkpointing,
        ).to(
            device
        )
    )

    trainable_params = list(
        model.trainable_parameters()
    )

    n_trainable = sum(
        p.numel()
        for p in trainable_params
    )

    n_llm = sum(
        p.numel()
        for p in (
            model.llm.parameters()
        )
    )

    print(
        f"\nLLM params (frozen): "
        f"{n_llm:,}"
    )

    print(
        f"Trainable params    : "
        f"{n_trainable:,}"
    )

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=
            args.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=35,
            min_lr=1.0e-6,
        )
    )

    criterion = (
        EnergyRegressionLoss(
            beta=
                SMOOTH_L1_BETA
        ).to(
            device
        )
    )

    # --------------------------------------------------------
    # 9. Save scalers / config
    # --------------------------------------------------------
    np.savez(
        os.path.join(
            model_dir,
            "scalers.npz",
        ),
        atom_feature_mean=
            atom_scaler.mean_,
        atom_feature_scale=
            atom_scaler.scale_,
        atom_feature_var=
            atom_scaler.var_,
        global_feature_mean=
            global_scaler.mean_,
        global_feature_scale=
            global_scaler.scale_,
        global_feature_var=
            global_scaler.var_,
        target_mean=
            target_scaler.mean_,
        target_scale=
            target_scaler.scale_,
        target_var=
            target_scaler.var_,
    )

    with open(
        os.path.join(
            model_dir,
            "four_state_config.json",
        ),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "energy_names":
                    ENERGY_NAMES,
                "out_dim":
                    EXPECTED_OUT_DIM,
                "seq_len":
                    args.seq_len,
                "train_ratio":
                    args.train_ratio,
                "batch_size":
                    args.batch_size,
                "grad_accum":
                    args.grad_accum,
                "vicuna_frozen":
                    True,
                "projector_type":
                    "self_attention",
                "projector_attention_dim":
                    256,
                "projector_num_heads":
                    8,
                "projector_num_layers":
                    2,
                "vicuna_model_path":
                    args.model_path,
                "prompt":
                    args.prompt,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # 10. Training
    # --------------------------------------------------------
    history = {
        "epoch": [],
        "lr": [],
        "train_loss": [],
        "test_loss": [],
        "train_mae": [],
        "test_mae": [],
        "train_rmse": [],
        "test_rmse": [],
        "train_r2": [],
        "test_r2": [],
    }

    best_test_loss = (
        float("inf")
    )

    epochs_without_improvement = (
        0
    )

    best_model_path = os.path.join(
        model_dir,
        "best_frozen_vicuna_four_state_energy.pth",
    )

    print(
        "\n" + "=" * 108
    )

    print(
        "开始训练："
        "self-attention projector + four-state regression head 更新；"
        "Vicuna-7B 完全冻结"
    )

    print(
        "=" * 108
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        train_m = process_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            target_scaler,
            is_train=True,
            grad_accum_steps=
                args.grad_accum,
        )

        test_m = process_epoch(
            model,
            test_loader,
            optimizer,
            criterion,
            device,
            target_scaler,
            is_train=False,
            grad_accum_steps=1,
        )

        scheduler.step(
            test_m["loss"]
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        history["epoch"].append(
            epoch
        )

        history["lr"].append(
            current_lr
        )

        history[
            "train_loss"
        ].append(
            train_m["loss"]
        )

        history[
            "test_loss"
        ].append(
            test_m["loss"]
        )

        history[
            "train_mae"
        ].append(
            train_m["mae"]
        )

        history[
            "test_mae"
        ].append(
            test_m["mae"]
        )

        history[
            "train_rmse"
        ].append(
            train_m["rmse"]
        )

        history[
            "test_rmse"
        ].append(
            test_m["rmse"]
        )

        history[
            "train_r2"
        ].append(
            train_m["r2"]
        )

        history[
            "test_r2"
        ].append(
            test_m["r2"]
        )

        print(
            f"Epoch "
            f"{epoch:04d}/"
            f"{args.epochs} | "
            f"Loss "
            f"{train_m['loss']:.5f}/"
            f"{test_m['loss']:.5f} | "
            f"MAE "
            f"{train_m['mae']:.3e}/"
            f"{test_m['mae']:.3e} | "
            f"RMSE "
            f"{train_m['rmse']:.3e}/"
            f"{test_m['rmse']:.3e} | "
            f"R2 "
            f"{train_m['r2']:.5f}/"
            f"{test_m['r2']:.5f} | "
            f"LR={current_lr:.2e}"
        )

        if (
            epoch == 1
            or epoch % 20 == 0
        ):
            for m in (
                test_m["per_dim"]
            ):
                print(
                    f"    Test "
                    f"{m['name']}: "
                    f"MAE="
                    f"{m['mae']:.3e}, "
                    f"RMSE="
                    f"{m['rmse']:.3e}, "
                    f"R2="
                    f"{m['r2']:.6f}"
                )

        if (
            test_m["loss"]
            < best_test_loss
            - MIN_DELTA
        ):
            best_test_loss = (
                test_m["loss"]
            )

            epochs_without_improvement = (
                0
            )

            checkpoint = {
                "trainable_state_dict":
                    model
                    .trainable_state_dict(),
                "model_path":
                    args.model_path,
                "hidden_size":
                    int(
                        model.hidden_size
                    ),
                "atom_dim":
                    int(
                        frame_atom_scaled
                        .shape[-1]
                    ),
                "global_dim":
                    int(
                        frame_global_scaled
                        .shape[-1]
                    ),
                "out_dim":
                    EXPECTED_OUT_DIM,
                "energy_names":
                    ENERGY_NAMES,
                "seq_len":
                    int(
                        args.seq_len
                    ),
                "train_ratio":
                    float(
                        args.train_ratio
                    ),
                "batch_size":
                    int(
                        args.batch_size
                    ),
                "grad_accum":
                    int(
                        args.grad_accum
                    ),
                "prompt":
                    args.prompt,
                "best_epoch":
                    int(epoch),
                "best_test_loss":
                    float(
                        best_test_loss
                    ),
                "atom_scaler":
                    scaler_to_dict(
                        atom_scaler
                    ),
                "global_scaler":
                    scaler_to_dict(
                        global_scaler
                    ),
                "target_scaler":
                    scaler_to_dict(
                        target_scaler
                    ),
                "descriptor_state":
                    descriptor_state,
                "cache_file":
                    cache_file,
                "energy_file":
                    energy_file,
                "vicuna_frozen":
                    True,
                "projector_type":
                    "self_attention",
                "projector_attention_dim":
                    256,
                "projector_num_heads":
                    8,
                "projector_num_layers":
                    2,
            }

            torch.save(
                checkpoint,
                best_model_path,
            )

            print(
                "  --> 保存最优轻量 checkpoint: "
                f"{best_model_path}"
            )

        else:
            epochs_without_improvement += 1

        pd.DataFrame(
            history
        ).to_csv(
            os.path.join(
                log_dir,
                "training_metrics_history.csv",
            ),
            index=False,
        )

        # Plot every 10 epochs to avoid unnecessary I/O.
        if (
            epoch == 1
            or epoch % 10 == 0
        ):
            plot_training_history(
                history,
                os.path.join(
                    plot_dir,
                    "training_curves.png",
                ),
            )

        if (
            epochs_without_improvement
            >= args.patience
        ):
            print(
                f"Early stopping: "
                f"连续 "
                f"{args.patience} "
                f"个 epoch 无改善，"
                f"停止于 epoch="
                f"{epoch}"
            )

            break

    # Final history plot
    plot_training_history(
        history,
        os.path.join(
            plot_dir,
            "training_curves.png",
        ),
    )

    # --------------------------------------------------------
    # 11. Final evaluation using best checkpoint
    # --------------------------------------------------------
    print(
        "\n加载最优 checkpoint "
        "进行最终评估..."
    )

    best_checkpoint = torch.load(
        best_model_path,
        map_location="cpu",
        weights_only=False,
    )

    model.load_trainable_state_dict(
        best_checkpoint[
            "trainable_state_dict"
        ]
    )

    model.eval()

    (
        train_true,
        train_pred,
        train_cif_idx,
    ) = predict_loader(
        model,
        eval_train_loader,
        device,
        target_scaler,
    )

    (
        test_true,
        test_pred,
        test_cif_idx,
    ) = predict_loader(
        model,
        test_loader,
        device,
        target_scaler,
    )

    train_metrics = (
        calc_multidim_metrics(
            train_true,
            train_pred,
            ENERGY_NAMES,
        )
    )

    test_metrics = (
        calc_multidim_metrics(
            test_true,
            test_pred,
            ENERGY_NAMES,
        )
    )

    print(
        "\n" + "=" * 108
    )

    print(
        "最终 four-state Energy 结果"
    )

    print(
        "=" * 108
    )

    print(
        f"Train Macro: "
        f"MAE="
        f"{train_metrics['mae']:.8e} | "
        f"RMSE="
        f"{train_metrics['rmse']:.8e} | "
        f"R2="
        f"{train_metrics['r2']:.8f}"
    )

    print(
        f"Test  Macro: "
        f"MAE="
        f"{test_metrics['mae']:.8e} | "
        f"RMSE="
        f"{test_metrics['rmse']:.8e} | "
        f"R2="
        f"{test_metrics['r2']:.8f}"
    )

    print("\nTest per state:")

    for m in (
        test_metrics["per_dim"]
    ):
        print(
            f"  {m['name']}: "
            f"MAE="
            f"{m['mae']:.8e} | "
            f"RMSE="
            f"{m['rmse']:.8e} | "
            f"R2="
            f"{m['r2']:.8f}"
        )

    print(
        "=" * 108
    )

    # --------------------------------------------------------
    # 12. Plots
    # --------------------------------------------------------
    plot_energy_comparison(
        train_true,
        train_pred,
        "Train",
        os.path.join(
            plot_dir,
            "train_four_state_energy_prediction.png",
        ),
    )

    plot_energy_comparison(
        test_true,
        test_pred,
        "Test",
        os.path.join(
            plot_dir,
            "test_four_state_energy_prediction.png",
        ),
    )

    plot_full_time_series(
        targets_np,
        train_positions,
        train_pred,
        test_positions,
        test_pred,
        target_cif_indices,
        os.path.join(
            plot_dir,
            "full_md_four_state_energy_evolution.png",
        ),
    )

    # --------------------------------------------------------
    # 13. CSV / per-state metrics
    # --------------------------------------------------------
    save_predictions(
        train_true,
        train_pred,
        train_cif_idx,
        os.path.join(
            result_dir,
            "train_four_state_energy_predictions.csv",
        ),
    )

    save_predictions(
        test_true,
        test_pred,
        test_cif_idx,
        os.path.join(
            result_dir,
            "test_four_state_energy_predictions.csv",
        ),
    )

    save_full_time_series_csv(
        targets_np,
        train_positions,
        train_pred,
        test_positions,
        test_pred,
        target_cif_indices,
        os.path.join(
            result_dir,
            "full_md_four_state_energy_evolution.csv",
        ),
    )

    save_metrics_per_state(
        train_true,
        train_pred,
        "Train",
        os.path.join(
            result_dir,
            "train_metrics_per_state.csv",
        ),
    )

    save_metrics_per_state(
        test_true,
        test_pred,
        "Test",
        os.path.join(
            result_dir,
            "test_metrics_per_state.csv",
        ),
    )

    # --------------------------------------------------------
    # 14. Summary
    # --------------------------------------------------------
    summary = {
        "architecture": (
            "existing atom-wise invariant cache -> "
            "masked atom mean + global -> "
            "Self-Attention + Cross-Attention + FC projector -> "
            "frozen Vicuna-7B -> "
            "four-state Energy regression head"
        ),
        "model_path":
            args.model_path,
        "cache_file":
            cache_file,
        "energy_file":
            energy_file,
        "energy_names":
            ENERGY_NAMES,
        "out_dim":
            EXPECTED_OUT_DIM,
        "seq_len":
            int(
                args.seq_len
            ),
        "train_ratio":
            float(
                args.train_ratio
            ),
        "batch_size":
            int(
                args.batch_size
            ),
        "grad_accum":
            int(
                args.grad_accum
            ),
        "samples":
            int(
                num_samples
            ),
        "train_samples":
            int(
                len(
                    train_positions
                )
            ),
        "test_samples":
            int(
                len(
                    test_positions
                )
            ),
        "atom_feature_dim":
            int(
                frame_atom_scaled
                .shape[-1]
            ),
        "global_feature_dim":
            int(
                frame_global_scaled
                .shape[-1]
            ),
        "vicuna_hidden_size":
            int(
                model.hidden_size
            ),
        "vicuna_frozen":
            True,
        "projector_type":
            "self_attention",
        "projector_attention_dim":
            256,
        "projector_num_heads":
            8,
        "projector_num_layers":
            2,
        "best_epoch":
            int(
                best_checkpoint[
                    "best_epoch"
                ]
            ),
        "best_test_loss":
            float(
                best_checkpoint[
                    "best_test_loss"
                ]
            ),
        "train_macro":
            train_metrics,
        "test_macro":
            test_metrics,
        "structure_data_note": (
            "No CIF reading and no descriptor recomputation. "
            "The existing frame cache is loaded directly. "
            "16-frame windows are sliced on demand."
        ),
    }

    with open(
        os.path.join(
            result_dir,
            "run_summary.json",
        ),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n全部完成。重点查看:")

    print(
        f"  {plot_dir}/"
        "test_four_state_energy_prediction.png"
    )

    print(
        f"  {plot_dir}/"
        "full_md_four_state_energy_evolution.png"
    )

    print(
        f"  {result_dir}/"
        "test_four_state_energy_predictions.csv"
    )

    print(
        f"  {result_dir}/"
        "test_metrics_per_state.csv"
    )

    print(
        f"  best checkpoint: "
        f"{best_model_path}"
    )


if __name__ == "__main__":
    main()
