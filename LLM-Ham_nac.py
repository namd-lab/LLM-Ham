import os
import re
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
# 0. Paths / task settings
# ============================================================

DEFAULT_RUN_DIR = "/root/vicuna/311_VS_nac"
DEFAULT_CACHE_FILE = os.path.join(
    DEFAULT_RUN_DIR,
    "atomwise_invariant_v3_Z16-42.pt",
)
DEFAULT_NAC_DIR = os.path.join(DEFAULT_RUN_DIR, "nac")
DEFAULT_ENERGY_FILE = os.path.join(DEFAULT_RUN_DIR, "energy", "energy")

DEFAULT_MODEL_PATH = "/root/vicuna/models/vicuna-7b-v1.5"
DEFAULT_OUTPUT_ROOT = os.path.join(
    DEFAULT_RUN_DIR,
    "outputs_vicuna_three_nac",
)

SEED = 42
SEQ_LEN = 16
TRAIN_RATIO = 0.15

# User request: real physical batch = 32
BATCH_SIZE = 32
GRAD_ACCUM = 1

EPOCHS = 1000
LR = 2.0e-4
WEIGHT_DECAY = 1.0e-4
PATIENCE = 200
MIN_DELTA = 1.0e-7

NAC_REAL_KEYWORD = "real"
NAC_UNIT_LABEL = "native"

# Three adjacent state pairs, 1-based
TARGET_STATE_PAIRS_1BASED = [
    (1, 2),
    (2, 3),
    (3, 4),
]

# Strictly use the upper-triangle real entries as in the original code:
# raw[0,2], raw[1,4], raw[2,6], then abs().
USE_ANTISYMMETRIC_AVERAGE = False
TARGET_ABS_REAL = True

GAP_EPS = 1.0e-8
GAP_FLOOR = 1.0e-6

LOSS_W_NAC = 1.00
LOSS_W_H = 0.80
LOSS_W_LOG = 0.35
SMOOTH_L1_BETA = 0.5
H_SOFTPLUS_BETA = 5.0


def pairs_zero_based():
    pairs = []
    for a, b in TARGET_STATE_PAIRS_1BASED:
        if a < 1 or b < 1 or a == b:
            raise ValueError(f"非法 state pair: {(a, b)}")
        i, j = a - 1, b - 1
        if i > j:
            i, j = j, i
        pairs.append((i, j))
    if len(set(pairs)) != len(pairs):
        raise ValueError("TARGET_STATE_PAIRS_1BASED 中存在重复 pair")
    return pairs


TARGET_PAIRS = pairs_zero_based()
OUT_DIM = len(TARGET_PAIRS)
PAIR_NAMES = [
    f"AbsReNAC_{i+1}_{j+1}"
    for i, j in TARGET_PAIRS
]


# ============================================================
# 1. Utilities / metrics
# ============================================================

def set_seed(seed: int = 42):
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
    if np.allclose(np.max(y_true), np.min(y_true)):
        return np.nan
    return float(r2_score(y_true, y_pred))


def calc_multidim_metrics(y_true, y_pred, names=None):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"metric shape mismatch: {y_true.shape} != {y_pred.shape}"
        )

    if names is None:
        names = [f"Dim{i+1}" for i in range(y_true.shape[1])]

    per_dim = []
    for d in range(y_true.shape[1]):
        yt = y_true[:, d]
        yp = y_pred[:, d]
        mae = float(np.mean(np.abs(yp - yt)))
        rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
        r2 = safe_r2(yt, yp)
        per_dim.append({
            "dim": d + 1,
            "name": names[d],
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        })

    macro_mae = float(np.mean([m["mae"] for m in per_dim]))
    macro_rmse = float(np.mean([m["rmse"] for m in per_dim]))
    valid_r2 = [m["r2"] for m in per_dim if np.isfinite(m["r2"])]
    macro_r2 = float(np.mean(valid_r2)) if valid_r2 else np.nan

    return {
        "mae": macro_mae,
        "rmse": macro_rmse,
        "r2": macro_r2,
        "per_dim": per_dim,
    }


# ============================================================
# 2. Directly load the EXISTING structural cache
# ============================================================

def load_feature_cache(cache_file: str):
    if not os.path.isfile(cache_file):
        raise FileNotFoundError(
            f"已有结构 cache 不存在: {cache_file}"
        )

    print(f"直接加载已有结构 cache: {cache_file}")
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
    missing = [k for k in required if k not in cache]
    if missing:
        raise KeyError(
            f"cache 缺少字段: {missing}; "
            f"keys={list(cache.keys())}"
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
    descriptor_state = cache.get("descriptor_state", {})

    if atom_features.ndim != 3:
        raise ValueError(
            f"atom_features 应为 [frames,atoms,F]，"
            f"实际 {atom_features.shape}"
        )
    if atom_masks.shape != atom_features.shape[:2]:
        raise ValueError(
            f"atom_masks shape 不匹配: "
            f"{atom_masks.shape} vs {atom_features.shape[:2]}"
        )
    if global_features.ndim != 2:
        raise ValueError(
            f"global_features 应为 [frames,G]，"
            f"实际 {global_features.shape}"
        )
    if len(global_features) != len(atom_features):
        raise ValueError(
            "global_features 与 atom_features frame 数不一致"
        )
    if len(frame_indices) != len(atom_features):
        raise ValueError(
            "frame_indices 与 atom_features frame 数不一致"
        )

    print("cache 校验通过")
    print(f"  atom_features   : {atom_features.shape}")
    print(f"  atom_masks      : {atom_masks.shape}")
    print(f"  global_features : {global_features.shape}")
    print(
        f"  frame_indices   : {frame_indices.shape}, "
        f"range={frame_indices.min()}~{frame_indices.max()}"
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
# 3. NAC file alignment / three targets
# ============================================================

def extract_all_integers(filename: str) -> List[int]:
    basename = os.path.basename(filename)
    return [
        int(x)
        for x in re.findall(r"\d+", basename)
    ]


def build_nac_file_index(
    nac_dir: str,
    keyword: str = "real",
) -> Dict[int, str]:

    if not os.path.isdir(nac_dir):
        raise FileNotFoundError(
            f"NAC 目录不存在: {nac_dir}"
        )

    keyword_lower = keyword.lower()
    candidates: Dict[int, List[Tuple[Tuple, str]]] = {}

    for path in glob.glob(
        os.path.join(nac_dir, "**", "*"),
        recursive=True,
    ):
        if not os.path.isfile(path):
            continue

        basename = os.path.basename(path)
        if keyword_lower not in basename.lower():
            continue

        nums = extract_all_integers(basename)
        if not nums:
            continue

        priority = (
            0 if len(nums) == 1 else 1,
            len(basename),
            basename,
        )

        for idx in set(nums):
            candidates.setdefault(
                int(idx),
                [],
            ).append((priority, path))

    index = {}
    for idx, items in candidates.items():
        items.sort(key=lambda x: x[0])
        index[idx] = items[0][1]

    print(
        f"NAC 文件索引建立完成: {len(index)} 个 index"
    )

    if not index:
        raise ValueError(
            f"没有在 {nac_dir} 中找到包含 "
            f"keyword='{keyword}' 的 NAC 文件"
        )

    return index


def read_nac_real_raw(filepath: str) -> np.ndarray:
    raw = np.loadtxt(
        filepath,
        dtype=np.float64,
    )

    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    if raw.ndim != 2:
        raise ValueError(
            f"NAC real 文件必须二维，实际 ndim={raw.ndim}"
        )

    if raw.shape[1] % 2 != 0:
        raise ValueError(
            f"NAC real 列数必须为偶数(Re/Im成对)，"
            f"实际 {raw.shape}: {filepath}"
        )

    return raw


def extract_three_nac_targets(
    filepath: str,
    target_pairs=None,
):
    """
    对 pair (i,j), 0-based:
        forward = raw[i, 2*j] = Re d_ij
        reverse = raw[j, 2*i] = Re d_ji

    默认严格使用 forward，然后取 abs：
        (1,2): abs(raw[0,2])
        (2,3): abs(raw[1,4])
        (3,4): abs(raw[2,6])
    """
    if target_pairs is None:
        target_pairs = TARGET_PAIRS

    raw = read_nac_real_raw(filepath)

    n_states_from_cols = raw.shape[1] // 2
    n_states = min(
        raw.shape[0],
        n_states_from_cols,
    )

    values = []
    metadata = []

    for i, j in target_pairs:
        if i >= n_states or j >= n_states:
            raise IndexError(
                f"pair ({i+1},{j+1}) 超出 NAC 状态数 "
                f"{n_states}; raw shape={raw.shape}"
            )

        forward = float(raw[i, 2 * j])
        reverse = float(raw[j, 2 * i])

        if not np.isfinite(forward):
            raise ValueError(
                f"forward NAC 非有限值: {forward}"
            )
        if not np.isfinite(reverse):
            raise ValueError(
                f"reverse NAC 非有限值: {reverse}"
            )

        if USE_ANTISYMMETRIC_AVERAGE:
            signed_real = 0.5 * (
                forward - reverse
            )
        else:
            signed_real = forward

        target = (
            abs(signed_real)
            if TARGET_ABS_REAL
            else signed_real
        )

        values.append(target)
        metadata.append({
            "pair": (i + 1, j + 1),
            "forward": forward,
            "reverse": reverse,
            "antisym_residual": (
                forward + reverse
            ),
        })

    return np.asarray(
        values,
        dtype=np.float64,
    ), {
        "raw_shape": tuple(raw.shape),
        "n_states": int(n_states),
        "pairs": metadata,
    }


# ============================================================
# 4. Energy file -> all states -> three pair gaps
# ============================================================

def _read_numeric_energy_rows(filepath: str):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"energy 文件不存在: {filepath}"
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

            if len(values) < 1:
                continue

            rows.append(
                (line_no, values)
            )

    if not rows:
        raise ValueError(
            f"没有从 energy 文件读取到数字数据: "
            f"{filepath}"
        )

    return rows


def read_energy_all_states(
    filepath: str,
    sorted_frame_indices: List[int],
):
    """
    Supports:
      A) E1 E2 E3 E4 ...
      B) index E1 E2 E3 E4 ...
    """
    rows = _read_numeric_energy_rows(filepath)

    sorted_frame_indices = [
        int(x)
        for x in sorted_frame_indices
    ]
    frame_set = set(sorted_frame_indices)

    explicit_candidates = []

    for _, values in rows:
        if (
            len(values) >= 2
            and abs(
                values[0] - round(values[0])
            ) < 1.0e-10
        ):
            explicit_candidates.append(
                int(round(values[0]))
            )
        else:
            explicit_candidates.append(None)

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
            int(0.8 * len(rows)),
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

    result = {}
    n_energy_states = None

    if use_explicit_index:
        print(
            "Energy 格式识别: "
            "显式 index E1 E2 E3 ..."
        )

        for (
            line_no,
            values,
        ), idx_candidate in zip(
            rows,
            explicit_candidates,
        ):
            if (
                idx_candidate is None
                or len(values) < 2
            ):
                continue

            idx = int(idx_candidate)
            energies = np.asarray(
                values[1:],
                dtype=np.float64,
            )

            if not np.all(
                np.isfinite(energies)
            ):
                raise ValueError(
                    f"energy 第 {line_no} 行 "
                    f"包含非有限值"
                )

            if n_energy_states is None:
                n_energy_states = len(energies)

            if (
                len(energies)
                != n_energy_states
            ):
                raise ValueError(
                    f"energy 列数不一致: "
                    f"第 {line_no} 行={len(energies)}, "
                    f"预期={n_energy_states}"
                )

            result[idx] = energies

    else:
        print(
            "Energy 格式识别: "
            "无显式 index，按 frame index 顺序对齐"
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
                f"⚠️ energy 行数={len(rows)}, "
                f"frame 数={len(sorted_frame_indices)}; "
                f"仅对齐前 {n} 个"
            )

        for pos in range(n):
            line_no, values = rows[pos]
            idx = int(
                sorted_frame_indices[pos]
            )

            energies = np.asarray(
                values,
                dtype=np.float64,
            )

            if not np.all(
                np.isfinite(energies)
            ):
                raise ValueError(
                    f"energy 第 {line_no} 行 "
                    f"包含非有限值"
                )

            if n_energy_states is None:
                n_energy_states = len(energies)

            if (
                len(energies)
                != n_energy_states
            ):
                raise ValueError(
                    f"energy 列数不一致: "
                    f"第 {line_no} 行={len(energies)}, "
                    f"预期={n_energy_states}"
                )

            result[idx] = energies

    if (
        not result
        or n_energy_states is None
    ):
        raise ValueError(
            "没有成功建立 energy 与 frame 的对应关系"
        )

    max_state_needed = (
        max(
            max(i, j)
            for i, j in TARGET_PAIRS
        )
        + 1
    )

    if n_energy_states < max_state_needed:
        raise ValueError(
            f"三个目标 pair 至少需要 "
            f"{max_state_needed} 个 energy states，"
            f"当前只有 {n_energy_states}"
        )

    mat = np.stack(
        list(result.values()),
        axis=0,
    )

    print(
        f"Energy 对齐成功: {len(result)} 帧 | "
        f"states={n_energy_states} | "
        f"range=[{mat.min():.8e}, "
        f"{mat.max():.8e}]"
    )

    return result, int(n_energy_states)


def energy_pair_gaps(
    energies,
    target_pairs=None,
):
    if target_pairs is None:
        target_pairs = TARGET_PAIRS

    energies = np.asarray(
        energies,
        dtype=np.float64,
    ).reshape(-1)

    gaps = []

    for i, j in target_pairs:
        if j >= len(energies):
            raise IndexError(
                f"energy 只有 {len(energies)} states，"
                f"不能计算 pair ({i+1},{j+1})"
            )

        gaps.append(
            abs(
                float(
                    energies[j]
                    - energies[i]
                )
            )
        )

    return np.asarray(
        gaps,
        dtype=np.float64,
    )


# ============================================================
# 5. Align existing cache + energy + NAC
#    NO structural dataset reconstruction
# ============================================================

def align_existing_samples(
    frame_indices,
    energy_dict,
    nac_file_index,
    seq_len=16,
    result_dir=None,
):
    """
    Only build lightweight sample metadata:
      start frame position
      target frame index
      three raw gaps
      three raw NAC targets
      all energy states at target frame

    Structural windows are NOT copied here.
    Dataset.__getitem__ later slices directly from the existing
    frame-level cache.
    """
    starts = []
    target_indices = []
    gaps_list = []
    targets_list = []
    energies_list = []
    source_rows = []

    skipped_noncontinuous = 0
    skipped_missing_energy = 0
    skipped_missing_nac = 0
    skipped_parse = 0

    n = len(frame_indices)

    print(
        "\n开始对齐已有 cache / energy / NAC 标签..."
    )

    for end in range(
        seq_len - 1,
        n,
    ):
        start = end - seq_len + 1

        window_indices = frame_indices[
            start:end + 1
        ]

        if not np.all(
            np.diff(window_indices) == 1
        ):
            skipped_noncontinuous += 1
            continue

        target_idx = int(
            frame_indices[end]
        )

        if target_idx not in energy_dict:
            skipped_missing_energy += 1
            continue

        nac_path = nac_file_index.get(
            target_idx
        )

        if nac_path is None:
            skipped_missing_nac += 1
            continue

        try:
            targets, meta = (
                extract_three_nac_targets(
                    nac_path,
                    target_pairs=TARGET_PAIRS,
                )
            )

            energies = np.asarray(
                energy_dict[target_idx],
                dtype=np.float64,
            )

            gaps = energy_pair_gaps(
                energies,
                target_pairs=TARGET_PAIRS,
            )

            if targets.shape != (OUT_DIM,):
                raise ValueError(
                    f"target shape={targets.shape}, "
                    f"预期={(OUT_DIM,)}"
                )

            if gaps.shape != (OUT_DIM,):
                raise ValueError(
                    f"gap shape={gaps.shape}, "
                    f"预期={(OUT_DIM,)}"
                )

            if not np.all(
                np.isfinite(targets)
            ):
                raise ValueError(
                    f"NAC targets 非有限: {targets}"
                )

            if not np.all(
                np.isfinite(gaps)
            ):
                raise ValueError(
                    f"gaps 非有限: {gaps}"
                )

            starts.append(start)
            target_indices.append(
                target_idx
            )
            gaps_list.append(gaps)
            targets_list.append(targets)
            energies_list.append(energies)

            row = {
                "CIF_Index": target_idx,
                "NAC_File": nac_path,
                "Window_Start_Frame_Position": start,
                "Window_End_Frame_Position": end,
                "Raw_Rows": meta["raw_shape"][0],
                "Raw_Cols": meta["raw_shape"][1],
                "N_States": meta["n_states"],
            }

            for d, name in enumerate(
                PAIR_NAMES
            ):
                i, j = TARGET_PAIRS[d]
                pair_meta = meta["pairs"][d]

                row[
                    f"{name}_Forward"
                ] = pair_meta["forward"]

                row[
                    f"{name}_Reverse"
                ] = pair_meta["reverse"]

                row[
                    f"{name}_AntisymResidual"
                ] = pair_meta[
                    "antisym_residual"
                ]

                row[
                    f"{name}_Target"
                ] = targets[d]

                row[
                    f"Gap_{i+1}_{j+1}"
                ] = gaps[d]

            source_rows.append(row)

        except Exception as e:
            skipped_parse += 1

            if skipped_parse <= 10:
                print(
                    f"⚠️ CIF index={target_idx} "
                    f"对齐失败: {e}"
                )

    starts = np.asarray(
        starts,
        dtype=np.int64,
    )
    target_indices = np.asarray(
        target_indices,
        dtype=np.int64,
    )
    gaps_arr = np.asarray(
        gaps_list,
        dtype=np.float64,
    )
    targets_arr = np.asarray(
        targets_list,
        dtype=np.float64,
    )

    if len(starts) < 2:
        raise ValueError(
            "有效监督样本少于 2，请检查 "
            "cache / energy / NAC 对齐关系"
        )

    # Energy state count is fixed by parser
    energies_arr = np.asarray(
        energies_list,
        dtype=np.float64,
    )

    print(
        "\n" + "=" * 96
    )
    print(
        "已有数据对齐完成 "
        "(未重新计算结构特征，未复制16帧结构窗口)"
    )
    print(
        "=" * 96
    )
    print(
        f"有效样本数           : {len(starts)}"
    )
    print(
        f"轻量 start index     : {starts.shape}"
    )
    print(
        f"Gap shape            : {gaps_arr.shape}"
    )
    print(
        f"Target shape         : {targets_arr.shape}"
    )
    print(
        f"Energy shape         : {energies_arr.shape}"
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
        f"缺少 NAC             : "
        f"{skipped_missing_nac}"
    )
    print(
        f"解析失败             : "
        f"{skipped_parse}"
    )
    print(
        f"target CIF index     : "
        f"{target_indices.min()} ~ "
        f"{target_indices.max()}"
    )

    for d, name in enumerate(PAIR_NAMES):
        print(
            f"{name}: NAC min/max="
            f"{targets_arr[:, d].min():.8e} / "
            f"{targets_arr[:, d].max():.8e} | "
            f"gap min/max="
            f"{gaps_arr[:, d].min():.8e} / "
            f"{gaps_arr[:, d].max():.8e}"
        )

    print("=" * 96)

    if result_dir is not None:
        os.makedirs(
            result_dir,
            exist_ok=True,
        )
        pd.DataFrame(
            source_rows
        ).to_csv(
            os.path.join(
                result_dir,
                "nac_target_source_mapping.csv",
            ),
            index=False,
            float_format="%.12e",
        )

    return (
        starts,
        gaps_arr,
        targets_arr,
        target_indices,
        energies_arr,
    )


# ============================================================
# 6. Same uniform split logic
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
        0.0 < train_ratio < 1.0
    ):
        raise ValueError(
            "train_ratio 必须在 (0,1)"
        )

    train_num = int(
        round(
            total_num * train_ratio
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
        np.arange(total_num),
        train_idx,
    )

    if len(test_idx) == 0:
        raise ValueError(
            "测试集为空"
        )

    print(
        f"采用原代码全局均匀间隔采样 -> "
        f"Train={len(train_idx)} "
        f"({len(train_idx)/total_num:.2%}), "
        f"Test={len(test_idx)} "
        f"({len(test_idx)/total_num:.2%})"
    )

    print(
        "前 20 个训练样本位置:",
        train_idx[:20].tolist(),
    )

    return train_idx, test_idx


# ============================================================
# 7. Train-only scaling, without building [N,S,A,F]
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
    Fit StandardScaler using exactly the training windows.

    Repeated frames appearing in multiple overlapping training windows
    are counted repeatedly, matching the statistics that would be obtained
    by first materializing atom_X[train_positions] and global_X[train_positions].
    """
    atom_scaler = StandardScaler()
    global_scaler = StandardScaler()

    for sample_pos in train_positions:
        start = int(
            sample_starts[sample_pos]
        )
        end = start + seq_len

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
            atom_window[mask_window]
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
    ).astype(np.float32)

    return out


def transform_frame_global_features(
    frame_global_features,
    scaler,
):
    return scaler.transform(
        frame_global_features
    ).astype(np.float32)


class PairGapFeatureScaler:
    """
    gaps [N,P] -> [gap_1..gap_P, 1/gap_1..1/gap_P]
    then train-only standardization.
    """

    def fit(self, gaps):
        gaps = np.asarray(
            gaps,
            dtype=np.float64,
        )

        if gaps.ndim == 1:
            gaps = gaps.reshape(-1, 1)

        safe_gap = np.maximum(
            gaps,
            GAP_FLOOR,
        )

        x = np.concatenate(
            [
                safe_gap,
                1.0 / safe_gap,
            ],
            axis=1,
        )

        self.mean_ = x.mean(
            axis=0,
            keepdims=True,
        )
        self.std_ = x.std(
            axis=0,
            keepdims=True,
        )

        self.std_[
            self.std_ < 1.0e-12
        ] = 1.0

        self.n_pairs_ = gaps.shape[1]

        return self

    def transform(self, gaps):
        gaps = np.asarray(
            gaps,
            dtype=np.float64,
        )

        if gaps.ndim == 1:
            gaps = gaps.reshape(-1, 1)

        safe_gap = np.maximum(
            gaps,
            GAP_FLOOR,
        )

        x = np.concatenate(
            [
                safe_gap,
                1.0 / safe_gap,
            ],
            axis=1,
        )

        return (
            (x - self.mean_)
            / self.std_
        ).astype(np.float32)


def scaler_to_dict(scaler):
    return {
        "mean_": scaler.mean_.tolist(),
        "scale_": scaler.scale_.tolist(),
        "var_": scaler.var_.tolist(),
        "n_features_in_": int(
            scaler.n_features_in_
        ),
    }


# ============================================================
# 8. Index-based Dataset: direct slices from existing frame cache
# ============================================================

class CachedWindowThreeNACDataset(Dataset):
    """
    No duplicated [N,S,A,F] storage.

    frame_atom_X / frame_global_X are stored once.
    Each item dynamically slices the 16-frame window from the cache.
    """

    def __init__(
        self,
        frame_atom_X_tensor,
        frame_atom_mask_tensor,
        frame_global_X_tensor,
        sample_starts,
        gap_features,
        gaps_raw,
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
        self.gap_features = torch.tensor(
            gap_features,
            dtype=torch.float32,
        )
        self.gaps_raw = torch.tensor(
            gaps_raw,
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
        self.seq_len = int(seq_len)

    def __len__(self):
        return len(
            self.subset_positions
        )

    def __getitem__(self, idx):
        sample_pos = int(
            self.subset_positions[idx]
        )
        start = int(
            self.sample_starts[
                sample_pos
            ]
        )
        end = start + self.seq_len

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
            self.gap_features[
                sample_pos
            ],
            self.gaps_raw[
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
# 9. Frozen Vicuna multi-NAC physics model
# ============================================================

class SelfCrossAttentionFCProjector(nn.Module):
    """
    Trainable structure projector:

        raw structural sequence [B,S,input_dim]
            -> LayerNorm
            -> Linear(input_dim -> attention_dim)
            -> temporal position embedding
            -> SELF-ATTENTION across the S structural frames
            -> structural memory [B,S,attention_dim]

        trainable query tokens [B,S,attention_dim]
            -> CROSS-ATTENTION
               Q = trainable query tokens
               K = structural memory
               V = structural memory
            -> residual + LayerNorm

            -> FULLY CONNECTED / FFN
               attention_dim -> 4*attention_dim -> attention_dim
            -> residual + LayerNorm

            -> Linear(attention_dim -> Vicuna hidden size)
            -> structural pseudo-tokens [B,S,Vicuna_hidden_size]

    The number of trainable query tokens is kept equal to seq_len so that
    the downstream frozen-Vicuna input still receives the same number of
    structure tokens as before.
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

        # =====================================================
        # A. Input projection
        # =====================================================
        self.input_norm = nn.LayerNorm(
            self.input_dim
        )

        self.input_proj = nn.Linear(
            self.input_dim,
            self.attention_dim,
        )

        # Preserve temporal ordering information.
        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                self.seq_len,
                self.attention_dim,
            )
        )

        # =====================================================
        # B. SELF-ATTENTION
        # =====================================================
        self_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.attention_dim,
            nhead=self.num_heads,
            dim_feedforward=self.attention_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.self_attention = nn.TransformerEncoder(
            self_encoder_layer,
            num_layers=self.num_self_layers,
        )

        self.memory_norm = nn.LayerNorm(
            self.attention_dim
        )

        # =====================================================
        # C. CROSS-ATTENTION
        # =====================================================
        # Trainable queries retrieve information from the
        # self-attended structural memory.
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

        # =====================================================
        # D. FULLY CONNECTED / FEED-FORWARD NETWORK
        # =====================================================
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

        # =====================================================
        # E. Project to Vicuna hidden size
        # =====================================================
        self.output_norm = nn.LayerNorm(
            self.attention_dim
        )

        self.output_proj = nn.Linear(
            self.attention_dim,
            self.output_dim,
        )

        # =====================================================
        # Initialization
        # =====================================================
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

        # -----------------------------------------------------
        # 1. Input projection
        # -----------------------------------------------------
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

        # -----------------------------------------------------
        # 2. SELF-ATTENTION across the 16 structural frames
        # -----------------------------------------------------
        memory = self.self_attention(
            h
        )

        memory = self.memory_norm(
            memory
        )

        # -----------------------------------------------------
        # 3. CROSS-ATTENTION
        #    Q = trainable queries
        #    K,V = self-attended structural memory
        # -----------------------------------------------------
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

        # -----------------------------------------------------
        # 4. FULLY CONNECTED / FFN
        # -----------------------------------------------------
        fc_out = self.fully_connected(
            h
        )

        h = self.fc_norm(
            h
            + self.fc_dropout(
                fc_out
            )
        )

        # -----------------------------------------------------
        # 5. Output projection to Vicuna hidden size
        # -----------------------------------------------------
        h = self.output_proj(
            self.output_norm(
                h
            )
        )

        return h


class FrozenVicunaThreeNAC(nn.Module):
    def __init__(
        self,
        model_path: str,
        atom_dim: int,
        global_dim: int,
        out_dim: int,
        gap_input_dim: int,
        prompt: str,
        h_scale,
        h_softplus_beta: float = 5.0,
        gap_hidden: int = 128,
        seq_len: int = SEQ_LEN,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self.model_path = model_path
        self.atom_dim = int(atom_dim)
        self.global_dim = int(global_dim)
        self.out_dim = int(out_dim)
        self.gap_input_dim = int(
            gap_input_dim
        )
        self.prompt = prompt
        self.seq_len = int(
            seq_len
        )
        self.h_softplus_beta = float(
            h_softplus_beta
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_path,
                use_fast=False,
            )
        )

        if (
            self.tokenizer.pad_token_id
            is None
        ):
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )

        if (
            torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        ):
            self.llm_dtype = torch.bfloat16
        else:
            self.llm_dtype = torch.float16

        print(
            f"加载 Vicuna: {model_path}"
        )
        print(
            f"LLM dtype: {self.llm_dtype}"
        )

        self.llm = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=self.llm_dtype,
                low_cpu_mem_usage=True,
            )
        )

        self.llm.config.use_cache = False

        # Completely freeze Vicuna
        for p in self.llm.parameters():
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
            self.llm.config.hidden_size
        )

        raw_frame_dim = (
            self.atom_dim
            + self.global_dim
        )

        # Trainable structure projector:
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

        # Three-pair gap branch:
        # [gap12,gap23,gap34,1/gap12,1/gap23,1/gap34]
        self.gap_encoder = nn.Sequential(
            nn.Linear(
                self.gap_input_dim,
                gap_hidden,
            ),
            nn.LayerNorm(gap_hidden),
            nn.GELU(),
        )

        fusion_dim = (
            self.hidden_size
            + gap_hidden
        )

        # Three positive h_theta values simultaneously
        self.h_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(
                fusion_dim,
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
                self.out_dim,
            ),
        )

        h_scale = np.asarray(
            h_scale,
            dtype=np.float32,
        ).reshape(1, -1)

        if h_scale.shape[1] != self.out_dim:
            raise ValueError(
                f"h_scale shape={h_scale.shape}, "
                f"out_dim={self.out_dim}"
            )

        self.register_buffer(
            "h_scale",
            torch.tensor(
                h_scale,
                dtype=torch.float32,
            ),
        )

        prompt_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        )["input_ids"]

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
            f"{self.seq_len} trainable-query outputs"
        )
        print(
            f"Gap input dim        : "
            f"{self.gap_input_dim}"
        )
        print(
            f"NAC outputs          : "
            f"{self.out_dim}"
        )
        print(
            "Physics              : "
            "NAC_ij = h_ij / gap_ij"
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # frozen backbone always eval
        self.llm.eval()
        return self

    def trainable_parameters(self):
        modules = [
            self.structure_projector,
            self.gap_encoder,
            self.h_head,
        ]

        for module in modules:
            for p in module.parameters():
                yield p

    def trainable_state_dict(self):
        return {
            "structure_projector":
                self.structure_projector.state_dict(),
            "gap_encoder":
                self.gap_encoder.state_dict(),
            "h_head":
                self.h_head.state_dict(),
        }

    def load_trainable_state_dict(
        self,
        state,
    ):
        self.structure_projector.load_state_dict(
            state["structure_projector"]
        )
        self.gap_encoder.load_state_dict(
            state["gap_encoder"]
        )
        self.h_head.load_state_dict(
            state["h_head"]
        )

    def forward(
        self,
        atom_x,
        atom_mask,
        global_x,
        gap_features,
        gap_raw,
    ):
        # atom_x      [B,S,A,F]
        # atom_mask   [B,S,A]
        # global_x    [B,S,G]
        # gap_features[B,2P]
        # gap_raw     [B,P]

        B, S, A, Fdim = atom_x.shape

        # Same direct raw-feature baseline as previous code:
        # no atom Transformer, masked atom mean only.
        mask_f = (
            atom_mask
            .unsqueeze(-1)
            .to(atom_x.dtype)
        )

        atom_sum = (
            atom_x * mask_f
        ).sum(dim=2)

        atom_count = (
            mask_f
            .sum(dim=2)
            .clamp_min(1.0)
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
            self.structure_projector(
                raw_frame
            )
        )

        struct_tokens = (
            struct_tokens_fp32.to(
                dtype=self.llm_dtype
            )
        )

        ids = self.prompt_ids.expand(
            B,
            -1,
        )

        text_emb = (
            self.llm
            .get_input_embeddings()(
                ids
            )
        )

        bos_id = (
            self.tokenizer.bos_token_id
        )

        has_bos = bool(
            ids.shape[1] > 0
            and bos_id is not None
            and torch.all(
                ids[:, 0] == bos_id
            )
        )

        if has_bos:
            input_embeds = torch.cat(
                [
                    text_emb[:, :1],
                    struct_tokens,
                    text_emb[:, 1:],
                ],
                dim=1,
            )
        else:
            input_embeds = torch.cat(
                [
                    struct_tokens,
                    text_emb,
                ],
                dim=1,
            )

        attention_mask = torch.ones(
            input_embeds.shape[:2],
            dtype=torch.long,
            device=input_embeds.device,
        )

        # IMPORTANT:
        # no torch.no_grad() here.
        # Gradient must pass THROUGH frozen Vicuna
        # to the Self-Attention + Cross-Attention + FC projector.
        outputs = self.llm.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )

        # Last prompt token can causally attend
        # to all preceding structure tokens.
        llm_h = (
            outputs
            .last_hidden_state[
                :, -1, :
            ]
            .float()
        )

        gap_emb = self.gap_encoder(
            gap_features.float()
        )

        fused = torch.cat(
            [
                llm_h,
                gap_emb,
            ],
            dim=-1,
        )

        raw_h = self.h_head(
            fused
        )

        # Positive numerator for each pair
        h_scaled = F.softplus(
            raw_h,
            beta=self.h_softplus_beta,
            threshold=20.0,
        )

        h_phys = (
            h_scaled
            * self.h_scale
        )

        safe_gap = torch.clamp(
            gap_raw.float(),
            min=GAP_FLOOR,
        )

        nac_phys = (
            h_phys
            / (
                safe_gap
                + GAP_EPS
            )
        )

        return (
            nac_phys,
            h_phys,
        )


# ============================================================
# 10. Multi-output balanced physics loss
# ============================================================

class MultiBalancedPhysicsLoss(nn.Module):
    def __init__(
        self,
        nac_scale,
        h_scale,
        log_scale,
        w_nac=1.0,
        w_h=0.8,
        w_log=0.35,
        beta=0.5,
    ):
        super().__init__()

        self.w_nac = float(w_nac)
        self.w_h = float(w_h)
        self.w_log = float(w_log)
        self.beta = float(beta)

        nac_scale = np.asarray(
            nac_scale,
            dtype=np.float32,
        ).reshape(1, -1)

        h_scale = np.asarray(
            h_scale,
            dtype=np.float32,
        ).reshape(1, -1)

        log_scale = np.asarray(
            log_scale,
            dtype=np.float32,
        ).reshape(1, -1)

        self.register_buffer(
            "nac_scale",
            torch.tensor(
                nac_scale,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "h_scale",
            torch.tensor(
                h_scale,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "log_scale",
            torch.tensor(
                log_scale,
                dtype=torch.float32,
            ),
        )

    def forward(
        self,
        pred_nac,
        pred_h,
        target_nac,
        gap_raw,
    ):
        safe_gap = torch.clamp(
            gap_raw.float(),
            min=GAP_FLOOR,
        )

        true_h = (
            target_nac.float()
            * safe_gap
        )

        pred_nac_s = (
            pred_nac
            / self.nac_scale
        )
        true_nac_s = (
            target_nac
            / self.nac_scale
        )

        pred_h_s = (
            pred_h
            / self.h_scale
        )
        true_h_s = (
            true_h
            / self.h_scale
        )

        loss_nac = (
            F.smooth_l1_loss(
                pred_nac_s,
                true_nac_s,
                beta=self.beta,
                reduction="mean",
            )
        )

        loss_h = (
            F.smooth_l1_loss(
                pred_h_s,
                true_h_s,
                beta=self.beta,
                reduction="mean",
            )
        )

        pred_log = torch.log1p(
            torch.clamp(
                pred_nac,
                min=0.0,
            )
            / self.log_scale
        )

        true_log = torch.log1p(
            torch.clamp(
                target_nac,
                min=0.0,
            )
            / self.log_scale
        )

        loss_log = (
            F.smooth_l1_loss(
                pred_log,
                true_log,
                beta=self.beta,
                reduction="mean",
            )
        )

        total = (
            self.w_nac * loss_nac
            + self.w_h * loss_h
            + self.w_log * loss_log
        )

        return (
            total,
            loss_nac.detach(),
            loss_h.detach(),
            loss_log.detach(),
        )


# ============================================================
# 11. Train / inference
# ============================================================

def process_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
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
    total_nac_loss = 0.0
    total_h_loss = 0.0
    total_log_loss = 0.0

    all_pred = []
    all_true = []
    all_h_pred = []
    all_h_true = []

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
                gap_features,
                gap_raw,
                target_raw,
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
            gap_features = gap_features.to(
                device,
                non_blocking=True,
            )
            gap_raw = gap_raw.to(
                device,
                non_blocking=True,
            )
            target_raw = target_raw.to(
                device,
                non_blocking=True,
            )

            pred_nac, pred_h = model(
                atom_x,
                atom_mask,
                global_x,
                gap_features,
                gap_raw,
            )

            (
                loss,
                l_nac,
                l_h,
                l_log,
            ) = criterion(
                pred_nac,
                pred_h,
                target_raw,
                gap_raw,
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

            bs = atom_x.size(0)

            total_loss += (
                float(
                    loss.detach().item()
                )
                * bs
            )
            total_nac_loss += (
                float(l_nac.item())
                * bs
            )
            total_h_loss += (
                float(l_h.item())
                * bs
            )
            total_log_loss += (
                float(l_log.item())
                * bs
            )

            all_pred.append(
                pred_nac
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            all_true.append(
                target_raw
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            all_h_pred.append(
                pred_h
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            all_h_true.append(
                (
                    target_raw
                    * torch.clamp(
                        gap_raw,
                        min=GAP_FLOOR,
                    )
                )
                .detach()
                .float()
                .cpu()
                .numpy()
            )

    all_pred = np.concatenate(
        all_pred,
        axis=0,
    )
    all_true = np.concatenate(
        all_true,
        axis=0,
    )
    all_h_pred = np.concatenate(
        all_h_pred,
        axis=0,
    )
    all_h_true = np.concatenate(
        all_h_true,
        axis=0,
    )

    n = len(loader.dataset)

    nac_metrics = (
        calc_multidim_metrics(
            all_true,
            all_pred,
            PAIR_NAMES,
        )
    )

    h_names = [
        f"h_{i+1}_{j+1}"
        for i, j in TARGET_PAIRS
    ]

    h_metrics = (
        calc_multidim_metrics(
            all_h_true,
            all_h_pred,
            h_names,
        )
    )

    return {
        "loss":
            total_loss / n,
        "loss_nac":
            total_nac_loss / n,
        "loss_h":
            total_h_loss / n,
        "loss_log":
            total_log_loss / n,
        "mae":
            nac_metrics["mae"],
        "rmse":
            nac_metrics["rmse"],
        "r2":
            nac_metrics["r2"],
        "per_dim":
            nac_metrics["per_dim"],
        "h_mae":
            h_metrics["mae"],
        "h_rmse":
            h_metrics["rmse"],
        "h_r2":
            h_metrics["r2"],
        "h_per_dim":
            h_metrics["per_dim"],
    }


@torch.no_grad()
def predict_loader(
    model,
    loader,
    device,
):
    model.eval()

    true_all = []
    pred_all = []
    h_all = []
    gap_all = []
    idx_all = []

    for (
        atom_x,
        atom_mask,
        global_x,
        gap_features,
        gap_raw,
        target_raw,
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
        gap_features = gap_features.to(
            device,
            non_blocking=True,
        )
        gap_device = gap_raw.to(
            device,
            non_blocking=True,
        )

        pred_nac, pred_h = model(
            atom_x,
            atom_mask,
            global_x,
            gap_features,
            gap_device,
        )

        true_all.append(
            target_raw.numpy()
        )
        pred_all.append(
            pred_nac
            .float()
            .cpu()
            .numpy()
        )
        h_all.append(
            pred_h
            .float()
            .cpu()
            .numpy()
        )
        gap_all.append(
            gap_raw.numpy()
        )
        idx_all.append(
            np.asarray(idx)
        )

    return (
        np.concatenate(
            true_all,
            axis=0,
        ),
        np.concatenate(
            pred_all,
            axis=0,
        ),
        np.concatenate(
            h_all,
            axis=0,
        ),
        np.concatenate(
            gap_all,
            axis=0,
        ),
        np.concatenate(
            idx_all,
            axis=0,
        ),
    )


# ============================================================
# 12. Plots
# ============================================================

def plot_pair_comparison(
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
        OUT_DIM,
        2,
        figsize=(
            15,
            5 * OUT_DIM,
        ),
        squeeze=False,
    )

    for d, name in enumerate(
        PAIR_NAMES
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

        order = np.argsort(yt)
        x = np.arange(
            1,
            len(yt) + 1,
        )

        axes[d, 0].plot(
            x,
            yt[order],
            label="True",
            linewidth=1.5,
        )
        axes[d, 0].plot(
            x,
            yp[order],
            label="Pred",
            linewidth=1.5,
            linestyle="--",
        )
        axes[d, 0].set_xlabel(
            "Samples sorted by true NAC"
        )
        axes[d, 0].set_ylabel(
            f"{name} ({NAC_UNIT_LABEL})"
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

        if np.isclose(lo, hi):
            hi = lo + 1.0e-12

        axes[d, 1].plot(
            [lo, hi],
            [lo, hi],
            "k--",
            linewidth=2,
            label="y=x",
        )
        axes[d, 1].set_xlabel(
            f"True {name}"
        )
        axes[d, 1].set_ylabel(
            f"Pred {name}"
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
            transform=(
                axes[d, 1].transAxes
            ),
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


def plot_h_comparison(
    true_nac,
    pred_h,
    gaps,
    title_prefix,
    filename,
):
    true_nac = np.asarray(
        true_nac,
        dtype=np.float64,
    )
    pred_h = np.asarray(
        pred_h,
        dtype=np.float64,
    )
    gaps = np.asarray(
        gaps,
        dtype=np.float64,
    )

    true_h = (
        true_nac
        * np.maximum(
            gaps,
            GAP_FLOOR,
        )
    )

    fig, axes = plt.subplots(
        OUT_DIM,
        1,
        figsize=(
            8,
            5 * OUT_DIM,
        ),
        squeeze=False,
    )
    axes = axes[:, 0]

    for d, name in enumerate(
        PAIR_NAMES
    ):
        yt = true_h[:, d]
        yp = pred_h[:, d]

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

        if np.isclose(lo, hi):
            hi = lo + 1.0e-12

        axes[d].scatter(
            yt,
            yp,
            alpha=0.65,
        )
        axes[d].plot(
            [lo, hi],
            [lo, hi],
            "k--",
            linewidth=2,
        )
        axes[d].set_xlabel(
            "True h = NAC * gap"
        )
        axes[d].set_ylabel(
            "Pred h"
        )
        axes[d].set_title(
            f"[{title_prefix}] "
            f"h for {name}"
        )
        axes[d].grid(
            True,
            linestyle=":",
            alpha=0.6,
        )
        axes[d].text(
            0.95,
            0.05,
            (
                f"R2={r2:.6f}\n"
                f"MAE={mae:.6e}\n"
                f"RMSE={rmse:.6e}"
            ),
            transform=(
                axes[d].transAxes
            ),
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


def plot_gap_relationship(
    gaps,
    targets,
    filename,
):
    gaps = np.asarray(
        gaps,
        dtype=np.float64,
    )
    targets = np.asarray(
        targets,
        dtype=np.float64,
    )

    fig, axes = plt.subplots(
        OUT_DIM,
        2,
        figsize=(
            14,
            4.5 * OUT_DIM,
        ),
        squeeze=False,
    )

    for d, name in enumerate(
        PAIR_NAMES
    ):
        gap = gaps[:, d]
        nac = targets[:, d]
        inv_gap = (
            1.0
            / np.maximum(
                gap,
                GAP_FLOOR,
            )
        )

        axes[d, 0].scatter(
            gap,
            nac,
            s=12,
            alpha=0.55,
        )
        axes[d, 0].set_xlabel(
            "Energy gap"
        )
        axes[d, 0].set_ylabel(
            name
        )
        axes[d, 0].set_title(
            f"{name} vs gap"
        )
        axes[d, 0].grid(
            True,
            linestyle=":",
            alpha=0.5,
        )

        axes[d, 1].scatter(
            inv_gap,
            nac,
            s=12,
            alpha=0.55,
        )
        axes[d, 1].set_xlabel(
            "1 / gap"
        )
        axes[d, 1].set_ylabel(
            name
        )
        axes[d, 1].set_title(
            f"{name} vs 1/gap"
        )
        axes[d, 1].grid(
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
        "Balanced physics loss"
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
        "Macro NAC MAE"
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
        "Macro NAC R2"
    )

    axes[1, 1].plot(
        epochs,
        history["train_h_r2"],
        label="Train h R2",
    )
    axes[1, 1].plot(
        epochs,
        history["test_h_r2"],
        label="Test h R2",
        linestyle="--",
    )
    axes[1, 1].set_title(
        "Macro numerator h R2"
    )

    for ax in axes.flat:
        ax.set_xlabel("Epoch")
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
        OUT_DIM,
        1,
        figsize=(
            16,
            4.8 * OUT_DIM,
        ),
        squeeze=False,
    )
    axes = axes[:, 0]

    for d, name in enumerate(
        PAIR_NAMES
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
            f"{name} ({NAC_UNIT_LABEL})"
        )
        axes[d].set_title(
            f"Full MD evolution - {name}"
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
# 13. CSV saving
# ============================================================

def save_predictions(
    true_nac,
    pred_nac,
    pred_h,
    gaps,
    energies,
    file_indices,
    filename,
):
    true_nac = np.asarray(
        true_nac,
        dtype=np.float64,
    )
    pred_nac = np.asarray(
        pred_nac,
        dtype=np.float64,
    )
    pred_h = np.asarray(
        pred_h,
        dtype=np.float64,
    )
    gaps = np.asarray(
        gaps,
        dtype=np.float64,
    )
    energies = np.asarray(
        energies,
        dtype=np.float64,
    )
    idx = np.asarray(
        file_indices,
        dtype=np.int64,
    ).reshape(-1)

    data = {
        "CIF_Index": idx,
    }

    for s in range(
        energies.shape[1]
    ):
        data[
            f"E{s+1}"
        ] = energies[:, s]

    for d, name in enumerate(
        PAIR_NAMES
    ):
        i, j = TARGET_PAIRS[d]

        true_h = (
            true_nac[:, d]
            * np.maximum(
                gaps[:, d],
                GAP_FLOOR,
            )
        )

        data[
            f"Gap_{i+1}_{j+1}"
        ] = gaps[:, d]

        data[
            f"True_{name}"
        ] = true_nac[:, d]

        data[
            f"Pred_{name}"
        ] = pred_nac[:, d]

        data[
            f"AbsError_{name}"
        ] = np.abs(
            pred_nac[:, d]
            - true_nac[:, d]
        )

        data[
            f"True_h_{i+1}_{j+1}"
        ] = true_h

        data[
            f"Pred_h_{i+1}_{j+1}"
        ] = pred_h[:, d]

        data[
            f"AbsError_h_{i+1}_{j+1}"
        ] = np.abs(
            pred_h[:, d]
            - true_h
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
    gaps,
    energies,
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
    gaps = np.asarray(
        gaps,
        dtype=np.float64,
    )
    energies = np.asarray(
        energies,
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

    total_len = len(full_true)

    data = {
        "Dataset_Position":
            np.arange(total_len),
        "CIF_Index":
            file_indices,
    }

    for s in range(
        energies.shape[1]
    ):
        data[
            f"E{s+1}"
        ] = energies[:, s]

    for d, name in enumerate(
        PAIR_NAMES
    ):
        i, j = TARGET_PAIRS[d]

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
            f"Gap_{i+1}_{j+1}"
        ] = gaps[:, d]

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
        f"全局 NAC evolution CSV 已保存: "
        f"{filename}"
    )


def save_metrics_per_pair(
    true_nac,
    pred_nac,
    split_name,
    filename,
):
    metrics = (
        calc_multidim_metrics(
            true_nac,
            pred_nac,
            PAIR_NAMES,
        )
    )

    rows = []

    for m in metrics["per_dim"]:
        rows.append({
            "Split":
                split_name,
            "Pair":
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
# 14. CLI
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
        "--nac-dir",
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

    # Requested real batch size
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

    # For direct cache slicing, worker=0 avoids duplicating
    # large frame tensors into worker processes.
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    p.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help=(
            "Enable only if batch=32 causes OOM; "
            "it saves memory but is slower."
        ),
    )

    p.add_argument(
        "--nac-keyword",
        default=NAC_REAL_KEYWORD,
    )

    p.add_argument(
        "--prompt",
        default=(
            "Predict the three adjacent-state nonadiabatic "
            "coupling magnitudes |Re NAC(1,2)|, "
            "|Re NAC(2,3)|, and |Re NAC(3,4)| "
            "at the target frame from the 16-frame "
            "invariant structural sequence."
        ),
    )

    return p.parse_args()


# ============================================================
# 15. Main
# ============================================================

def main():
    args = parse_args()
    set_seed(SEED)

    run_dir = args.run_dir

    cache_file = (
        args.cache_file
        or os.path.join(
            run_dir,
            "atomwise_invariant_v3_Z16-42.pt",
        )
    )

    nac_dir = (
        args.nac_dir
        or os.path.join(
            run_dir,
            "nac",
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
            "outputs_vicuna_three_nac",
        )
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "本脚本需要 CUDA GPU 运行 Vicuna-7B"
        )

    device = torch.device("cuda")

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
        "EXISTING invariant cache + frozen Vicuna-7B "
        "+ three physics-informed NAC outputs"
    )
    print("=" * 108)
    print(
        f"Run dir       : {run_dir}"
    )
    print(
        f"Cache         : {cache_file}"
    )
    print(
        f"NAC dir       : {nac_dir}"
    )
    print(
        f"Energy        : {energy_file}"
    )
    print(
        f"Vicuna        : {args.model_path}"
    )
    print(
        f"Output        : {output_root}"
    )
    print(
        f"Pairs         : "
        f"{TARGET_STATE_PAIRS_1BASED}"
    )
    print(
        f"Outputs       : {PAIR_NAMES}"
    )
    print(
        f"SEQ_LEN       : {args.seq_len}"
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
        "Structure     : DIRECT CACHE LOAD; "
        "NO CIF; NO descriptor recomputation; "
        "NO copied [N,16,A,F] dataset"
    )
    print("=" * 108)

    # --------------------------------------------------------
    # 1. Directly load existing frame-level structural cache
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
    # 2. Read existing energy / NAC labels
    # --------------------------------------------------------
    (
        energy_dict,
        n_energy_states,
    ) = read_energy_all_states(
        energy_file,
        frame_indices.tolist(),
    )

    nac_file_index = (
        build_nac_file_index(
            nac_dir,
            keyword=args.nac_keyword,
        )
    )

    # --------------------------------------------------------
    # 3. Only align lightweight sample metadata.
    #    Structural windows are NOT rebuilt/copied.
    # --------------------------------------------------------
    (
        sample_starts,
        gaps_np,
        targets_np,
        target_cif_indices,
        energies_np,
    ) = align_existing_samples(
        frame_indices,
        energy_dict,
        nac_file_index,
        seq_len=args.seq_len,
        result_dir=result_dir,
    )

    num_samples = len(
        sample_starts
    )

    # --------------------------------------------------------
    # 4. Same uniform split
    # --------------------------------------------------------
    (
        train_positions,
        test_positions,
    ) = uniform_split_indices(
        num_samples,
        train_ratio=args.train_ratio,
    )

    # --------------------------------------------------------
    # 5. Train-only feature scaling.
    #    Fit directly from training windows without materializing
    #    a separate [N,S,A,F] array.
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

    gap_scaler = (
        PairGapFeatureScaler()
        .fit(
            gaps_np[
                train_positions
            ]
        )
    )

    gap_features_all = (
        gap_scaler.transform(
            gaps_np
        )
    )

    gap_input_dim = int(
        gap_features_all.shape[1]
    )

    # --------------------------------------------------------
    # 6. Per-pair train-set loss scales, same physics logic
    #    as the previous single-NAC Vicuna code.
    # --------------------------------------------------------
    train_nac = targets_np[
        train_positions
    ]

    train_gap_safe = np.maximum(
        gaps_np[
            train_positions
        ],
        GAP_FLOOR,
    )

    train_h = (
        train_nac
        * train_gap_safe
    )

    nac_scale = np.std(
        train_nac,
        axis=0,
    )

    h_scale = np.std(
        train_h,
        axis=0,
    )

    for d in range(OUT_DIM):
        if nac_scale[d] < 1.0e-12:
            nac_scale[d] = np.mean(
                np.abs(
                    train_nac[:, d]
                )
            )
        if nac_scale[d] < 1.0e-12:
            nac_scale[d] = 1.0

        if h_scale[d] < 1.0e-12:
            h_scale[d] = np.mean(
                np.abs(
                    train_h[:, d]
                )
            )
        if h_scale[d] < 1.0e-12:
            h_scale[d] = 1.0

    log_scale = np.zeros(
        OUT_DIM,
        dtype=np.float64,
    )

    for d in range(OUT_DIM):
        positive = (
            train_nac[:, d][
                train_nac[:, d] > 0
            ]
        )

        if len(positive) > 0:
            log_scale[d] = float(
                np.median(
                    positive
                )
            )
        else:
            log_scale[d] = float(
                nac_scale[d]
            )

        log_scale[d] = max(
            log_scale[d],
            1.0e-12,
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
        f"Energy state count  : "
        f"{n_energy_states}"
    )
    print(
        f"Gap input dim       : "
        f"{gap_input_dim}"
    )
    print(
        f"NAC OUT_DIM         : "
        f"{OUT_DIM}"
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
        PAIR_NAMES
    ):
        print(
            f"{name}: "
            f"nac_scale={nac_scale[d]:.12e} | "
            f"h_scale={h_scale[d]:.12e} | "
            f"log_scale={log_scale[d]:.12e}"
        )

    # --------------------------------------------------------
    # 7. Shared frame tensors stored ONCE.
    # --------------------------------------------------------
    frame_atom_tensor = torch.from_numpy(
        frame_atom_scaled
    )
    frame_mask_tensor = torch.from_numpy(
        frame_atom_masks
    )
    frame_global_tensor = torch.from_numpy(
        frame_global_scaled
    )

    train_dataset = (
        CachedWindowThreeNACDataset(
            frame_atom_tensor,
            frame_mask_tensor,
            frame_global_tensor,
            sample_starts,
            gap_features_all,
            gaps_np,
            targets_np,
            target_cif_indices,
            train_positions,
            args.seq_len,
        )
    )

    test_dataset = (
        CachedWindowThreeNACDataset(
            frame_atom_tensor,
            frame_mask_tensor,
            frame_global_tensor,
            sample_starts,
            gap_features_all,
            gaps_np,
            targets_np,
            target_cif_indices,
            test_positions,
            args.seq_len,
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    eval_train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # --------------------------------------------------------
    # 8. Frozen Vicuna model
    # --------------------------------------------------------
    model = FrozenVicunaThreeNAC(
        model_path=args.model_path,
        atom_dim=(
            frame_atom_scaled.shape[-1]
        ),
        global_dim=(
            frame_global_scaled.shape[-1]
        ),
        out_dim=OUT_DIM,
        gap_input_dim=gap_input_dim,
        prompt=args.prompt,
        h_scale=h_scale,
        h_softplus_beta=H_SOFTPLUS_BETA,
        seq_len=args.seq_len,
        use_gradient_checkpointing=(
            args.gradient_checkpointing
        ),
    ).to(device)

    trainable_params = list(
        model.trainable_parameters()
    )

    n_trainable = sum(
        p.numel()
        for p in trainable_params
    )

    n_llm = sum(
        p.numel()
        for p in model.llm.parameters()
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
        weight_decay=args.weight_decay,
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
        MultiBalancedPhysicsLoss(
            nac_scale=nac_scale,
            h_scale=h_scale,
            log_scale=log_scale,
            w_nac=LOSS_W_NAC,
            w_h=LOSS_W_H,
            w_log=LOSS_W_LOG,
            beta=SMOOTH_L1_BETA,
        ).to(device)
    )

    # --------------------------------------------------------
    # 9. Save scalers/config
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
        gap_feature_mean=
            gap_scaler.mean_,
        gap_feature_std=
            gap_scaler.std_,
        nac_scale=
            nac_scale,
        h_scale=
            h_scale,
        log_scale=
            log_scale,
        gap_floor=
            np.asarray(
                [GAP_FLOOR]
            ),
    )

    with open(
        os.path.join(
            model_dir,
            "target_pairs.json",
        ),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "target_state_pairs_1based":
                    TARGET_STATE_PAIRS_1BASED,
                "pair_names":
                    PAIR_NAMES,
                "use_antisymmetric_average":
                    USE_ANTISYMMETRIC_AVERAGE,
                "target_abs_real":
                    TARGET_ABS_REAL,
                "physics":
                    "NAC_ij = h_ij / gap_ij",
                "vicuna_frozen":
                    True,
                "batch_size":
                    args.batch_size,
                "grad_accum":
                    args.grad_accum,
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
        "train_loss_nac": [],
        "test_loss_nac": [],
        "train_loss_h": [],
        "test_loss_h": [],
        "train_loss_log": [],
        "test_loss_log": [],
        "train_mae": [],
        "test_mae": [],
        "train_rmse": [],
        "test_rmse": [],
        "train_r2": [],
        "test_r2": [],
        "train_h_r2": [],
        "test_h_r2": [],
    }

    best_test_loss = float("inf")
    epochs_without_improvement = 0

    best_model_path = os.path.join(
        model_dir,
        "best_frozen_vicuna_three_nac.pth",
    )

    print(
        "\n" + "=" * 108
    )
    print(
        "开始训练: "
        "projector + gap encoder + 3-output h head 更新；"
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
            is_train=True,
            grad_accum_steps=(
                args.grad_accum
            ),
        )

        test_m = process_epoch(
            model,
            test_loader,
            optimizer,
            criterion,
            device,
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

        for key in [
            "loss",
            "loss_nac",
            "loss_h",
            "loss_log",
            "mae",
            "rmse",
            "r2",
            "h_r2",
        ]:
            history[
                f"train_{key}"
            ].append(
                train_m[key]
            )
            history[
                f"test_{key}"
            ].append(
                test_m[key]
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
            f"hR2 "
            f"{train_m['h_r2']:.5f}/"
            f"{test_m['h_r2']:.5f} | "
            f"LR={current_lr:.2e}"
        )

        if (
            epoch == 1
            or epoch % 20 == 0
        ):
            for m in test_m["per_dim"]:
                print(
                    f"    Test "
                    f"{m['name']}: "
                    f"MAE={m['mae']:.3e}, "
                    f"RMSE={m['rmse']:.3e}, "
                    f"R2={m['r2']:.6f}"
                )

        if (
            test_m["loss"]
            < best_test_loss
            - MIN_DELTA
        ):
            best_test_loss = (
                test_m["loss"]
            )
            epochs_without_improvement = 0

            checkpoint = {
                "trainable_state_dict":
                    model.trainable_state_dict(),
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
                "gap_input_dim":
                    int(gap_input_dim),
                "out_dim":
                    int(OUT_DIM),
                "seq_len":
                    int(args.seq_len),
                "train_ratio":
                    float(
                        args.train_ratio
                    ),
                "batch_size":
                    int(args.batch_size),
                "grad_accum":
                    int(args.grad_accum),
                "prompt":
                    args.prompt,
                "best_epoch":
                    int(epoch),
                "best_test_loss":
                    float(
                        best_test_loss
                    ),
                "h_softplus_beta":
                    float(
                        H_SOFTPLUS_BETA
                    ),
                "nac_scale":
                    nac_scale.tolist(),
                "h_scale":
                    h_scale.tolist(),
                "log_scale":
                    log_scale.tolist(),
                "target_pairs_1based":
                    TARGET_STATE_PAIRS_1BASED,
                "pair_names":
                    PAIR_NAMES,
                "target_definition":
                    "abs(real NAC) for "
                    "(1,2),(2,3),(3,4)",
                "physics":
                    "NAC_ij = h_ij / gap_ij",
                "atom_scaler":
                    scaler_to_dict(
                        atom_scaler
                    ),
                "global_scaler":
                    scaler_to_dict(
                        global_scaler
                    ),
                "gap_scaler": {
                    "mean_":
                        gap_scaler
                        .mean_
                        .tolist(),
                    "std_":
                        gap_scaler
                        .std_
                        .tolist(),
                    "n_pairs_":
                        int(
                            gap_scaler
                            .n_pairs_
                        ),
                },
                "descriptor_state":
                    descriptor_state,
                "cache_file":
                    cache_file,
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

        # save history every epoch
        pd.DataFrame(
            history
        ).to_csv(
            os.path.join(
                log_dir,
                "training_metrics_history.csv",
            ),
            index=False,
        )

        # Plot less frequently; this does not alter training logic.
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
                f"Early stopping: 连续 "
                f"{args.patience} 个 epoch "
                f"无改善，停止于 epoch={epoch}"
            )
            break

    # final training history plot
    plot_training_history(
        history,
        os.path.join(
            plot_dir,
            "training_curves.png",
        ),
    )

    # --------------------------------------------------------
    # 11. Diagnostics
    # --------------------------------------------------------
    plot_gap_relationship(
        gaps_np,
        targets_np,
        os.path.join(
            plot_dir,
            "nac_vs_gap.png",
        ),
    )

    # --------------------------------------------------------
    # 12. Final evaluation
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
        train_h_pred,
        train_gap,
        train_cif_idx,
    ) = predict_loader(
        model,
        eval_train_loader,
        device,
    )

    (
        test_true,
        test_pred,
        test_h_pred,
        test_gap,
        test_cif_idx,
    ) = predict_loader(
        model,
        test_loader,
        device,
    )

    train_metrics = (
        calc_multidim_metrics(
            train_true,
            train_pred,
            PAIR_NAMES,
        )
    )

    test_metrics = (
        calc_multidim_metrics(
            test_true,
            test_pred,
            PAIR_NAMES,
        )
    )

    train_h_true = (
        train_true
        * np.maximum(
            train_gap,
            GAP_FLOOR,
        )
    )

    test_h_true = (
        test_true
        * np.maximum(
            test_gap,
            GAP_FLOOR,
        )
    )

    h_names = [
        f"h_{i+1}_{j+1}"
        for i, j in TARGET_PAIRS
    ]

    train_h_metrics = (
        calc_multidim_metrics(
            train_h_true,
            train_h_pred,
            h_names,
        )
    )

    test_h_metrics = (
        calc_multidim_metrics(
            test_h_true,
            test_h_pred,
            h_names,
        )
    )

    print(
        "\n" + "=" * 108
    )
    print("最终结果")
    print("=" * 108)

    print(
        f"Train Macro NAC: "
        f"MAE={train_metrics['mae']:.8e} | "
        f"RMSE={train_metrics['rmse']:.8e} | "
        f"R2={train_metrics['r2']:.8f}"
    )

    print(
        f"Test  Macro NAC: "
        f"MAE={test_metrics['mae']:.8e} | "
        f"RMSE={test_metrics['rmse']:.8e} | "
        f"R2={test_metrics['r2']:.8f}"
    )

    print(
        f"Train Macro h  : "
        f"MAE={train_h_metrics['mae']:.8e} | "
        f"RMSE={train_h_metrics['rmse']:.8e} | "
        f"R2={train_h_metrics['r2']:.8f}"
    )

    print(
        f"Test  Macro h  : "
        f"MAE={test_h_metrics['mae']:.8e} | "
        f"RMSE={test_h_metrics['rmse']:.8e} | "
        f"R2={test_h_metrics['r2']:.8f}"
    )

    print("\nTest per pair:")
    for m in test_metrics["per_dim"]:
        print(
            f"  {m['name']}: "
            f"MAE={m['mae']:.8e} | "
            f"RMSE={m['rmse']:.8e} | "
            f"R2={m['r2']:.8f}"
        )

    print("=" * 108)

    # --------------------------------------------------------
    # 13. Plots
    # --------------------------------------------------------
    plot_pair_comparison(
        train_true,
        train_pred,
        "Train",
        os.path.join(
            plot_dir,
            "train_three_nac_prediction.png",
        ),
    )

    plot_pair_comparison(
        test_true,
        test_pred,
        "Test",
        os.path.join(
            plot_dir,
            "test_three_nac_prediction.png",
        ),
    )

    plot_h_comparison(
        train_true,
        train_h_pred,
        train_gap,
        "Train",
        os.path.join(
            plot_dir,
            "train_three_h_prediction.png",
        ),
    )

    plot_h_comparison(
        test_true,
        test_h_pred,
        test_gap,
        "Test",
        os.path.join(
            plot_dir,
            "test_three_h_prediction.png",
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
            "full_md_three_nac_evolution.png",
        ),
    )

    # --------------------------------------------------------
    # 14. CSVs
    # --------------------------------------------------------
    save_predictions(
        train_true,
        train_pred,
        train_h_pred,
        train_gap,
        energies_np[
            train_positions
        ],
        train_cif_idx,
        os.path.join(
            result_dir,
            "train_three_nac_predictions.csv",
        ),
    )

    save_predictions(
        test_true,
        test_pred,
        test_h_pred,
        test_gap,
        energies_np[
            test_positions
        ],
        test_cif_idx,
        os.path.join(
            result_dir,
            "test_three_nac_predictions.csv",
        ),
    )

    save_full_time_series_csv(
        targets_np,
        gaps_np,
        energies_np,
        train_positions,
        train_pred,
        test_positions,
        test_pred,
        target_cif_indices,
        os.path.join(
            result_dir,
            "full_md_three_nac_evolution.csv",
        ),
    )

    save_metrics_per_pair(
        train_true,
        train_pred,
        "Train",
        os.path.join(
            result_dir,
            "train_metrics_per_pair.csv",
        ),
    )

    save_metrics_per_pair(
        test_true,
        test_pred,
        "Test",
        os.path.join(
            result_dir,
            "test_metrics_per_pair.csv",
        ),
    )

    # --------------------------------------------------------
    # 15. Summary
    # --------------------------------------------------------
    summary = {
        "architecture": (
            "existing frame-level invariant cache -> "
            "masked atom mean + global -> "
            "Self-Attention + Cross-Attention + FC projector -> "
            "frozen Vicuna -> "
            "3-pair gap fusion -> 3 positive h -> "
            "NAC_ij=h_ij/gap_ij"
        ),
        "model_path":
            args.model_path,
        "cache_file":
            cache_file,
        "nac_dir":
            nac_dir,
        "energy_file":
            energy_file,
        "target_pairs_1based":
            TARGET_STATE_PAIRS_1BASED,
        "pair_names":
            PAIR_NAMES,
        "out_dim":
            OUT_DIM,
        "seq_len":
            int(args.seq_len),
        "train_ratio":
            float(args.train_ratio),
        "batch_size":
            int(args.batch_size),
        "grad_accum":
            int(args.grad_accum),
        "samples":
            int(num_samples),
        "train_samples":
            int(
                len(train_positions)
            ),
        "test_samples":
            int(
                len(test_positions)
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
        "train_macro_nac_mae":
            train_metrics["mae"],
        "train_macro_nac_rmse":
            train_metrics["rmse"],
        "train_macro_nac_r2":
            train_metrics["r2"],
        "test_macro_nac_mae":
            test_metrics["mae"],
        "test_macro_nac_rmse":
            test_metrics["rmse"],
        "test_macro_nac_r2":
            test_metrics["r2"],
        "train_per_pair":
            train_metrics["per_dim"],
        "test_per_pair":
            test_metrics["per_dim"],
        "train_macro_h_r2":
            train_h_metrics["r2"],
        "test_macro_h_r2":
            test_h_metrics["r2"],
        "target_definition":
            "abs(real NAC) for "
            "(1,2),(2,3),(3,4)",
        "loss":
            "SmoothL1(NAC) + "
            "SmoothL1(h) + "
            "SmoothL1(logNAC)",
        "physics":
            "NAC_ij = h_theta_ij / gap_ij",
        "structure_data_note": (
            "No CIF reading and no descriptor recomputation. "
            "The existing frame cache is loaded directly. "
            "16-frame windows are sliced on demand by index "
            "instead of materializing a duplicated structural dataset."
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
        f"test_three_nac_prediction.png"
    )
    print(
        f"  {plot_dir}/"
        f"full_md_three_nac_evolution.png"
    )
    print(
        f"  {plot_dir}/"
        f"test_three_h_prediction.png"
    )
    print(
        f"  {plot_dir}/"
        f"nac_vs_gap.png"
    )
    print(
        f"  {result_dir}/"
        f"test_three_nac_predictions.csv"
    )
    print(
        f"  {result_dir}/"
        f"test_metrics_per_pair.csv"
    )
    print(
        f"  best checkpoint: "
        f"{best_model_path}"
    )


if __name__ == "__main__":
    main()
