#!/usr/bin/env python
"""
Utility to convert Sheet1/Sheet2/Sheet3 CSV exports into an HDF5 dataset that mimics
the PKRNN-2CM training batches.

The generated HDF5 file will contain:
  - Groups `train`, `valid`, `test`, each with a single batch (`'0'`) exposing the
    tensors expected by `DataFromH5py` (ContTensor, CatTensor, LabelTensor, etc.).
  - A `CodeDict/data` dataset storing the categorical code mapping as a string.
  - A `stats` table (pandas HDF key) with per-feature mean/std for the continuous inputs.

Assumptions and simplifications compared to the original study:
  * The CSVs cover four patients; we split them into train/valid/test as (2/1/1).
  * Continuous features are z-scored with statistics computed across all events.
  * DoseTensor stores the administered amount in grams (DoseAtOnce/1000). TimeDiffTensor
    is expressed in days and derived from consecutive TestDay stamps.
  * Vancomycin labels use the first available measurement (VancomycinLevel1 or Level2);
    MaskTensor flags whether a measurement exists.
  * Volume and clearance scaffolding is approximated from weight (VTensor) and
    Cockcroft-Gault creatinine clearance (VancoClTensor).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd


# ------------------------------ Data containers ------------------------------ #

@dataclass
class PatientSequence:
    """Holds all arrays required for a single patient's sequence."""

    patient_id: str
    cont_tensor: np.ndarray  # (seq_len, n_cont)
    cat_tensor: np.ndarray  # (seq_len, max_cat)
    labels: np.ndarray  # (seq_len,)
    mask: np.ndarray  # (seq_len,)
    dose: np.ndarray  # (seq_len,)
    time_diff: np.ndarray  # (seq_len,)
    v_tensor: np.ndarray  # (seq_len,)
    vanco_cl: np.ndarray  # (seq_len,)
    length: int


# ------------------------------ CSV ingestion ------------------------------- #

def _load_sheet1(path: Path) -> pd.DataFrame:
    """Load Sheet1 and normalise column names."""
    df = pd.read_csv(path, encoding="cp949")
    rename_map = {
        "PatientID-samecycle": "cycle",
        "TestDay": "date_str",
        "Height": "height",
        "Weight": "weight",
        "BMI": "bmi",
        "VancomycinLevel_Collection time1": "vanco_time1",
        "VancomycinLevel1": "vanco_level1",
        "VancomycinLevel_Collection time2": "vanco_time2",
        "VancomycinLevel2": "vanco_level2",
        "Diagnosis": "diagnosis",
        "DoseAtOnce(mg)": "dose_at_once_mg",
        "DoseTimesPerDay": "dose_times_per_day",
        "DosePerDay(mg)": "dose_per_day_mg",
        "AUC": "auc",
        "Crcl (L/hr) \nCockcroft-Gault [PKS���� �ƴ� ���� ��] ": "crcl_l_hr",
        "Crcl (ml/min) Cockcroft-Gault [PKS���� �ƴ� ���� ��]\n CrCL (mL/min) = (140 - Age*) �� Weight (kg) / 72 �� Scr�� (mg/dL) x 0.85 [if female]": "crcl_ml_min",
        "Crcl (ml/min/kg) Cockcroft-Gault \n[PKS���� �ƴ� ���� ��] ": "crcl_ml_min_kg",
        "PKSestimated\r\nPeak": "pks_peak",
        "PKSestimated\r\nTrough": "pks_trough",
        "PKSestimated\nVDistLiter": "pks_vdist_l",
        "PKSestimated\nVDistLiterPerKilogram": "pks_vdist_l_kg",
        "PKSestimated\nClearenceLiterPerHour": "pks_clear_l_hr",
        "PKSestimated\nClearence(mL/min/kg)": "pks_clear_ml_min_kg",
        "PKSestimated\nHalflife": "pks_halflife",
        "WBC": "wbc",
        "RBC": "rbc",
        "Hb": "hb",
        "Hct": "hct",
        "MCV": "mcv",
        "MCH": "mch",
        "MCHC": "mchc",
        "RDW": "rdw",
        "Platelet": "platelet",
        "ANC": "anc",
        "Serum Cr": "serum_cr",
        "Calcium": "calcium",
        "Phosphorus": "phosphorus",
        "Uric acid": "uric_acid",
        "Chol.": "chol",
        "T. Protein": "total_protein",
        "Albumin": "albumin",
        "T. Bil.": "total_bili",
        "Alk. phos.": "alk_phos",
        "AST(GOT)": "ast",
        "ALT(GPT)": "alt",
        "Na": "na",
        "Na.1": "na_alt",
        "K": "k",
        "Cl": "cl",
        "TCO2": "tco2",
        "hs-CRP": "hs_crp",
    }
    df = df.rename(columns=rename_map)
    df["date"] = pd.to_datetime(df["date_str"], errors="coerce")
    return df


def _load_sheet2(path: Path) -> pd.DataFrame:
    """Load Sheet2 containing renal function markers."""
    df = pd.read_csv(path, encoding="cp949")
    df = df.rename(columns={"TestDay": "date", "BUN": "bun", "Creatinine": "creatinine"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def _expand_dose_map(row: pd.Series) -> List[Tuple[pd.Timestamp, float]]:
    """Expand a 'Daily_Dose_Map' string into (date, dose_mg) tuples."""
    date_entries: List[Tuple[pd.Timestamp, float]] = []
    raw_map = str(row["Daily_Dose_Map (M/D=mg; ...)"]).strip()
    if not raw_map or raw_map.lower() == "nan":
        return date_entries
    earliest = pd.to_datetime(row["Earliest E (yyyy-mm-dd)"], errors="coerce")
    current_year = earliest.year if pd.notnull(earliest) else 2000
    prev_month = earliest.month if pd.notnull(earliest) else 1
    for token in raw_map.split(";"):
        token = token.strip()
        if not token:
            continue
        try:
            date_part, value_part = token.split("=")
        except ValueError:
            continue
        month_str, day_str = [t.strip() for t in date_part.split("/")]
        try:
            month = int(month_str)
            day = int(day_str)
        except ValueError:
            continue
        if month < prev_month:
            current_year += 1
        prev_month = month
        try:
            dose_date = pd.Timestamp(year=current_year, month=month, day=day)
        except ValueError:
            continue
        try:
            dose_mg = float(value_part.strip())
        except ValueError:
            continue
        date_entries.append((dose_date, dose_mg))
    return date_entries


def _load_sheet3(path: Path) -> pd.DataFrame:
    """Load Sheet3 and explode the per-day dose mapping."""
    df = pd.read_csv(path, encoding="cp949")
    expanded_records: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        patient_id = row["PatientID"]
        cycle = row["PatientID-samecycle"]
        for date, dose_mg in _expand_dose_map(row):
            expanded_records.append(
                {
                    "PatientID": patient_id,
                    "cycle": cycle,
                    "date": date,
                    "daily_dose_mg": dose_mg,
                }
            )
    return pd.DataFrame(expanded_records)


# ---------------------------- Feature engineering --------------------------- #

CONT_FEATURES: Tuple[str, ...] = (
    "height",
    "weight",
    "bmi",
    "dose_at_once_mg",
    "dose_times_per_day",
    "dose_per_day_mg",
    "auc",
    "crcl_l_hr",
    "crcl_ml_min",
    "crcl_ml_min_kg",
    "pks_peak",
    "pks_trough",
    "pks_vdist_l",
    "pks_vdist_l_kg",
    "pks_clear_l_hr",
    "pks_clear_ml_min_kg",
    "pks_halflife",
    "wbc",
    "rbc",
    "hb",
    "hct",
    "mcv",
    "mch",
    "mchc",
    "rdw",
    "platelet",
    "anc",
    "serum_cr",
    "calcium",
    "phosphorus",
    "uric_acid",
    "chol",
    "total_protein",
    "albumin",
    "total_bili",
    "alk_phos",
    "ast",
    "alt",
    "na",
    "na_alt",
    "k",
    "cl",
    "tco2",
    "hs_crp",
    "bun",
    "creatinine",
    "daily_dose_mg",
)


def _build_patient_frames(
    sheet1: pd.DataFrame,
    sheet2: pd.DataFrame,
    sheet3: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Merge the three sheets into per-patient time-series frames."""
    patients: List[str] = sorted(sheet1["PatientID"].astype(str).unique())
    patient_frames: Dict[str, pd.DataFrame] = {}

    for patient_id in patients:
        df1 = sheet1[sheet1["PatientID"] == int(patient_id)].copy()
        df2 = sheet2[sheet2["PatientID"] == int(patient_id)].copy()
        df3 = sheet3[sheet3["PatientID"] == int(patient_id)].copy() if not sheet3.empty else pd.DataFrame(columns=["date", "daily_dose_mg"])

        df1["date"] = df1["date"]
        df2["date"] = df2["date"]

        date_union = sorted(
            set(df1["date"].dropna()).union(df2["date"].dropna()).union(df3["date"].dropna())
        )
        if not date_union:
            continue

        events: List[Dict[str, object]] = []
        for date in date_union:
            event: Dict[str, object] = {
                "PatientID": patient_id,
                "date": date,
            }
            row1 = df1[df1["date"] == date].head(1)
            if not row1.empty:
                event.update(row1.iloc[0].to_dict())
            row2 = df2[df2["date"] == date].head(1)
            if not row2.empty:
                event.update(row2.iloc[0].to_dict())
            row3 = df3[df3["date"] == date].head(1)
            if not row3.empty:
                event["daily_dose_mg"] = row3.iloc[0]["daily_dose_mg"]
            events.append(event)

        patient_df = pd.DataFrame(events).sort_values("date").reset_index(drop=True)

        # Forward/backward fill to reduce missingness.
        patient_df["cycle"] = patient_df["cycle"].ffill().bfill()
        patient_df["diagnosis"] = patient_df["diagnosis"].ffill().bfill().fillna("Unknown")
        for col in CONT_FEATURES:
            if col not in patient_df.columns:
                patient_df[col] = np.nan
        for col in CONT_FEATURES:
            patient_df[col] = (
                pd.to_numeric(patient_df[col], errors="coerce")
                .ffill()
                .bfill()
            )

        patient_frames[patient_id] = patient_df

    return patient_frames


def _compute_stats(frame_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute z-score statistics for continuous features across all patients."""
    concat = pd.concat(
        [df.set_index("date")[list(CONT_FEATURES)] for df in frame_dict.values()],
        axis=0,
        ignore_index=True,
    )
    stats_df = pd.DataFrame({"FEATURES": CONT_FEATURES})
    stats_df["MEAN"] = concat.mean(skipna=True).values
    stats_df["STD"] = concat.std(skipna=True, ddof=0).replace(0, 1e-6).values
    return stats_df


def _encode_categories(value: str, code_dict: Dict[str, int], next_code: int) -> Tuple[int, Dict[str, int], int]:
    """Assign an integer code to a categorical token."""
    if value not in code_dict:
        code_dict[value] = next_code
        next_code += 1
    return code_dict[value], code_dict, next_code


def _build_sequences(
    patient_frames: Dict[str, pd.DataFrame],
    stats_df: pd.DataFrame,
) -> Tuple[List[PatientSequence], Dict[str, int]]:
    """Convert per-patient frames into PKRNN-2CM ready tensors."""
    mean_lookup = stats_df.set_index("FEATURES")["MEAN"]
    std_lookup = stats_df.set_index("FEATURES")["STD"]

    sequences: List[PatientSequence] = []
    code_dict: Dict[str, int] = {}
    next_code = 1  # reserve 0 for padding
    max_cat_per_event = 0

    for patient_id, frame in patient_frames.items():
        frame = frame.sort_values("date").reset_index(drop=True)
        for col in CONT_FEATURES:
            if col not in frame:
                frame[col] = np.nan
        frame["daily_dose_mg"] = frame["daily_dose_mg"].fillna(0.0)
        if "bun" in frame:
            frame["bun"] = pd.to_numeric(frame["bun"], errors="coerce").ffill().bfill()
        else:
            frame["bun"] = 0.0
        if "creatinine" in frame:
            frame["creatinine"] = (
                pd.to_numeric(frame["creatinine"], errors="coerce").ffill().bfill()
            )
        else:
            frame["creatinine"] = 0.0

        cont_matrix = []
        label_list = []
        mask_list = []
        dose_list = []
        time_diff_list = []
        v_tensor_list = []
        vanco_cl_list = []
        cat_rows: List[List[int]] = []

        dates = frame["date"].tolist()
        for idx, row in frame.iterrows():
            cont_row = []
            for feat in CONT_FEATURES:
                value = row.get(feat, np.nan)
                if pd.isna(value):
                    value = mean_lookup[feat]
                mean = mean_lookup[feat]
                std = std_lookup[feat] if std_lookup[feat] != 0 else 1e-6
                cont_row.append((value - mean) / std)
            cont_matrix.append(cont_row)

            label = row.get("vanco_level1")
            if pd.isna(label):
                label = row.get("vanco_level2")
            if pd.isna(label):
                mask = 0.0
                label = 0.0
            else:
                mask = 1.0
            label_list.append(label)
            mask_list.append(mask)

            dose_tensor_value = row.get("dose_at_once_mg")
            dose_list.append(np.nan_to_num(dose_tensor_value, nan=0.0) / 1000.0)

            if idx < len(dates) - 1:
                delta = (dates[idx + 1] - dates[idx]).days
                delta = max(delta, 0)
            else:
                delta = 0
            time_diff_list.append(float(delta))

            weight = row.get("weight")
            v_tensor_list.append(np.nan_to_num(weight, nan=0.0) * 0.0007)

            cl_value = row.get("crcl_l_hr")
            if pd.isna(cl_value):
                cl_value = row.get("crcl_ml_min")
                cl_value = np.nan if pd.isna(cl_value) else cl_value * 0.06
            vanco_cl_list.append(np.nan_to_num(cl_value, nan=0.0))

            diag_token = f"diagnosis:{row.get('diagnosis', 'Unknown')}"
            cycle_token = f"cycle:{row.get('cycle', 'Unknown')}"
            dose_freq = row.get("dose_times_per_day")
            if pd.isna(dose_freq):
                dose_token = "dose_freq:missing"
            else:
                dose_token = f"dose_freq:{int(round(dose_freq))}"
            tokens = [diag_token, cycle_token, dose_token]
            encoded_row: List[int] = []
            for token in tokens:
                code, code_dict, next_code = _encode_categories(token, code_dict, next_code)
                encoded_row.append(code)
            cat_rows.append(encoded_row)
            max_cat_per_event = max(max_cat_per_event, len(encoded_row))

        cont_tensor = np.array(cont_matrix, dtype=np.float32)
        cat_tensor = np.array(cat_rows, dtype=np.int32)
        labels = np.array(label_list, dtype=np.float32)
        mask = np.array(mask_list, dtype=np.float32)
        dose = np.array(dose_list, dtype=np.float32)
        time_diff = np.array(time_diff_list, dtype=np.float32)
        v_tensor = np.array(v_tensor_list, dtype=np.float32)
        vanco_cl = np.array(vanco_cl_list, dtype=np.float32)

        sequences.append(
            PatientSequence(
                patient_id=patient_id,
                cont_tensor=cont_tensor,
                cat_tensor=cat_tensor,
                labels=labels,
                mask=mask,
                dose=dose,
                time_diff=time_diff,
                v_tensor=v_tensor,
                vanco_cl=vanco_cl,
                length=cont_tensor.shape[0],
            )
        )

    # Pad categorical rows to uniform length.
    for seq in sequences:
        if seq.cat_tensor.shape[1] < max_cat_per_event:
            padding = np.zeros(
                (seq.cat_tensor.shape[0], max_cat_per_event - seq.cat_tensor.shape[1]),
                dtype=np.int32,
            )
            seq.cat_tensor = np.hstack([seq.cat_tensor, padding])

    return sequences, code_dict


# ------------------------------ Batch assembly ------------------------------ #

def _pad_to_batch(arrays: Sequence[np.ndarray], pad_value: float = 0.0) -> np.ndarray:
    """Pad arrays along the sequence axis to create a batch tensor."""
    max_len = max(arr.shape[0] for arr in arrays)
    feature_dim = arrays[0].shape[1] if arrays[0].ndim == 2 else None
    dtype = arrays[0].dtype
    if feature_dim is None:
        batch = np.full((len(arrays), max_len), pad_value, dtype=dtype)
        for i, arr in enumerate(arrays):
            batch[i, : arr.shape[0]] = arr
        return batch

    batch = np.full((len(arrays), max_len, feature_dim), pad_value, dtype=dtype)
    for i, arr in enumerate(arrays):
        batch[i, : arr.shape[0], :] = arr
    return batch


def _pad_vector(arrays: Sequence[np.ndarray], pad_value: float = 0.0) -> np.ndarray:
    """Pad 1D arrays to form a 2D batch tensor."""
    max_len = max(arr.shape[0] for arr in arrays)
    dtype = arrays[0].dtype
    batch = np.full((len(arrays), max_len), pad_value, dtype=dtype)
    for i, arr in enumerate(arrays):
        batch[i, : arr.shape[0]] = arr
    return batch


def _write_batch(h5_group: h5py.Group, batch_name: str, sequences: Sequence[PatientSequence]) -> None:
    """Create datasets for a single batch within a split group."""
    batch_group = h5_group.create_group(batch_name)

    cont_batch = _pad_to_batch([seq.cont_tensor for seq in sequences], pad_value=0.0)
    cat_batch = _pad_to_batch([seq.cat_tensor for seq in sequences], pad_value=0)
    label_batch = _pad_vector([seq.labels for seq in sequences], pad_value=0.0)
    mask_batch = _pad_vector([seq.mask for seq in sequences], pad_value=0.0)
    dose_batch = _pad_vector([seq.dose for seq in sequences], pad_value=0.0)
    timediff_batch = _pad_vector([seq.time_diff for seq in sequences], pad_value=0.0)
    v_batch = _pad_vector([seq.v_tensor for seq in sequences], pad_value=0.0)
    vcl_batch = _pad_vector([seq.vanco_cl for seq in sequences], pad_value=0.0)
    pt_list = np.array([seq.patient_id for seq in sequences], dtype="S20")
    lengths = np.array([seq.length for seq in sequences], dtype=np.int32)

    batch_group.create_dataset("ContTensor", data=cont_batch)
    batch_group.create_dataset("CatTensor", data=cat_batch)
    batch_group.create_dataset("LabelTensor", data=label_batch)
    batch_group.create_dataset("MaskTensor", data=mask_batch)
    batch_group.create_dataset("DoseTensor", data=dose_batch)
    batch_group.create_dataset("TimeDiffTensor", data=timediff_batch)
    batch_group.create_dataset("VTensor", data=v_batch)
    batch_group.create_dataset("VancoClTensor", data=vcl_batch)
    batch_group.create_dataset("PtList", data=pt_list)
    batch_group.create_dataset("LengList", data=lengths)


def _write_split(
    parent: h5py.File,
    split_name: str,
    sequences: Sequence[PatientSequence],
    batch_size: int,
) -> None:
    """Chunk the sequences into batches and write them under `split_name`."""
    if not sequences:
        return
    split_group = parent.create_group(split_name)
    for batch_idx in range(0, len(sequences), batch_size):
        batch_seqs = list(sequences[batch_idx : batch_idx + batch_size])
        if not batch_seqs:
            continue
        while len(batch_seqs) < batch_size:
            batch_seqs.append(batch_seqs[-1])
        _write_batch(split_group, str(batch_idx // batch_size), batch_seqs)


# ------------------------------- Main routine ------------------------------- #

def main(args: argparse.Namespace) -> None:
    sheet1 = _load_sheet1(args.sheet1)
    sheet2 = _load_sheet2(args.sheet2)
    sheet3 = _load_sheet3(args.sheet3)

    patient_frames = _build_patient_frames(sheet1, sheet2, sheet3)
    stats_df = _compute_stats(patient_frames)
    sequences, code_dict = _build_sequences(patient_frames, stats_df)

    # Deterministic split: two patients for train, one for valid, one for test.
    sequences_sorted = sorted(sequences, key=lambda s: s.patient_id)
    train_ids = {seq.patient_id for seq in sequences_sorted[:2]}
    valid_ids = {sequences_sorted[2].patient_id} if len(sequences_sorted) > 2 else set()
    test_ids = {seq.patient_id for seq in sequences_sorted[3:]} if len(sequences_sorted) > 3 else set()

    train_sequences = [seq for seq in sequences_sorted if seq.patient_id in train_ids]
    valid_sequences = [seq for seq in sequences_sorted if seq.patient_id in valid_ids]
    test_sequences = [seq for seq in sequences_sorted if seq.patient_id in test_ids]

    output_path = Path(args.output)
    if output_path.exists():
        output_path.unlink()

    with h5py.File(output_path, "w") as h5f:
        if train_sequences:
            _write_split(h5f, "train", train_sequences, args.batch_size)
        if valid_sequences:
            _write_split(h5f, "valid", valid_sequences, args.batch_size)
        if test_sequences:
            _write_split(h5f, "test", test_sequences, args.batch_size)

        code_group = h5f.create_group("CodeDict")
        code_group.create_dataset("data", data=np.bytes_(str(code_dict)))

    # Append stats table via pandas (re-open in append mode).
    stats_df.to_hdf(output_path, key="stats", mode="a", index=False)

    print(f"HDF5 dataset written to {output_path}")
    print(f"Continuous features: {len(CONT_FEATURES)}")
    print(f"Categorical codes: {len(code_dict)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Sheet CSV files into PKRNN-2CM style HDF5.")
    parser.add_argument("--sheet1", type=Path, default=Path("Sheet1.csv"), help="Path to Sheet1 CSV.")
    parser.add_argument("--sheet2", type=Path, default=Path("Sheet2.csv"), help="Path to Sheet2 CSV.")
    parser.add_argument("--sheet3", type=Path, default=Path("Sheet3.csv"), help="Path to Sheet3 CSV.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Target batch size for each split (batches are padded by repeating the last patient).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pk_sheets_dataset.h5",
        help="Destination HDF5 filename.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(_parse_args())
