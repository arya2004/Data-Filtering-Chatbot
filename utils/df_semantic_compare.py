from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np

@dataclass
class CompareOptions:
    keys: Optional[List[str]] = None
    ignore_columns: List[str] = field(default_factory=list)
    ignore_column_order: bool = True
    ignore_row_order: bool = True
    atol: float = 0.0
    rtol: float = 0.0
    float_decimals: Optional[int] = None
    case_insensitive: bool = False
    strip_strings: bool = True
    collapse_whitespace: bool = True
    coerce_numeric: bool = True
    coerce_datetime: bool = True
    check_dtype: bool = False
    na_rep: Any = "<<NA>>"

def _normalize_strings(s: pd.Series, opts: CompareOptions) -> pd.Series:
    s = s.astype("string")
    if opts.strip_strings:
        s = s.str.strip()
    if opts.collapse_whitespace:
        s = s.str.replace(r"\s+", " ", regex=True)
    if opts.case_insensitive:
        s = s.str.lower()
    return s

def _maybe_coerce(df: pd.DataFrame, opts: CompareOptions) -> pd.DataFrame:
    out = df.copy()
    if opts.ignore_columns:
        out = out.drop(columns=[c for c in opts.ignore_columns if c in out.columns], errors="ignore")
    if opts.coerce_numeric:
        for c in out.columns:
            if out[c].dtype == "object":
                out[c] = pd.to_numeric(out[c], errors="ignore")
    for c in out.columns:
        if pd.api.types.is_string_dtype(out[c]):
            out[c] = _normalize_strings(out[c], opts)
    if opts.coerce_datetime:
        for c in out.columns:
            if pd.api.types.is_datetime64_any_dtype(out[c]):
                out[c] = pd.to_datetime(out[c], utc=False)
            elif out[c].dtype == "object":
                coerced = pd.to_datetime(out[c], errors="ignore")
                if pd.api.types.is_datetime64_any_dtype(coerced):
                    out[c] = coerced
    if opts.float_decimals is not None:
        for c in out.columns:
            if pd.api.types.is_float_dtype(out[c]):
                out[c] = out[c].round(opts.float_decimals)
    if opts.ignore_column_order:
        out = out.reindex(sorted(out.columns), axis=1)
    return out

def _row_signature_frame(df: pd.DataFrame, opts: CompareOptions) -> pd.Series:
    cols = list(df.columns)
    def normalize_value(v):
        if pd.isna(v):
            return opts.na_rep
        if isinstance(v, float) and opts.float_decimals is not None:
            return round(v, opts.float_decimals)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (pd.Timestamp, np.datetime64)):
            return pd.Timestamp(v).isoformat()
        return v
    tuples = [tuple(normalize_value(v) for v in row) for row in df[cols].itertuples(index=False, name=None)]
    return pd.Series(tuples)

def _numeric_close(a: pd.Series, b: pd.Series, opts: CompareOptions) -> pd.Series:
    a_values = a.to_numpy()
    b_values = b.to_numpy()
    both_nan = pd.isna(a_values) & pd.isna(b_values)
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        close = np.isclose(a_values, b_values, rtol=opts.rtol, atol=opts.atol, equal_nan=True)
        return pd.Series(close | both_nan, index=a.index)
    return pd.Series((a_values == b_values) | both_nan, index=a.index)

def _sort_all_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    if not cols:
        return df
    try:
        return df.sort_values(by=cols).reset_index(drop=True)
    except Exception:
        key = df.apply(lambda row: tuple((v if isinstance(v, (int, float)) else str(v)) for v in row), axis=1)
        return df.assign(__key=key).sort_values(by="__key").drop(columns="__key").reset_index(drop=True)

def compare_dataframes(df_left: pd.DataFrame, df_right: pd.DataFrame, opts: CompareOptions = CompareOptions()) -> Dict[str, Any]:
    L = _maybe_coerce(df_left, opts)
    R = _maybe_coerce(df_right, opts)

    missing_in_right = [c for c in L.columns if c not in R.columns]
    extra_in_right   = [c for c in R.columns if c not in L.columns]

    report: Dict[str, Any] = {
        "equal": False,
        "shape_left": L.shape,
        "shape_right": R.shape,
        "missing_in_right": missing_in_right,
        "extra_in_right": extra_in_right,
        "column_mismatch": list(sorted(set(missing_in_right + extra_in_right))),
        "row_deltas": None,
        "cell_mismatches": {},
        "keys_used": opts.keys,
        "tolerances": {"rtol": opts.rtol, "atol": opts.atol, "float_decimals": opts.float_decimals},
    }

    if report["column_mismatch"]:
        return report

    # Keyed alignment
    if opts.keys:
        key = opts.keys
        if not set(key).issubset(L.columns) or not set(key).issubset(R.columns):
            report["row_deltas"] = {"reason": "missing key in one side"}
            return report

        Lk = L.set_index(key, drop=False)
        Rk = R.set_index(key, drop=False)
        all_idx = Lk.index.union(Rk.index)
        Lk = Lk.reindex(all_idx)
        Rk = Rk.reindex(all_idx)

        if len(key) == 1:
            missing_rows = Rk[key[0]].isna().sum()
            extra_rows   = Lk[key[0]].isna().sum()
        else:
            missing_rows = Rk[key].isna().any(axis=1).sum()
            extra_rows   = Lk[key].isna().any(axis=1).sum()

        mismatches = {}
        for c in L.columns:
            if c in key:
                continue
            same = _numeric_close(Lk[c], Rk[c], opts)
            mismatches[c] = int((~same).sum())

        total_mismatch = sum(mismatches.values()) + int(missing_rows) + int(extra_rows)
        report["row_deltas"] = {"missing_rows_in_right": int(missing_rows), "extra_rows_in_right": int(extra_rows)}
        report["cell_mismatches"] = mismatches
        report["equal"] = (total_mismatch == 0)
        return report

    # No keys: order-independent
    if (opts.atol > 0 or opts.rtol > 0) and opts.ignore_row_order:
        if L.shape != R.shape:
            sigL = _row_signature_frame(L, opts).value_counts()
            sigR = _row_signature_frame(R, opts).value_counts()
            all_sigs = sigL.index.union(sigR.index)
            left_only  = int((sigL.reindex(all_sigs, fill_value=0) - sigR.reindex(all_sigs, fill_value=0)).clip(lower=0).sum())
            right_only = int((sigR.reindex(all_sigs, fill_value=0) - sigL.reindex(all_sigs, fill_value=0)).clip(lower=0).sum())
            report["row_deltas"] = {"rows_only_in_left": left_only, "rows_only_in_right": right_only}
            report["equal"] = (left_only == 0 and right_only == 0)
            return report

        Ls = _sort_all_cols(L)
        Rs = _sort_all_cols(R)
        mismatches = {}
        for c in Ls.columns:
            same = _numeric_close(Ls[c], Rs[c], opts)
            mismatches[c] = int((~same).sum())
        report["cell_mismatches"] = mismatches
        report["row_deltas"] = {"rows_only_in_left": 0, "rows_only_in_right": 0}
        report["equal"] = (sum(mismatches.values()) == 0)
        return report

    # Default multiset (exact or rounded via float_decimals)
    sigL = _row_signature_frame(L, opts).value_counts()
    sigR = _row_signature_frame(R, opts).value_counts()
    all_sigs = sigL.index.union(sigR.index)
    left_only = int((sigL.reindex(all_sigs, fill_value=0) - sigR.reindex(all_sigs, fill_value=0)).clip(lower=0).sum())
    right_only = int((sigR.reindex(all_sigs, fill_value=0) - sigL.reindex(all_sigs, fill_value=0)).clip(lower=0).sum())

    report["row_deltas"] = {"rows_only_in_left": left_only, "rows_only_in_right": right_only}
    report["equal"] = (left_only == 0 and right_only == 0)
    return report

def assert_semantic_frame_equal(df_left: pd.DataFrame, df_right: pd.DataFrame, **kwargs):
    opts = CompareOptions(**kwargs)
    rep = compare_dataframes(df_left, df_right, opts)
    if not rep["equal"]:
        lines = []
        if rep["column_mismatch"]:
            lines.append(f"Column mismatch: {rep['column_mismatch']}")
        if rep["row_deltas"]:
            lines.append(f"Row deltas: {rep['row_deltas']}")
        if rep["cell_mismatches"]:
            lines.append(f"Cell mismatches: { {k:v for k,v in rep['cell_mismatches'].items() if v} }")
        raise AssertionError("Frames are not semantically equal.\n" + "\n".join(lines))
