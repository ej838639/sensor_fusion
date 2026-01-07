from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd


# ----------------------------
# Models (CV + position meas)
# ----------------------------
def F_cv(dt: float) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def H_pos() -> np.ndarray:
    return np.array([[1.0, 0.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0, 0.0]], dtype=float)


def Q_cv(dt: float, sigma_a: float) -> np.ndarray:
    """
    Discrete white-noise acceleration model for CV state [x,y,vx,vy].
    sigma_a is acceleration std dev [m/s^2].
    """
    q = sigma_a**2
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt2 * dt2
    return q * np.array(
        [
            [dt4 / 4, 0.0,     dt3 / 2, 0.0],
            [0.0,     dt4 / 4, 0.0,     dt3 / 2],
            [dt3 / 2, 0.0,     dt2,     0.0],
            [0.0,     dt3 / 2, 0.0,     dt2],
        ],
        dtype=float,
    )


def R_from_df_row(row: pd.Series, prefix: str = "R_rep_") -> np.ndarray:
    """Build 2x2 R from scalar dataframe columns."""
    return np.array(
        [
            [row[f"{prefix}xx"], row[f"{prefix}xy"]],
            [row[f"{prefix}yx"], row[f"{prefix}yy"]],
        ],
        dtype=float,
    )


# ----------------------------
# Simple track prediction state
# ----------------------------
@dataclass
class Track:
    track_id: str
    x: np.ndarray  # state estimate [x,y,vx,vy]
    P: np.ndarray  # state covariance (4x4)


def init_tracks_from_truth(df: pd.DataFrame, *, init_P: np.ndarray) -> Dict[str, Track]:
    """
    Initialize one track per target_id using the earliest truth state in df.
    (This is a scaffold for gating; Step 3/4 will replace this with real filters.)
    """
    tracks: Dict[str, Track] = {}
    first = df.sort_values(["time_s", "target_id"]).groupby("target_id").head(1)
    for _, r in first.iterrows():
        x0 = np.array([r["x_true_m"], r["y_true_m"], r["vx_true_mps"], r["vy_true_mps"]], dtype=float)
        tracks[r["target_id"]] = Track(track_id=r["target_id"], x=x0, P=init_P.copy())
    return tracks


def predict_tracks(tracks: Dict[str, Track], *, dt: float, sigma_a: float) -> Dict[str, Track]:
    """One-step CV predict for each track."""
    F = F_cv(dt)
    Q = Q_cv(dt, sigma_a)
    out: Dict[str, Track] = {}
    for tid, trk in tracks.items():
        x_pred = F @ trk.x
        P_pred = F @ trk.P @ F.T + Q
        out[tid] = Track(track_id=tid, x=x_pred, P=P_pred)
    return out


# ----------------------------
# Step 2: Gating
# ----------------------------
def gate_detections(
    df: pd.DataFrame,
    *,
    dt: float = 1.0,
    sigma_a: float = 1.0,
    gate_d2: float = 9.21,  # ~Chi-square(2 dof) at 99% (commonly used)
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - gated_pairs_df: one row per (scan, track, detection) with d2 + pass/fail
      - tracks_pred_df: per-scan predicted track states (for debugging/plotting)

    Uses R_rep per detection in S = HPH^T + R_rep.
    """
    H = H_pos()

    # Reasonable initial covariance (tune later)
    init_P = np.diag([50.0**2, 50.0**2, 10.0**2, 10.0**2]).astype(float)

    # Init tracks from truth (scaffold)
    tracks = init_tracks_from_truth(df, init_P=init_P)

    gated_rows = []
    tracks_rows = []

    df_sorted = df.sort_values(["scan_idx", "time_s", "target_id"]).reset_index(drop=True)
    for scan_idx, df_scan in df_sorted.groupby("scan_idx"):
        # Predict tracks to this scan (skip prediction for scan 0)
        if int(scan_idx) > 0:
            tracks = predict_tracks(tracks, dt=dt, sigma_a=sigma_a)

        # Log predicted tracks (debug)
        for tid, trk in tracks.items():
            tracks_rows.append({
                "scan_idx": int(scan_idx),
                "track_id": tid,
                "x_pred_m": float(trk.x[0]),
                "y_pred_m": float(trk.x[1]),
                "vx_pred_mps": float(trk.x[2]),
                "vy_pred_mps": float(trk.x[3]),
            })

        # Evaluate gating for every track vs every detection in this scan
        for det_i, (row_idx, det) in enumerate(df_scan.iterrows()):
            z = np.array([det["z_x_m"], det["z_y_m"]], dtype=float)
            R_rep = R_from_df_row(det, prefix="R_rep_")

            for tid, trk in tracks.items():
                z_hat = H @ trk.x
                v = z - z_hat
                S = H @ trk.P @ H.T + R_rep

                # Robust inverse
                try:
                    S_inv = np.linalg.inv(S)
                except np.linalg.LinAlgError:
                    S_inv = np.linalg.pinv(S)

                d2 = float(v.T @ S_inv @ v)
                passed = d2 <= gate_d2

                gated_rows.append({
                    "scan_idx": int(scan_idx),
                    "time_s": float(det["time_s"]),
                    "track_id": tid,
                    "det_row": int(row_idx),          # index into original df_sorted
                    "det_target_id": det["target_id"], # simulator truth label (debug only)
                    "z_x_m": float(z[0]),
                    "z_y_m": float(z[1]),
                    "v_x_m": float(v[0]),
                    "v_y_m": float(v[1]),
                    "d2": d2,
                    "gate_d2": float(gate_d2),
                    "passed_gate": bool(passed),
                })

    gated_pairs_df = pd.DataFrame(gated_rows).sort_values(
        ["scan_idx", "track_id", "d2"]
    ).reset_index(drop=True)

    tracks_pred_df = pd.DataFrame(tracks_rows).sort_values(
        ["scan_idx", "track_id"]
    ).reset_index(drop=True)

    return gated_pairs_df, tracks_pred_df
