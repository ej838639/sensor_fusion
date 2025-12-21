from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TargetSpec:
    target_id: str
    # True measurement noise (what actually corrupts the measurement)
    sigma_meas_xy_m: float
    # Reported covariance (what the radar *claims* the uncertainty is)
    sigma_reported_xy_m: float
    # Initial truth state [x, y, vx, vy] in meters and m/s
    x0: np.ndarray


def _cv_step(x: np.ndarray, dt: float) -> np.ndarray:
    """Constant-velocity (CV) truth propagation."""
    F = np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return F @ x


def _pos_meas(x: np.ndarray) -> np.ndarray:
    """Position-only measurement model z = Hx."""
    H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float)
    return H @ x


def _cov_from_sigma_xy(sig: float) -> np.ndarray:
    """2D isotropic covariance for x/y."""
    return np.array([[sig**2, 0.0], [0.0, sig**2]], dtype=float)


def generate_radar_detections_df(
    *,
    n_scans: int = 20,
    dt: float = 1.0,
    start_time_s: float = 0.0,
    p_detect: float = 1.0,
    seed: int = 7,
) -> pd.DataFrame:
    """
    Generate synthetic radar detections for 3 targets as a Pandas DataFrame.

    Columns include:
      - time_s, scan_idx, target_id
      - z_x_m, z_y_m (measured)
      - z_true_x_m, z_true_y_m
      - x_true_m, y_true_m, vx_true_mps, vy_true_mps
      - R_true_xx, R_true_xy, R_true_yx, R_true_yy
      - R_rep_xx,  R_rep_xy,  R_rep_yx,  R_rep_yy
    """
    rng = np.random.default_rng(seed)

    targets = [
        TargetSpec(
            target_id="A_lowNoise_highCov",
            sigma_meas_xy_m=5.0,       # low measurement noise
            sigma_reported_xy_m=50.0,  # high reported covariance
            x0=np.array([0.0, 0.0, 25.0, 10.0], dtype=float),
        ),
        TargetSpec(
            target_id="B_highNoise_lowCov",
            sigma_meas_xy_m=40.0,      # high measurement noise
            sigma_reported_xy_m=5.0,   # low reported covariance
            x0=np.array([2000.0, -500.0, -15.0, 20.0], dtype=float),
        ),
        TargetSpec(
            target_id="C_lowNoise_lowCov",
            sigma_meas_xy_m=5.0,       # low measurement noise
            sigma_reported_xy_m=5.0,   # low reported covariance
            x0=np.array([-1500.0, 1200.0, 18.0, -12.0], dtype=float),
        ),
    ]

    # Initialize truth states
    truth: Dict[str, np.ndarray] = {t.target_id: t.x0.copy() for t in targets}

    rows: List[dict] = []

    for k in range(n_scans):
        t_s = start_time_s + k * dt

        for spec in targets:
            # Propagate truth (skip on k=0 so x0 is the truth at t=start_time_s)
            if k > 0:
                truth[spec.target_id] = _cv_step(truth[spec.target_id], dt)

            # Optional missed detections
            if rng.uniform() > p_detect:
                continue

            x_true = truth[spec.target_id]
            z_true = _pos_meas(x_true)

            R_true = _cov_from_sigma_xy(spec.sigma_meas_xy_m)
            R_rep = _cov_from_sigma_xy(spec.sigma_reported_xy_m)

            noise = rng.multivariate_normal(mean=np.zeros(2), cov=R_true)
            z = z_true + noise

            rows.append(
                {
                    "time_s": float(t_s),
                    "scan_idx": int(k),
                    "target_id": spec.target_id,
                    # measurement
                    "z_x_m": float(z[0]),
                    "z_y_m": float(z[1]),
                    # truth measurement
                    "z_true_x_m": float(z_true[0]),
                    "z_true_y_m": float(z_true[1]),
                    # truth state
                    "x_true_m": float(x_true[0]),
                    "y_true_m": float(x_true[1]),
                    "vx_true_mps": float(x_true[2]),
                    "vy_true_mps": float(x_true[3]),
                    # true covariance
                    "R_true_xx": float(R_true[0, 0]),
                    "R_true_xy": float(R_true[0, 1]),
                    "R_true_yx": float(R_true[1, 0]),
                    "R_true_yy": float(R_true[1, 1]),
                    # reported covariance
                    "R_rep_xx": float(R_rep[0, 0]),
                    "R_rep_xy": float(R_rep[0, 1]),
                    "R_rep_yx": float(R_rep[1, 0]),
                    "R_rep_yy": float(R_rep[1, 1]),
                }
            )

    df = pd.DataFrame(rows).sort_values(["time_s", "target_id"]).reset_index(drop=True)
    return df
