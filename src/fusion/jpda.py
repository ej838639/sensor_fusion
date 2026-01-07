from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


# ============================================================
# Step 3: Data association (JPDA-style)
#
# Input: gated_pairs_df from Step 2 (gate_detections)
# Output: association probabilities beta_ij (track i <-> detection j)
#         and beta_0 (missed detection probability for track i)
#
# Notes:
# - This is a practical JPDA-style approximation using iterative
#   normalization (Sinkhorn-style) over the gated likelihood matrix.
# - Full JPDA enumerates joint association events; we'll add that later
#   if you want exactness for small N.
# ============================================================


def jpda_associate(
    gated_pairs_df: pd.DataFrame,
    *,
    p_d: float = 0.9,
    lambda_fa: float = 1e-6,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> pd.DataFrame:
    """JPDA-style soft association using gated Mahalanobis distances.

    Required columns in gated_pairs_df:
      - scan_idx, track_id, det_row, passed_gate
      - d2  (Mahalanobis distance squared)

    Returns a DataFrame with:
      - scan_idx, track_id, det_row
      - beta_ij : association probability
      - beta_0  : missed-detection probability for this track in this scan

    This is designed to feed Step 4 (PDA/JPDA-weighted Kalman/IMM update).
    """
    if not 0.0 < p_d <= 1.0:
        raise ValueError("p_d must be in (0, 1].")
    if lambda_fa <= 0.0:
        raise ValueError("lambda_fa must be > 0.")

    df = gated_pairs_df[gated_pairs_df["passed_gate"]].copy()
    if df.empty:
        return pd.DataFrame(columns=["scan_idx", "track_id", "det_row", "beta_ij", "beta_0"])

    out_rows: List[dict] = []

    for scan_idx, g in df.groupby("scan_idx"):
        tracks = sorted(g["track_id"].unique().tolist())
        dets = sorted(g["det_row"].unique().tolist())

        if not tracks or not dets:
            continue

        ti = {t: i for i, t in enumerate(tracks)}
        dj = {d: j for j, d in enumerate(dets)}

        # Likelihood ratio matrix L(i,j)
        # Use exp(-0.5 d2) as the innovation likelihood term.
        L = np.zeros((len(tracks), len(dets)), dtype=float)

        for _, r in g.iterrows():
            i = ti[r["track_id"]]
            j = dj[int(r["det_row"])]
            like = float(np.exp(-0.5 * float(r["d2"])))
            L[i, j] = max(L[i, j], (p_d * like) / lambda_fa)

        # Keep positive values to allow iterative scaling
        M = L + 1e-12

        # Iteratively enforce soft one-to-one constraints:
        #   - each track associates to <= 1 detection
        #   - each detection associates to <= 1 track
        for _ in range(max_iter):
            prev = M.copy()

            # Row normalize (tracks)
            row_sums = M.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums > 1.0, row_sums, 1.0)
            M = M / row_sums

            # Column normalize (detections)
            col_sums = M.sum(axis=0, keepdims=True)
            col_sums = np.where(col_sums > 1.0, col_sums, 1.0)
            M = M / col_sums

            if np.max(np.abs(M - prev)) < tol:
                break

        # Convert per-track row into association probabilities.
        for t in tracks:
            i = ti[t]
            row = M[i, :]
            sum_row = min(float(row.sum()), 1.0)
            beta0 = max(0.0, 1.0 - sum_row)

            for d in dets:
                j = dj[d]
                beta_ij = float(row[j])
                if beta_ij <= 0.0:
                    continue
                out_rows.append(
                    {
                        "scan_idx": int(scan_idx),
                        "track_id": t,
                        "det_row": int(d),
                        "beta_ij": beta_ij,
                        "beta_0": beta0,
                    }
                )

    assoc_df = pd.DataFrame(out_rows)
    if assoc_df.empty:
        return pd.DataFrame(columns=["scan_idx", "track_id", "det_row", "beta_ij", "beta_0"])

    return assoc_df.sort_values(
        ["scan_idx", "track_id", "beta_ij"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
