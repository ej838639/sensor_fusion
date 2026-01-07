from __future__ import annotations

import numpy as np
import pandas as pd

# ----------------------------
# Step 3: Data association (JPDA-style)
# ----------------------------
def jpda_associate(
    gated_pairs_df: pd.DataFrame,
    *,
    p_d: float = 0.9,
    lambda_fa: float = 1e-6,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> pd.DataFrame:
    """
    JPDA-style soft association.

    Inputs:
      gated_pairs_df: output of gate_detections(), containing only
        - scan_idx
        - track_id
        - det_row
        - passed_gate
        - d2

    Outputs:
      DataFrame with:
        - beta_ij : association probability track i ↔ detection j
        - beta_0  : missed-detection probability for track i

    Notes:
      This is a practical approximation to full JPDA using iterative
      normalization (Sinkhorn-style). It is widely used in real systems.
    """

    # Keep only gated pairs
    df = gated_pairs_df[gated_pairs_df["passed_gate"]].copy()
    if df.empty:
        return pd.DataFrame()

    rows = []

    for scan_idx, g in df.groupby("scan_idx"):
        tracks = sorted(g["track_id"].unique())
        dets = sorted(g["det_row"].unique())

        if not tracks or not dets:
            continue

        ti = {t: i for i, t in enumerate(tracks)}
        dj = {d: j for j, d in enumerate(dets)}

        # Likelihood matrix L(i,j)
        # Use exp(-0.5 d2) as innovation likelihood
        L = np.zeros((len(tracks), len(dets)), dtype=float)

        for _, r in g.iterrows():
            i = ti[r["track_id"]]
            j = dj[int(r["det_row"])]

            # JPDA likelihood ratio
            L[i, j] = max(
                L[i, j],
                (p_d * np.exp(-0.5 * r["d2"])) / lambda_fa
            )

        # Add epsilon to avoid zeros
        M = L + 1e-12

        # Iterative row/column normalization
        for _ in range(max_iter):
            prev = M.copy()

            # Each track can associate to at most one detection
            row_sums = M.sum(axis=1, keepdims=True)
            row_sums[row_sums < 1.0] = 1.0
            M /= row_sums

            # Each detection can associate to at most one track
            col_sums = M.sum(axis=0, keepdims=True)
            col_sums[col_sums < 1.0] = 1.0
            M /= col_sums

            if np.max(np.abs(M - prev)) < tol:
                break

        # Convert matrix into association probabilities
        for t in tracks:
            i = ti[t]
            row = M[i, :]
            sum_row = min(float(row.sum()), 1.0)
            beta_0 = 1.0 - sum_row  # missed detection prob

            for d in dets:
                j = dj[d]
                beta_ij = float(row[j])
                if beta_ij > 0.0:
                    rows.append({
                        "scan_idx": int(scan_idx),
                        "track_id": t,
                        "det_row": int(d),
                        "beta_ij": beta_ij,
                        "beta_0": beta_0,
                    })

    return (
        pd.DataFrame(rows)
        .sort_values(["scan_idx", "track_id", "beta_ij"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
