from fusion.detections import generate_radar_detections_df
from fusion.gating import gate_detections
import numpy as np


if __name__ == "__main__":
    df = generate_radar_detections_df(n_scans=10, dt=1.0, p_detect=1.0, seed=7)

    # Show a few rows
    print(df.head(9).to_string(index=False))

    # Quick sanity check: compare sigma_true vs sigma_reported per target
    summary = (
        df.assign(
            sigma_true=lambda d: np.sqrt(d["R_true_xx"]),
            sigma_rep=lambda d: np.sqrt(d["R_rep_xx"]),
            err_x=lambda d: d["z_x_m"] - d["z_true_x_m"],
            err_y=lambda d: d["z_y_m"] - d["z_true_y_m"],
        )
        .groupby("target_id")[["sigma_true", "sigma_rep", "err_x", "err_y"]]
        .agg(["mean", "std"])
    )
    print("\nPer-target summary:\n", summary)

    # Step 2: covariance-based gating (Mahalanobis distance) using per-detection R_rep
    gated_pairs_df, tracks_pred_df = gate_detections(
        df,
        dt=1.0,
        sigma_a=1.0,
        gate_d2=9.21,  # ~Chi-square(2 dof) at 99%
    )

    # Show a quick view of gating results for one scan
    scan_to_view = 1
    view = gated_pairs_df[gated_pairs_df["scan_idx"] == scan_to_view]
    print("\nGating results (first rows) for scan", scan_to_view)
    print(view[["scan_idx", "track_id", "det_target_id", "d2", "passed_gate"]].head(30).to_string(index=False))