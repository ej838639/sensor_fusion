from fusion.detections import generate_radar_detections_df
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