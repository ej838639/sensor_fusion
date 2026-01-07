from fusion.detections import generate_radar_detections_df
from fusion.gating import gate_detections
from fusion.jpda import jpda_associate
from fusion.kalman import KFState, IMMState, KalmanCV, IMMCV, TrackFilter
from fusion.asterix import TrackStatusI062_080, serialize_items_debug, system_track_to_cat062

import numpy as np
import pandas as pd


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

    # Step 3: JPDA-style soft association
    assoc_df = jpda_associate(gated_pairs_df, p_d=0.9, lambda_fa=1e-6)

    assoc_view = assoc_df[assoc_df["scan_idx"] == scan_to_view]
    print("\nJPDA association (top rows) for scan", scan_to_view)
    print(assoc_view.head(30).to_string(index=False))


    # ============================================================
    # Step 4: Tracking filters (KF CV and IMM for maneuvers)
    #   - Predict each track
    #   - Use JPDA association probabilities (betas)
    #   - Apply JPDA-weighted (PDA) update per track
    # ============================================================

    def R_rep_from_row(r: pd.Series) -> np.ndarray:
        return np.array(
            [[r["R_rep_xx"], r["R_rep_xy"]], [r["R_rep_yx"], r["R_rep_yy"]]],
            dtype=float,
        )

    # Initialize one filter per target_id using truth at first appearance (scaffold)
    init_P = np.diag([50.0**2, 50.0**2, 10.0**2, 10.0**2]).astype(float)

    filters: dict[str, TrackFilter] = {}
    df0 = df.sort_values(["scan_idx", "target_id"]).groupby("target_id").head(1)

    for _, r0 in df0.iterrows():
        tid = str(r0["target_id"])
        x0 = np.array([r0["x_true_m"], r0["y_true_m"], r0["vx_true_mps"], r0["vy_true_mps"]], dtype=float)

        # Demo: use IMM for one target and KF for the others.
        # (Change this selection logic as you like.)
        use_imm = ("B" in tid)

        if use_imm:
            imm = IMMCV(dt=1.0, sigma_a_low=1.0, sigma_a_high=8.0)
            mu0 = np.array([0.9, 0.1], dtype=float)
            filters[tid] = TrackFilter(
                track_id=tid,
                use_imm=True,
                imm=imm,
                state_imm=IMMState(
                    models=[
                        KFState(x=x0.copy(), P=init_P.copy()),
                        KFState(x=x0.copy(), P=init_P.copy()),
                    ],
                    mu=mu0,
                ),
            )
        else:
            kf = KalmanCV(dt=1.0, sigma_a=1.0)
            filters[tid] = TrackFilter(
                track_id=tid,
                use_imm=False,
                kf=kf,
                state_kf=KFState(x=x0.copy(), P=init_P.copy()),
            )

    # Stable Track Number assignment (CAT062 I062/040)
    track_number_map: dict[str, int] = {tid: i + 1 for i, tid in enumerate(sorted(filters.keys()))}

    # Step 5 output collection (CAT062 items + debug bytes)
    cat062_rows: list[dict] = []

    # Run filter updates per scan using JPDA association probabilities.
    # IMPORTANT: df index must remain intact because assoc_df.det_row refers to df.loc[det_row].
    track_rows: list[dict] = []

    for scan_idx in sorted(df["scan_idx"].unique()):
        # Predict to this scan (skip predict at scan 0)
        if int(scan_idx) > 0:
            for tf in filters.values():
                tf.predict()

        # Update each track from JPDA weights
        for tid, tf in filters.items():
            a = assoc_df[(assoc_df["scan_idx"] == scan_idx) & (assoc_df["track_id"] == tid)]

            if a.empty:
                # Missed detection
                tf.update_jpda(zs=[], Rs=[], betas=[], beta0=1.0)
            else:
                beta0 = float(a["beta_0"].iloc[0])
                zs: list[np.ndarray] = []
                Rs: list[np.ndarray] = []
                betas: list[float] = []

                for _, ar in a.iterrows():
                    det_idx = int(ar["det_row"])
                    dr = df.loc[det_idx]
                    zs.append(np.array([dr["z_x_m"], dr["z_y_m"]], dtype=float))
                    Rs.append(R_rep_from_row(dr))
                    betas.append(float(ar["beta_ij"]))

                tf.update_jpda(zs=zs, Rs=Rs, betas=betas, beta0=beta0)

            s = tf.fused_state()

            track_rows.append(
                {
                    "scan_idx": int(scan_idx),
                    "track_id": tid,
                    "x_hat_m": float(s.x[0]),
                    "y_hat_m": float(s.x[1]),
                    "vx_hat_mps": float(s.x[2]),
                    "vy_hat_mps": float(s.x[3]),
                    "sigma_x_m": float(np.sqrt(max(s.P[0, 0], 0.0))),
                    "sigma_y_m": float(np.sqrt(max(s.P[1, 1], 0.0))),
                }
            )

            # ============================================================
            # Step 5: System tracks -> ASTERIX CAT062 fields (debug-friendly)
            # ============================================================
            # Track Status: mark first 2 scans as tentative (CNF=1), then confirmed (CNF=0)
            status = TrackStatusI062_080(
                mon=0,  # multi-sensor (set to 1 if you want mono-sensor)
                cnf=1 if int(scan_idx) < 2 else 0,
                sim=0,
                include_first_extent=True,
            )

            cat062_items = system_track_to_cat062(
                sac=1,
                sic=7,
                track_number=int(track_number_map[tid]),
                time_s=float(scan_idx),
                state_xyvv=s.x,
                cov_xyvv=s.P,
                status=status,
            )

            cat062_debug_bytes = serialize_items_debug(cat062_items)
            cat062_rows.append(
                {
                    "scan_idx": int(scan_idx),
                    "track_id": tid,
                    "track_number": int(track_number_map[tid]),
                    "cat062_items": cat062_items,
                    "cat062_debug_hex": cat062_debug_bytes.hex(),
                }
            )

    tracks_df = pd.DataFrame(track_rows).sort_values(["scan_idx", "track_id"]).reset_index(drop=True)
    print("\nStep 4 track estimates (first rows):")
    print(tracks_df.head(20).to_string(index=False))

    # ----------------------------
    # Step 5: Show one example CAT062 record
    # ----------------------------
    cat062_df = pd.DataFrame(cat062_rows)

    example = cat062_df[(cat062_df["scan_idx"] == scan_to_view)].head(1)
    if not example.empty:
        row = example.iloc[0]
        items = row["cat062_items"]
        print("\nStep 5 example CAT062 (debug) for scan", int(row["scan_idx"]), "track", row["track_id"], "(#", int(row["track_number"]), ")")
        print("  I062/010:", items.get("I062/010"))
        print("  I062/040:", items.get("I062/040"))
        print("  I062/070:", items.get("I062/070"))
        print("  I062/080 raw:", items.get("I062/080", {}).get("raw", b"").hex())
        print("  I062/100:", items.get("I062/100"))
        print("  I062/185:", items.get("I062/185"))
        print("  I062/500:", items.get("I062/500"))
        print("  debug_hex:", row["cat062_debug_hex"][:120] + ("..." if len(row["cat062_debug_hex"]) > 120 else ""))
    else:
        print("\nStep 5: No CAT062 records found to display.")