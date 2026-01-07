from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ============================================================
# Step 4: Tracking Filters
#   - Linear Kalman Filter for CV (constant velocity)
#   - IMM (Interacting Multiple Model) for maneuvers (2-model IMM)
#
# State: x = [x, y, vx, vy]^T
# Measurement: z = [x, y]^T
# ============================================================


def F_cv(dt: float) -> np.ndarray:
    """CV state transition for [x,y,vx,vy]."""
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
    """Position-only measurement matrix."""
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=float,
    )


def Q_white_accel(dt: float, sigma_a: float) -> np.ndarray:
    """Discrete white-noise acceleration process noise for CV model."""
    q = float(sigma_a) ** 2
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt2 * dt2
    return q * np.array(
        [
            [dt4 / 4, 0.0, dt3 / 2, 0.0],
            [0.0, dt4 / 4, 0.0, dt3 / 2],
            [dt3 / 2, 0.0, dt2, 0.0],
            [0.0, dt3 / 2, 0.0, dt2],
        ],
        dtype=float,
    )


def gaussian_likelihood(v: np.ndarray, S: np.ndarray) -> float:
    """Gaussian likelihood N(v; 0, S) for v in R^m."""
    m = int(v.shape[0])
    try:
        det_S = float(np.linalg.det(S))
        inv_S = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_reg = S + 1e-9 * np.eye(m)
        det_S = float(np.linalg.det(S_reg))
        inv_S = np.linalg.pinv(S_reg)

    det_S = max(det_S, 1e-30)
    d2 = float(v.T @ inv_S @ v)
    norm = float(np.sqrt(((2.0 * np.pi) ** m) * det_S))
    return float(np.exp(-0.5 * d2) / norm)


@dataclass
class KFState:
    """Kalman state container."""

    x: np.ndarray  # (4,)
    P: np.ndarray  # (4,4)


class KalmanCV:
    """Linear Kalman filter for constant velocity (CV) model."""

    def __init__(self, *, dt: float, sigma_a: float):
        self.dt = float(dt)
        self.sigma_a = float(sigma_a)
        self.F = F_cv(self.dt)
        self.H = H_pos()
        self.Q = Q_white_accel(self.dt, self.sigma_a)

    def predict(self, s: KFState) -> KFState:
        x = self.F @ s.x
        P = self.F @ s.P @ self.F.T + self.Q
        return KFState(x=x, P=P)

    def innovation(self, s_pred: KFState, z: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        z_hat = self.H @ s_pred.x
        v = z - z_hat
        S = self.H @ s_pred.P @ self.H.T + R
        return v, S

    def update(self, s_pred: KFState, z: np.ndarray, R: np.ndarray) -> Tuple[KFState, float]:
        """Standard KF measurement update. Returns (updated_state, likelihood)."""
        v, S = self.innovation(s_pred, z, R)

        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        K = s_pred.P @ self.H.T @ S_inv
        x_upd = s_pred.x + K @ v
        P_upd = (np.eye(4) - K @ self.H) @ s_pred.P

        lh = gaussian_likelihood(v, S)
        return KFState(x=x_upd, P=P_upd), lh

    def pda_update(
        self,
        s_pred: KFState,
        zs: List[np.ndarray],
        Rs: List[np.ndarray],
        betas: List[float],
        beta0: float,
    ) -> KFState:
        """\
        PDA/JPDA-weighted update (moment matching) for one track.

        Inputs:
          - s_pred: predicted state
          - zs, Rs: list of candidate measurements and their covariances
          - betas: association probabilities for each measurement
          - beta0: missed-detection probability
        """
        if len(zs) == 0 or float(sum(betas)) <= 0.0:
            return s_pred

        H = self.H

        # Effective R for gain (beta-weighted)
        R_eff = np.zeros((2, 2), dtype=float)
        for b, R in zip(betas, Rs):
            R_eff += float(b) * R
        if np.trace(R_eff) <= 0.0:
            R_eff = Rs[0]

        S_eff = H @ s_pred.P @ H.T + R_eff
        try:
            S_eff_inv = np.linalg.inv(S_eff)
        except np.linalg.LinAlgError:
            S_eff_inv = np.linalg.pinv(S_eff)
        K = s_pred.P @ H.T @ S_eff_inv

        # Innovations per measurement
        v_list: List[np.ndarray] = []
        for z, R in zip(zs, Rs):
            v, _ = self.innovation(s_pred, z, R)
            v_list.append(v)

        # Weighted innovation mean
        v_bar = np.zeros((2,), dtype=float)
        for b, v in zip(betas, v_list):
            v_bar += float(b) * v

        # Innovation spread
        V = np.zeros((2, 2), dtype=float)
        for b, v in zip(betas, v_list):
            v = v.reshape(2, 1)
            V += float(b) * (v @ v.T)
        spread = V - v_bar.reshape(2, 1) @ v_bar.reshape(1, 2)

        # State update
        x_upd = s_pred.x + K @ v_bar

        # Covariance update (PDA)
        P_pred = s_pred.P
        P_meas = K @ S_eff @ K.T
        P_upd = P_pred - (1.0 - float(beta0)) * P_meas + K @ spread @ K.T

        return KFState(x=x_upd, P=P_upd)


# ----------------------------
# IMM (two-model) filter
# ----------------------------
@dataclass
class IMMState:
    """IMM container: per-model states + mode probabilities."""

    models: List[KFState]
    mu: np.ndarray  # (2,)


class IMMCV:
    """\
    Two-model IMM for maneuvers using the same CV state but different process noise.

    Model 0: low maneuver (small sigma_a)
    Model 1: maneuver (large sigma_a)
    """

    def __init__(
        self,
        *,
        dt: float,
        sigma_a_low: float = 1.0,
        sigma_a_high: float = 8.0,
        PI: Optional[np.ndarray] = None,
    ):
        self.dt = float(dt)
        self.filters = [
            KalmanCV(dt=self.dt, sigma_a=float(sigma_a_low)),
            KalmanCV(dt=self.dt, sigma_a=float(sigma_a_high)),
        ]

        # Mode transition matrix PI[i,j] = P(mode_j | mode_i)
        if PI is None:
            self.PI = np.array([[0.95, 0.05], [0.05, 0.95]], dtype=float)
        else:
            PI = np.asarray(PI, dtype=float)
            if PI.shape != (2, 2):
                raise ValueError("PI must be 2x2 for two-model IMM")
            self.PI = PI

    def _mixing_probabilities(self, mu: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # c_j = sum_i PI[i,j] * mu_i
        c = self.PI.T @ mu
        c = np.maximum(c, 1e-12)
        mu_ij = (self.PI * mu.reshape(-1, 1)) / c.reshape(1, -1)
        return mu_ij, c

    def _mix_states(self, imm: IMMState, mu_ij: np.ndarray) -> List[KFState]:
        mixed: List[KFState] = []
        for j in range(2):
            x0 = np.zeros((4,), dtype=float)
            for i in range(2):
                x0 += float(mu_ij[i, j]) * imm.models[i].x

            P0 = np.zeros((4, 4), dtype=float)
            for i in range(2):
                dx = (imm.models[i].x - x0).reshape(4, 1)
                P0 += float(mu_ij[i, j]) * (imm.models[i].P + dx @ dx.T)

            mixed.append(KFState(x=x0, P=P0))
        return mixed

    def predict(self, imm: IMMState) -> IMMState:
        mu = np.asarray(imm.mu, dtype=float).reshape(2,)
        mu_ij, c = self._mixing_probabilities(mu)
        mixed = self._mix_states(imm, mu_ij)

        preds = [kf.predict(s) for kf, s in zip(self.filters, mixed)]
        mu_pred = c / float(np.sum(c))
        return IMMState(models=preds, mu=mu_pred)

    def update(self, imm_pred: IMMState, z: np.ndarray, R: np.ndarray) -> IMMState:
        likes = np.zeros((2,), dtype=float)
        upds: List[KFState] = []

        for j, (kf, s_pred) in enumerate(zip(self.filters, imm_pred.models)):
            s_upd, lh = kf.update(s_pred, z, R)
            upds.append(s_upd)
            likes[j] = max(float(lh), 1e-30)

        mu = imm_pred.mu * likes
        mu = mu / float(np.sum(mu))
        return IMMState(models=upds, mu=mu)

    def pda_update(
        self,
        imm_pred: IMMState,
        zs: List[np.ndarray],
        Rs: List[np.ndarray],
        betas: List[float],
        beta0: float,
    ) -> IMMState:
        upds: List[KFState] = []
        likes = np.zeros((2,), dtype=float)

        for j, (kf, s_pred) in enumerate(zip(self.filters, imm_pred.models)):
            s_upd = kf.pda_update(s_pred, zs, Rs, betas, beta0)
            upds.append(s_upd)

            if len(zs) == 0 or float(sum(betas)) <= 0.0:
                likes[j] = 1.0
            else:
                idx = int(np.argmax(np.asarray(betas)))
                v, S = kf.innovation(s_pred, zs[idx], Rs[idx])
                likes[j] = max(gaussian_likelihood(v, S), 1e-30)

        mu = imm_pred.mu * likes
        mu = mu / float(np.sum(mu))
        return IMMState(models=upds, mu=mu)

    def fuse(self, imm: IMMState) -> KFState:
        mu = np.asarray(imm.mu, dtype=float).reshape(2,)
        mu = mu / float(np.sum(mu))

        x = mu[0] * imm.models[0].x + mu[1] * imm.models[1].x

        P = np.zeros((4, 4), dtype=float)
        for j in range(2):
            dx = (imm.models[j].x - x).reshape(4, 1)
            P += float(mu[j]) * (imm.models[j].P + dx @ dx.T)

        return KFState(x=x, P=P)


# ----------------------------
# Convenience wrapper for per-track filtering
# ----------------------------
@dataclass
class TrackFilter:
    track_id: str
    use_imm: bool
    kf: Optional[KalmanCV] = None
    imm: Optional[IMMCV] = None
    state_kf: Optional[KFState] = None
    state_imm: Optional[IMMState] = None

    def predict(self) -> None:
        if self.use_imm:
            if self.imm is None or self.state_imm is None:
                raise ValueError("IMM not initialized")
            self.state_imm = self.imm.predict(self.state_imm)
        else:
            if self.kf is None or self.state_kf is None:
                raise ValueError("KF not initialized")
            self.state_kf = self.kf.predict(self.state_kf)

    def update_single(self, z: np.ndarray, R: np.ndarray) -> None:
        if self.use_imm:
            if self.imm is None or self.state_imm is None:
                raise ValueError("IMM not initialized")
            self.state_imm = self.imm.update(self.state_imm, z, R)
        else:
            if self.kf is None or self.state_kf is None:
                raise ValueError("KF not initialized")
            self.state_kf, _ = self.kf.update(self.state_kf, z, R)

    def update_jpda(self, zs: List[np.ndarray], Rs: List[np.ndarray], betas: List[float], beta0: float) -> None:
        if self.use_imm:
            if self.imm is None or self.state_imm is None:
                raise ValueError("IMM not initialized")
            self.state_imm = self.imm.pda_update(self.state_imm, zs, Rs, betas, beta0)
        else:
            if self.kf is None or self.state_kf is None:
                raise ValueError("KF not initialized")
            self.state_kf = self.kf.pda_update(self.state_kf, zs, Rs, betas, beta0)

    def fused_state(self) -> KFState:
        if self.use_imm:
            if self.imm is None or self.state_imm is None:
                raise ValueError("IMM not initialized")
            return self.imm.fuse(self.state_imm)
        if self.state_kf is None:
            raise ValueError("KF not initialized")
        return self.state_kf
