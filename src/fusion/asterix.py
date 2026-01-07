

from __future__ import annotations

"""Step 5: System tracks -> ASTERIX CAT062 fields.

This module intentionally focuses on *field construction* and *per-item byte packing*
for common CAT062 items you can drive from your tracker output.

It does NOT attempt to implement a full ASTERIX encoder with FSPEC/UAP ordering
(and therefore should not be used as a drop-in binary transmitter to an
operational ASTERIX consumer). In Step 5 you get:

- A structured, human-readable dict of CAT062 items
- Per-item packed bytes following typical CAT062 resolutions

When you're ready for a full encoder, we can add:
- FSPEC construction
- UAP ordering
- record length and multi-record message framing
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import math
import numpy as np


# ============================================================
# Quantization / packing helpers
# ============================================================

def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def to_bytes_be(x: int, nbytes: int) -> bytes:
    return int(x).to_bytes(nbytes, byteorder="big", signed=False)


def quantize_unsigned(value: float, lsb: float, nbits: int) -> int:
    """Quantize an unsigned value to an n-bit integer with given LSB."""
    if lsb <= 0:
        raise ValueError("lsb must be > 0")
    q = int(round(float(value) / float(lsb)))
    q = _clamp_int(q, 0, (1 << nbits) - 1)
    return q


def quantize_signed_twos_complement(value: float, lsb: float, nbits: int) -> int:
    """Quantize a signed value into n-bit two's complement raw bits."""
    if lsb <= 0:
        raise ValueError("lsb must be > 0")

    q = int(round(float(value) / float(lsb)))
    min_q = -(1 << (nbits - 1))
    max_q = (1 << (nbits - 1)) - 1
    q = _clamp_int(q, min_q, max_q)

    if q < 0:
        q = (1 << nbits) + q
    return q


# ============================================================
# CAT062: Track Status (I062/080)
# ============================================================

@dataclass
class TrackStatusI062_080:
    """Minimal Track Status builder.

    This models the first octet and an optional first extent octet.
    Many systems set only a subset of these bits.

    Primary octet bits (MSB->LSB):
      MON SPI MRH SRC(3) CNF FX

    First extent octet bits (MSB->LSB):
      SIM TSE TSB FPC AFF STP KOS FX
    """

    # Primary octet
    mon: int = 0  # 0 multi-sensor, 1 mono-sensor
    spi: int = 0
    mrh: int = 0
    src: int = 0  # 0..7
    cnf: int = 0  # 0 confirmed, 1 tentative

    # First extent
    sim: int = 0  # 0 actual, 1 simulated
    tse: int = 0
    tsb: int = 0
    fpc: int = 0
    aff: int = 0
    stp: int = 0
    kos: int = 0

    include_first_extent: bool = True

    def pack(self) -> bytes:
        fx = 1 if self.include_first_extent else 0
        src3 = _clamp_int(int(self.src), 0, 7)

        b0 = 0
        b0 |= (_clamp_int(int(self.mon), 0, 1) & 1) << 7
        b0 |= (_clamp_int(int(self.spi), 0, 1) & 1) << 6
        b0 |= (_clamp_int(int(self.mrh), 0, 1) & 1) << 5
        b0 |= (src3 & 0b111) << 2
        b0 |= (_clamp_int(int(self.cnf), 0, 1) & 1) << 1
        b0 |= (fx & 1)

        if not self.include_first_extent:
            return bytes([b0])

        b1 = 0
        b1 |= (_clamp_int(int(self.sim), 0, 1) & 1) << 7
        b1 |= (_clamp_int(int(self.tse), 0, 1) & 1) << 6
        b1 |= (_clamp_int(int(self.tsb), 0, 1) & 1) << 5
        b1 |= (_clamp_int(int(self.fpc), 0, 1) & 1) << 4
        b1 |= (_clamp_int(int(self.aff), 0, 1) & 1) << 3
        b1 |= (_clamp_int(int(self.stp), 0, 1) & 1) << 2
        b1 |= (_clamp_int(int(self.kos), 0, 1) & 1) << 1
        b1 |= 0  # FX=0 (no further extents)

        return bytes([b0, b1])


# ============================================================
# CAT062 field construction
# ============================================================

def build_cat062_items(
    *,
    sac: int,
    sic: int,
    track_number: int,
    time_s: float,
    x_m: float,
    y_m: float,
    vx_mps: float,
    vy_mps: float,
    P_xyvv: np.ndarray,
    status: Optional[TrackStatusI062_080] = None,
    include_i062_185: bool = True,
    include_i062_500: bool = True,
) -> Dict[str, Any]:
    """Build a dict of common CAT062 items plus per-item packed bytes.

    Args:
      sac/sic: Data Source Identifier (I062/010)
      track_number: Track Number (I062/040)
      time_s: Time of Track Information (I062/070). In real CAT062 this is seconds since UTC midnight.
      x_m,y_m: Cartesian position estimate (I062/100)
      vx_mps,vy_mps: Cartesian velocity estimate (I062/185)
      P_xyvv: 4x4 covariance for [x,y,vx,vy]
      status: Track Status (I062/080)

    Notes on resolutions (typical CAT062):
      - I062/070: 1/128 s (24-bit)
      - I062/100: 0.5 m (24-bit signed)
      - I062/185: 0.25 m/s (16-bit signed)
      - I062/500: includes position std devs and an XY covariance component encoding

    Returns:
      dict with keys like "I062/010", ... each containing a structured sub-dict.
    """

    if P_xyvv.shape != (4, 4):
        raise ValueError("P_xyvv must be 4x4 covariance for [x,y,vx,vy].")

    if status is None:
        status = TrackStatusI062_080(mon=0, cnf=0, include_first_extent=True)

    # I062/010
    sac = _clamp_int(int(sac), 0, 255)
    sic = _clamp_int(int(sic), 0, 255)
    i062_010_raw = bytes([sac, sic])

    # I062/040
    track_number = _clamp_int(int(track_number), 0, 65535)
    i062_040_raw = to_bytes_be(track_number, 2)

    # I062/070 time, 24-bit unsigned with LSB = 1/128 s
    t_q = quantize_unsigned(time_s, lsb=1.0 / 128.0, nbits=24)
    i062_070_raw = to_bytes_be(t_q, 3)

    # I062/080
    i062_080_raw = status.pack()

    # I062/100 position, 24-bit signed two's complement, LSB=0.5 m
    x_q = quantize_signed_twos_complement(x_m, lsb=0.5, nbits=24)
    y_q = quantize_signed_twos_complement(y_m, lsb=0.5, nbits=24)
    i062_100_raw = to_bytes_be(x_q, 3) + to_bytes_be(y_q, 3)

    items: Dict[str, Any] = {
        "cat": 62,
        "I062/010": {"SAC": sac, "SIC": sic, "raw": i062_010_raw},
        "I062/040": {"track_number": track_number, "raw": i062_040_raw},
        "I062/070": {"time_s": float(time_s), "time_q_1_128s": t_q, "raw": i062_070_raw},
        "I062/080": {"status": status, "raw": i062_080_raw},
        "I062/100": {
            "x_m": float(x_m),
            "y_m": float(y_m),
            "x_q_0_5m": x_q,
            "y_q_0_5m": y_q,
            "raw": i062_100_raw,
        },
    }

    if include_i062_185:
        vx_q = quantize_signed_twos_complement(vx_mps, lsb=0.25, nbits=16)
        vy_q = quantize_signed_twos_complement(vy_mps, lsb=0.25, nbits=16)
        i062_185_raw = to_bytes_be(vx_q, 2) + to_bytes_be(vy_q, 2)
        items["I062/185"] = {
            "vx_mps": float(vx_mps),
            "vy_mps": float(vy_mps),
            "vx_q_0_25mps": vx_q,
            "vy_q_0_25mps": vy_q,
            "raw": i062_185_raw,
        }

    if include_i062_500:
        # APC: std devs of x,y quantized with 0.5 m LSB (16-bit unsigned)
        sigma_x = float(math.sqrt(max(float(P_xyvv[0, 0]), 0.0)))
        sigma_y = float(math.sqrt(max(float(P_xyvv[1, 1]), 0.0)))
        apc_x_q = quantize_unsigned(sigma_x, lsb=0.5, nbits=16)
        apc_y_q = quantize_unsigned(sigma_y, lsb=0.5, nbits=16)

        # COV: special "XY covariance component" encoding:
        # component = sign(CovXY) * sqrt(|CovXY|), then quantize with 0.5 m LSB as 16-bit signed.
        cov_xy = float(P_xyvv[0, 1])
        cov_component_m = math.copysign(math.sqrt(abs(cov_xy)), cov_xy) if cov_xy != 0.0 else 0.0
        cov_q = quantize_signed_twos_complement(cov_component_m, lsb=0.5, nbits=16)

        # Primary subfield presence bitmap (2 octets). We set APC + COV present.
        primary = 0
        primary |= 1 << 15  # APC present
        primary |= 1 << 14  # COV present
        i062_500_primary = to_bytes_be(primary, 2)
        i062_500_apc = to_bytes_be(apc_x_q, 2) + to_bytes_be(apc_y_q, 2)
        i062_500_cov = to_bytes_be(cov_q, 2)
        i062_500_raw = i062_500_primary + i062_500_apc + i062_500_cov

        items["I062/500"] = {
            "sigma_x_m": sigma_x,
            "sigma_y_m": sigma_y,
            "cov_xy_m2": cov_xy,
            "cov_component_m": cov_component_m,
            "APC_x_q_0_5m": apc_x_q,
            "APC_y_q_0_5m": apc_y_q,
            "COV_q_0_5m": cov_q,
            "raw": i062_500_raw,
        }

    return items


# ============================================================
# "System track" record wrapper
# ============================================================

def system_track_to_cat062(
    *,
    sac: int,
    sic: int,
    track_number: int,
    time_s: float,
    state_xyvv: np.ndarray,
    cov_xyvv: np.ndarray,
    status: Optional[TrackStatusI062_080] = None,
) -> Dict[str, Any]:
    """Convenience wrapper: build CAT062 items from a fused track state."""
    state_xyvv = np.asarray(state_xyvv, dtype=float).reshape(4,)
    cov_xyvv = np.asarray(cov_xyvv, dtype=float).reshape(4, 4)

    return build_cat062_items(
        sac=sac,
        sic=sic,
        track_number=track_number,
        time_s=time_s,
        x_m=float(state_xyvv[0]),
        y_m=float(state_xyvv[1]),
        vx_mps=float(state_xyvv[2]),
        vy_mps=float(state_xyvv[3]),
        P_xyvv=cov_xyvv,
        status=status,
        include_i062_185=True,
        include_i062_500=True,
    )


# ============================================================
# Optional: simple (non-compliant) serializer for debugging
# ============================================================

def serialize_items_debug(cat062_items: Dict[str, Any]) -> bytes:
    """Serialize items into a simple debug-friendly byte stream.

    This is NOT a standards-compliant ASTERIX record (no FSPEC/UAP).

    Format:
      - 1 byte: Category (62)
      - Repeated TLVs:
          2 bytes: item ID as ASCII-ish (e.g., b"10", b"40"...) is not used.

    Instead we use a compact custom TLV:
      - 2 bytes: hash-like item code (not standard)
      - 2 bytes: length
      - N bytes: raw payload

    Use this only for internal debugging (e.g., to verify quantization).
    """

    def code_for(item_name: str) -> int:
        # Convert "I062/100" -> 100
        try:
            return int(item_name.split("/")[1])
        except Exception:
            return 0

    out = bytearray()
    out.append(int(cat062_items.get("cat", 62)) & 0xFF)

    for k, v in cat062_items.items():
        if not isinstance(k, str) or not k.startswith("I062/"):
            continue
        raw = v.get("raw", b"")
        if not isinstance(raw, (bytes, bytearray)):
            continue

        code = _clamp_int(code_for(k), 0, 65535)
        out += to_bytes_be(code, 2)
        out += to_bytes_be(len(raw), 2)
        out += raw

    return bytes(out)