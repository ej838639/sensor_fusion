

# Sensor Fusion – Radar Tracking Pipeline (Learning Project)

This repository is a **step‑by‑step educational implementation** of a modern radar tracking and data‑fusion pipeline, built in Python. The goal is to understand *how detections become system tracks*, not to build a production‑grade tracker.

The architecture intentionally mirrors how real radar systems are structured:

```
Detections
   ↓
Prediction
   ↓
Gating
   ↓
JPDA (data association)
   ↓
Kalman / IMM filtering
   ↓
System Tracks (ASTERIX CAT062)
```

---

## Project Structure

```
src/
  app.py                 # End‑to‑end pipeline driver
  fusion/
    detections.py        # Step 1: synthetic radar detections
    gating.py            # Step 2: covariance‑based gating
    jpda.py              # Step 3: JPDA association (βᵢⱼ, β₀)
    kalman.py            # Step 4: Kalman + IMM tracking filters
    asterix.py           # Step 5: ASTERIX CAT062 system tracks
```

Each step is isolated into its own module, reflecting how real systems separate responsibilities.

---

## Step‑by‑Step Overview

### **Step 1 – Radar Detections** (`detections.py`)

Generates synthetic radar detections for multiple targets with:
- Ground‑truth state `[x, y, vx, vy]`
- Measurement noise
- Reported covariance (which may differ from true noise)

This allows experiments with:
- Over‑confident sensors
- Under‑confident sensors
- Correctly modeled sensors

Output: Pandas DataFrame of detections.

---

### **Step 2 – Gating** (`gating.py`)

Implements **Mahalanobis distance gating** using predicted track state and measurement covariance:

```
d² = (z − Hx̂⁻)ᵀ S⁻¹ (z − Hx̂⁻)
```

Purpose:
- Reduce combinatorics before data association
- Reject physically implausible associations

Output:
- Gated (track, detection) pairs
- Innovation and covariance per pair

---

### **Step 3 – Data Association (JPDA)** (`jpda.py`)

Implements a **JPDA‑style soft association**:

- Computes association probabilities **βᵢⱼ**
- Computes missed‑detection probability **β₀**

Notes:
- This step *does not update track states*
- It only answers: *"Which detections probably belong to which tracks?"*

Output: Association probabilities for each `(track, detection)`.

---

### **Step 4 – Tracking Filters** (`kalman.py`)

Implements state estimation using association probabilities:

#### Kalman Filter (CV)
- Linear Kalman filter
- Constant‑velocity motion model

#### IMM (Interacting Multiple Model)
- Two CV models with different process noise
- Handles maneuvering targets
- Produces mode probabilities and fused estimates

Includes **JPDA‑weighted (PDA) updates**, where measurements are weighted by βᵢⱼ instead of choosing a single detection.

Output: **Fused track state** `[x, y, vx, vy]` and covariance `P`.

---

### **Step 5 – System Tracks (ASTERIX CAT062)** (`asterix.py`)

Converts fused track states into **ASTERIX CAT062‑style system tracks**, including:

- `I062/010` Data Source Identifier (SAC/SIC)
- `I062/040` Track Number
- `I062/070` Time of Track Information
- `I062/080` Track Status flags
- `I062/100` Cartesian position
- `I062/185` Cartesian velocity
- `I062/500` Accuracy (position covariance)

This module focuses on:
- Correct **quantization**
- Bit‑level **field construction**
- Educational transparency

It intentionally does **not** implement full FSPEC/UAP message framing.

---

## Running the Pipeline

From the project root:

```bash
uv run python src/app.py
```

The script will:
- Generate detections
- Gate and associate them
- Track targets using KF/IMM
- Output fused tracks
- Show an example CAT062 record (debug view)

See example [output.log](docs/output.log)

---

## Design Philosophy

- **Clarity over performance**
- One responsibility per module
- Explicit math (no hidden magic)
- Easy to modify for experiments

This code is well‑suited for:
- Interview preparation (radar / sensor fusion roles)
- Learning JPDA, IMM, and tracking fundamentals
- Prototyping data‑fusion concepts

---

## Next Possible Extensions

- Track initiation and deletion logic
- Full JPDA joint‑event enumeration
- MHT (Multiple Hypothesis Tracking)
- Multi‑sensor fusion (track‑to‑track)
- Full ASTERIX CAT062 binary encoder

---

## Disclaimer

This project is **not** a certified or operational radar tracker.
It is intended for learning, experimentation, and technical discussion.