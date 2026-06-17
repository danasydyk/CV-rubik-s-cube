# Rubik's Cube CV Solver

Scans all 6 faces of a Rubik's cube using your webcam and prints the solution moves.

**Tech:** OpenCV · Lab color space · K-means clustering · kociemba solver

---

## How it works

Instead of hardcoded color thresholds, all 54 sticker colors are sampled in the **Lab color space** (perceptually uniform) and grouped into 6 clusters using **K-means**. This makes detection robust to different lighting and cube color schemes — no calibration needed.

---

## Setup

```bash
pip install opencv-python numpy kociemba
python main.py
```

---

## Scanning sequence

Hold the cube so one face points at the camera. Follow the on-screen steps:

| Step | Action |
|------|--------|
| 1 | Show any face toward the camera |
| 2 | Spin left — right side swings toward you |
| 3 | Spin left again |
| 4 | Spin left again |
| 5 | Back to start, tilt top toward camera |
| 6 | Flip 180° |

**Controls**

- `SPACE` — capture current face  
- `BACKSPACE` — redo previous face  
- `Q` — quit  
- `ENTER` (on verify screen) — solve  
- `R` (on verify screen) — rescan

---

## Output

After scanning, a verification screen shows the detected colors as a cube net. If everything looks correct, press `ENTER` and the solution prints:

```
U R F2 B' L2 D R' U2 ...
```

---

## Files

| File | Description |
|------|-------------|
| `scanner.py` | Webcam scanner — Lab sampling, K-means, verification screen |
| `solver.py` | Thin wrapper around `kociemba.solve()` |
| `main.py` | Entry point — ties scanner and solver together |
