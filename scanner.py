import cv2
import numpy as np
import argparse
from collections import Counter

GRID_SIZE = 240
CELL      = GRID_SIZE // 3
SAMPLE_R  = 16
DOT_R     = 12
FACES     = ['U', 'R', 'F', 'D', 'L', 'B']

SCAN_STEPS = [
    dict(face='F', title="Step 1/6 — Show any face toward the camera"),
    dict(face='R', title="Step 2/6 — Spin left: right side swings toward you"),
    dict(face='B', title="Step 3/6 — Spin left again"),
    dict(face='L', title="Step 4/6 — Spin left again"),
    dict(face='U', title="Step 5/6 — Back to start, tilt top toward camera"),
    dict(face='D', title="Step 6/6 — Flip 180 degrees"),
]


def grid_origin(w, h):
    return (w - GRID_SIZE) // 2, (h - GRID_SIZE) // 2

def cell_centers(ox, oy):
    return [(ox + c * CELL + CELL // 2, oy + r * CELL + CELL // 2)
            for r in range(3) for c in range(3)]

def sample_sticker(frame_lab, frame_bgr, cx, cy):
    h, w = frame_lab.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), SAMPLE_R, 255, -1)
    lab = np.array(cv2.mean(frame_lab, mask=mask)[:3])
    bgr = tuple(int(v) for v in cv2.mean(frame_bgr, mask=mask)[:3])
    return lab, bgr

def draw_guide(frame, ox, oy, dot_colors=None):
    overlay = frame.copy()
    cv2.rectangle(overlay, (ox, oy), (ox + GRID_SIZE, oy + GRID_SIZE), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    for i in range(4):
        cv2.line(frame, (ox + i*CELL, oy), (ox + i*CELL, oy + GRID_SIZE), (180, 180, 180), 1)
        cv2.line(frame, (ox, oy + i*CELL), (ox + GRID_SIZE, oy + i*CELL), (180, 180, 180), 1)
    arm = 20
    for cx2, cy2 in [(ox, oy), (ox+GRID_SIZE, oy), (ox+GRID_SIZE, oy+GRID_SIZE), (ox, oy+GRID_SIZE)]:
        dx = arm if cx2 == ox else -arm
        dy = arm if cy2 == oy else -arm
        cv2.line(frame, (cx2, cy2), (cx2+dx, cy2), (0, 255, 0), 3)
        cv2.line(frame, (cx2, cy2), (cx2, cy2+dy), (0, 255, 0), 3)
    if dot_colors:
        for i, (cx2, cy2) in enumerate(cell_centers(ox, oy)):
            cv2.circle(frame, (cx2, cy2), DOT_R+3, (255, 255, 255), 2)
            cv2.circle(frame, (cx2, cy2), DOT_R, dot_colors[i], -1)

def draw_hud(frame, title, step_idx):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 50), (20, 20, 20), -1)
    cv2.putText(frame, title, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)
    cv2.rectangle(frame, (0, h - 36), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, "SPACE = capture    BACKSPACE = redo    Q = quit",
                (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (130, 130, 130), 1)
    # Progress dots
    dot_y = h - 22
    start_x = w // 2 - 55
    for i in range(6):
        color = (0, 200, 80) if i < step_idx else (60, 60, 60)
        cv2.circle(frame, (start_x + i * 22, dot_y), 6, color, -1)

def cluster_colors(all_lab):
    data = np.array(all_lab, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    _, labels, _ = cv2.kmeans(data, 6, None, criteria, attempts=10, flags=cv2.KMEANS_PP_CENTERS)
    return labels.flatten()

def build_cube_string(labels_per_face):
    cluster_to_face = {}
    for face in FACES:
        cluster_to_face[int(labels_per_face[face][4])] = face
    all_clusters = set(int(l) for f in FACES for l in labels_per_face[f])
    for uc in all_clusters - set(cluster_to_face):
        nearest = min(cluster_to_face, key=lambda c: abs(c - uc))
        cluster_to_face[uc] = cluster_to_face[nearest]
    return "".join(cluster_to_face[int(labels_per_face[f][i])] for f in FACES for i in range(9))

def validate(labels_per_face):
    counts = Counter(int(l) for f in FACES for l in labels_per_face[f])
    bad = {c: n for c, n in counts.items() if n != 9}
    return (False, str(bad)) if bad else (True, "")

def show_verification(captured_bgr, labels_per_face):
    S = 52; PAD = 6; CS = S + PAD
    W_V = 4*3*CS + 80; H_V = 3*3*CS + 100
    canvas = np.zeros((H_V, W_V, 3), dtype=np.uint8)
    canvas[:] = (45, 45, 45)
    net_pos = {'U':(1,0), 'L':(0,1), 'F':(1,1), 'R':(2,1), 'B':(3,1), 'D':(1,2)}
    ox, oy = 40, 40
    unique = list(dict.fromkeys(int(labels_per_face[f][4]) for f in FACES))
    cluster_to_num = {c: i+1 for i, c in enumerate(unique)}
    for face in FACES:
        fc, fr = net_pos[face]
        bx = ox + fc * 3 * CS
        by = oy + fr * 3 * CS
        cv2.putText(canvas, face, (bx, by - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
        for si in range(9):
            r, c = si // 3, si % 3
            x = bx + c*CS; y = by + r*CS
            cv2.rectangle(canvas, (x, y), (x+S, y+S), captured_bgr[face][si], -1)
            cv2.rectangle(canvas, (x, y), (x+S, y+S), (200, 200, 200), 3 if si == 4 else 1)
            num = str(cluster_to_num.get(int(labels_per_face[face][si]), 0))
            cv2.putText(canvas, num, (x+S//2-7, y+S//2+7), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 4)
            cv2.putText(canvas, num, (x+S//2-7, y+S//2+7), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
    cv2.putText(canvas, "ENTER = solve    R = rescan",
                (ox, H_V - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    cv2.imshow("Verify", canvas)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (13, 10):
            cv2.destroyAllWindows(); return True
        if key == ord('r'):
            cv2.destroyAllWindows(); return False

def run_scanner(debug=False):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam."); return None
    for _ in range(5):
        cap.read()

    captured = {}
    step_idx = 0
    flash_t  = 0

    while step_idx < 6:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        ox, oy = grid_origin(w, h)
        frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)

        live_labs, live_bgrs = [], []
        for cx2, cy2 in cell_centers(ox, oy):
            lab, bgr = sample_sticker(frame_lab, frame, cx2, cy2)
            live_labs.append(lab)
            live_bgrs.append(bgr)

        draw_guide(frame, ox, oy, dot_colors=live_bgrs)
        draw_hud(frame, SCAN_STEPS[step_idx]['title'], step_idx)

        if flash_t > 0:
            overlay = frame.copy()
            cv2.rectangle(overlay, (ox, oy), (ox+GRID_SIZE, oy+GRID_SIZE), (255,255,255), -1)
            cv2.addWeighted(overlay, flash_t / 10.0, frame, 1 - flash_t / 10.0, 0, frame)
            flash_t -= 1

        cv2.imshow("Scanner", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            cap.release(); cv2.destroyAllWindows(); return None
        elif key in (8, 127) and step_idx > 0:
            step_idx -= 1
            captured.pop(SCAN_STEPS[step_idx]['face'], None)
        elif key == ord(' '):
            face = SCAN_STEPS[step_idx]['face']
            captured[face] = {'lab': list(live_labs), 'bgr': list(live_bgrs)}
            flash_t = 10
            step_idx += 1
            if debug:
                for i, lab in enumerate(live_labs):
                    print(f"[{i//3},{i%3}] Lab=({lab[0]:.1f},{lab[1]:.1f},{lab[2]:.1f})")

    cap.release()
    cv2.destroyAllWindows()

    if len(captured) < 6:
        return None

    all_lab = [lab for f in FACES for lab in captured[f]['lab']]
    global_labels = cluster_colors(all_lab)
    labels_per_face = {f: global_labels[i*9:(i+1)*9] for i, f in enumerate(FACES)}

    is_valid, msg = validate(labels_per_face)
    if not is_valid:
        print(f"Invalid scan: {msg}"); return None

    if not show_verification({f: captured[f]['bgr'] for f in FACES}, labels_per_face):
        return None

    cube_string = build_cube_string(labels_per_face)
    unique = list(dict.fromkeys(int(labels_per_face[f][4]) for f in FACES))
    cluster_to_num = {c: i+1 for i, c in enumerate(unique)}
    numbered = [cluster_to_num.get(int(global_labels[i]), 0) for i in range(54)]
    return cube_string, numbered


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    run_scanner(debug=args.debug)
