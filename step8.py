import cv2
import csv
import os
import time
import threading
import winsound
from datetime import datetime
import numpy as np
import mediapipe as mp

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH_V = [(82, 87), (13, 14), (312, 317)]
MOUTH_H = (78, 308)

POSE_IDS = [1, 152, 33, 263, 61, 291]
MODEL_3D = np.array([
    (0, 0, 0),
    (0, 330, 65),
    (-225, -170, 135),
    (225, -170, 135),
    (-150, 150, 125),
    (150, 150, 125),
], dtype=np.float64)

CALIBRATION_SECONDS = 5
CLOSED_FRAMES_LIMIT = 20
MAR_THRESHOLD = 0.5
YAWN_FRAMES_LIMIT = 15
YAW_LIMIT = 25
PITCH_LIMIT = 20
HEAD_FRAMES_LIMIT = 20
LOG_FILE = "alerts_log.csv"
WINDOW = "AI Driver Alert System"

# ---------- alarm ----------
alarm_playing = False

def _beep():
    global alarm_playing
    winsound.Beep(1200, 600)
    alarm_playing = False

def play_alarm():
    global alarm_playing
    if not alarm_playing:
        alarm_playing = True
        threading.Thread(target=_beep, daemon=True).start()

# ---------- log ----------
def log_alert(status, ear_v, mar_v, pitch, yaw):
    new_file = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer_csv = csv.writer(f)
        if new_file:
            writer_csv.writerow(["time", "alert", "EAR", "MAR", "pitch", "yaw"])
        writer_csv.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status,
                             f"{ear_v:.2f}", f"{mar_v:.2f}", f"{pitch:.1f}", f"{yaw:.1f}"])

# ---------- calculations ----------
def ear(pts):
    a = np.linalg.norm(pts[1] - pts[5])
    b = np.linalg.norm(pts[2] - pts[4])
    c = np.linalg.norm(pts[0] - pts[3])
    return (a + b) / (2.0 * c)

def mar(lm, w, h):
    p = lambda i: np.array([lm[i].x * w, lm[i].y * h])
    vertical = sum(np.linalg.norm(p(a) - p(b)) for a, b in MOUTH_V) / 3
    horizontal = np.linalg.norm(p(MOUTH_H[0]) - p(MOUTH_H[1]))
    return vertical / horizontal

def head_pose(lm, w, h):
    pts_2d = np.array([[lm[i].x * w, lm[i].y * h] for i in POSE_IDS],
                      dtype=np.float64)
    cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(MODEL_3D, pts_2d, cam, np.zeros((4, 1)))
    if not ok:
        return 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    angles = cv2.RQDecomp3x3(rmat)[0]
    return angles[0], angles[1]

# ---------- drawing helpers ----------
def draw_panel(frame, lines):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (250, 20 + 30 * len(lines)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (12, 30 + 30 * i), FONT, 0.65, color, 2)

def draw_status_bar(frame, text, color):
    hh, ww = frame.shape[:2]
    cv2.rectangle(frame, (0, hh - 50), (ww, hh), color, -1)
    cv2.putText(frame, text, (15, hh - 15), FONT, 0.8, (255, 255, 255), 2)

# ---------- setup ----------
FONT = cv2.FONT_HERSHEY_SIMPLEX
mp_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
mesh = mp_mesh.FaceMesh(refine_landmarks=True)
mesh_style = mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=0)

cap = cv2.VideoCapture(0)
writer = None
show_mesh = True

closed_frames = yawn_frames = head_frames = 0
avg_ear = mouth = pitch = yaw = raw_pitch = raw_yaw = 0.0
prev_status = "NORMAL"
alert_count = 0
prev_time = time.time()

calibrating = True
start_time = time.time()
cal_ears, cal_pitch, cal_yaw = [], [], []
ear_threshold = 0.22
base_pitch = base_yaw = 0.0

print("Keys: ESC = quit | R = record demo | M = mesh on/off")

# ---------- main loop ----------
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)          # mirror view, feels natural
    h, w = frame.shape[:2]
    result = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    face_found = bool(result.multi_face_landmarks)
    status = "NORMAL"

    now = time.time()
    fps = 1.0 / max(now - prev_time, 1e-6)
    prev_time = now

    if face_found:
        face = result.multi_face_landmarks[0]
        lm = face.landmark
        get = lambda idx: np.array([[lm[i].x * w, lm[i].y * h] for i in idx])
        avg_ear = (ear(get(LEFT_EYE)) + ear(get(RIGHT_EYE))) / 2
        mouth = mar(lm, w, h)
        raw_pitch, raw_yaw = head_pose(lm, w, h)
        if show_mesh:
            mp_draw.draw_landmarks(frame, face, mp_mesh.FACEMESH_TESSELATION,
                                   mesh_style, mesh_style)

    if calibrating:
        # ----- calibration phase -----
        elapsed = now - start_time
        if face_found:
            cal_ears.append(avg_ear)
            cal_pitch.append(raw_pitch)
            cal_yaw.append(raw_yaw)
        draw_panel(frame, [("CALIBRATING", (0, 255, 255)),
                           ("Look at the camera", (255, 255, 255)),
                           (f"{max(0, CALIBRATION_SECONDS - elapsed):.0f} s", (0, 255, 255))])
        draw_status_bar(frame, "Calibrating... keep eyes open", (120, 120, 0))
        if elapsed >= CALIBRATION_SECONDS and len(cal_ears) > 30:
            ear_threshold = float(np.mean(cal_ears)) * 0.75
            base_pitch = float(np.mean(cal_pitch))
            base_yaw = float(np.mean(cal_yaw))
            calibrating = False
            print(f"Calibrated: EAR threshold = {ear_threshold:.2f}")
    else:
        # ----- detection phase -----
        if face_found:
            pitch = raw_pitch - base_pitch
            yaw = raw_yaw - base_yaw

            closed_frames = closed_frames + 1 if avg_ear < ear_threshold else 0
            yawn_frames = yawn_frames + 1 if mouth > MAR_THRESHOLD else 0
            away = abs(yaw) > YAW_LIMIT or abs(pitch) > PITCH_LIMIT
            head_frames = head_frames + 1 if away else 0

            if closed_frames > CLOSED_FRAMES_LIMIT:
                status = "DROWSY!"
            elif yawn_frames > YAWN_FRAMES_LIMIT:
                status = "YAWNING!"
            elif head_frames > HEAD_FRAMES_LIMIT:
                status = "LOOK AT THE ROAD!"
        else:
            status = "NO FACE FOUND"

        if status != "NORMAL" and status != prev_status:
            alert_count += 1
            log_alert(status, avg_ear, mouth, pitch, yaw)
        prev_status = status

        draw_panel(frame, [
            (f"EAR:   {avg_ear:.2f}", (255, 255, 255)),
            (f"MAR:   {mouth:.2f}", (255, 255, 255)),
            (f"Pitch: {pitch:.1f}", (255, 255, 0)),
            (f"Yaw:   {yaw:.1f}", (255, 255, 0)),
            (f"Alerts: {alert_count}", (0, 200, 255)),
            (f"FPS: {fps:.0f}", (180, 180, 180)),
        ])

        if status == "NORMAL":
            draw_status_bar(frame, "DRIVER STATUS: NORMAL", (0, 150, 0))
        else:
            draw_status_bar(frame, f"DRIVER STATUS: {status}", (0, 0, 220))
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
            play_alarm()

    # ----- recording -----
    if writer is not None:
        writer.write(frame)
        cv2.circle(frame, (w - 30, 30), 9, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (w - 85, 38), FONT, 0.6, (0, 0, 255), 2)

    cv2.imshow(WINDOW, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key in (ord("m"), ord("M")):
        show_mesh = not show_mesh
    elif key in (ord("r"), ord("R")):
        if writer is None:
            writer = cv2.VideoWriter("demo.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                                     20, (w, h))
            print("Recording started -> demo.mp4")
        else:
            writer.release()
            writer = None
            print("Recording saved: demo.mp4")

if writer is not None:
    writer.release()
cap.release()
cv2.destroyAllWindows()