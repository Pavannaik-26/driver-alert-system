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
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["time", "alert", "EAR", "MAR", "pitch", "yaw"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status,
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

# ---------- setup ----------
mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

closed_frames = yawn_frames = head_frames = 0
avg_ear = mouth = pitch = yaw = 0.0
prev_status = "NORMAL"

calibrating = True
start_time = time.time()
cal_ears, cal_pitch, cal_yaw = [], [], []
ear_threshold = 0.22
base_pitch = base_yaw = 0.0

# ---------- main loop ----------
while True:
    ok, frame = cap.read()
    if not ok:
        break
    h, w = frame.shape[:2]
    result = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    face_found = bool(result.multi_face_landmarks)
    status = "NORMAL"

    if face_found:
        lm = result.multi_face_landmarks[0].landmark
        get = lambda idx: np.array([[lm[i].x * w, lm[i].y * h] for i in idx])
        avg_ear = (ear(get(LEFT_EYE)) + ear(get(RIGHT_EYE))) / 2
        mouth = mar(lm, w, h)
        raw_pitch, raw_yaw = head_pose(lm, w, h)

    # ----- calibration phase -----
    if calibrating:
        elapsed = time.time() - start_time
        if face_found:
            cal_ears.append(avg_ear)
            cal_pitch.append(raw_pitch)
            cal_yaw.append(raw_yaw)
        cv2.putText(frame, "CALIBRATING... look at the camera, eyes open",
                    (10, 30), font, 0.65, (0, 255, 255), 2)
        cv2.putText(frame, f"{max(0, CALIBRATION_SECONDS - elapsed):.0f}s",
                    (10, 65), font, 0.9, (0, 255, 255), 2)
        if elapsed >= CALIBRATION_SECONDS and len(cal_ears) > 30:
            ear_threshold = float(np.mean(cal_ears)) * 0.75
            base_pitch = float(np.mean(cal_pitch))
            base_yaw = float(np.mean(cal_yaw))
            calibrating = False
            print(f"Calibrated: EAR threshold = {ear_threshold:.2f}")
        cv2.imshow("AI Driver Alert System", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

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

        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"MAR: {mouth:.2f}", (10, 60), font, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Pitch: {pitch:.1f}", (10, 100), font, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Yaw: {yaw:.1f}", (10, 130), font, 0.7, (255, 255, 0), 2)
    else:
        status = "NO FACE FOUND"

    # log only when a NEW alert starts (not every frame)
    if status != "NORMAL" and status != prev_status:
        log_alert(status, avg_ear, mouth, pitch, yaw)
    prev_status = status

    if status == "NORMAL":
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)
        play_alarm()

    cv2.putText(frame, f"DRIVER STATUS: {status}", (10, h - 20), font, 0.7, color, 2)
    cv2.imshow("AI Driver Alert System", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()