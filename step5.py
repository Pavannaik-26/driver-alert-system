import cv2
import numpy as np
import mediapipe as mp

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH_V = [(82, 87), (13, 14), (312, 317)]
MOUTH_H = (78, 308)

# Landmarks for head pose: nose tip, chin, eye corners, mouth corners
POSE_IDS = [1, 152, 33, 263, 61, 291]
MODEL_3D = np.array([
    (0, 0, 0),          # nose tip
    (0, 330, 65),       # chin
    (-225, -170, 135),  # eye corner (image left)
    (225, -170, 135),   # eye corner (image right)
    (-150, 150, 125),   # mouth corner (image left)
    (150, 150, 125),    # mouth corner (image right)
], dtype=np.float64)

EAR_THRESHOLD = 0.22
CLOSED_FRAMES_LIMIT = 20
MAR_THRESHOLD = 0.5
YAWN_FRAMES_LIMIT = 15
YAW_LIMIT = 25      # degrees: looking left/right
PITCH_LIMIT = 20    # degrees: head up/down
HEAD_FRAMES_LIMIT = 20

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
    return angles[0], angles[1]   # pitch, yaw

mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
cap = cv2.VideoCapture(0)
closed_frames = yawn_frames = head_frames = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    h, w = frame.shape[:2]
    result = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    status = "NORMAL"

    if result.multi_face_landmarks:
        lm = result.multi_face_landmarks[0].landmark
        get = lambda idx: np.array([[lm[i].x * w, lm[i].y * h] for i in idx])

        avg_ear = (ear(get(LEFT_EYE)) + ear(get(RIGHT_EYE))) / 2
        mouth = mar(lm, w, h)
        pitch, yaw = head_pose(lm, w, h)

        closed_frames = closed_frames + 1 if avg_ear < EAR_THRESHOLD else 0
        yawn_frames = yawn_frames + 1 if mouth > MAR_THRESHOLD else 0
        looking_away = abs(yaw) > YAW_LIMIT or abs(pitch) > PITCH_LIMIT
        head_frames = head_frames + 1 if looking_away else 0

        if closed_frames > CLOSED_FRAMES_LIMIT:
            status = "DROWSY!"
        elif yawn_frames > YAWN_FRAMES_LIMIT:
            status = "YAWNING!"
        elif head_frames > HEAD_FRAMES_LIMIT:
            status = "LOOK AT THE ROAD!"

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"MAR: {mouth:.2f}", (10, 60), font, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Pitch: {pitch:.1f}", (10, 100), font, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Yaw: {yaw:.1f}", (10, 130), font, 0.7, (255, 255, 0), 2)

    color = (0, 255, 0) if status == "NORMAL" else (0, 0, 255)
    cv2.putText(frame, f"DRIVER STATUS: {status}", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("AI Driver Alert System", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()