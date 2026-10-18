import cv2
import numpy as np
import mediapipe as mp

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Inner mouth points: 3 vertical pairs, plus the two corners
MOUTH_V = [(82, 87), (13, 14), (312, 317)]
MOUTH_H = (78, 308)

EAR_THRESHOLD = 0.22
CLOSED_FRAMES_LIMIT = 20
MAR_THRESHOLD = 0.5
YAWN_FRAMES_LIMIT = 15

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

mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
cap = cv2.VideoCapture(0)
closed_frames = 0
yawn_frames = 0

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

        closed_frames = closed_frames + 1 if avg_ear < EAR_THRESHOLD else 0
        yawn_frames = yawn_frames + 1 if mouth > MAR_THRESHOLD else 0

        if closed_frames > CLOSED_FRAMES_LIMIT:
            status = "DROWSY!"
        elif yawn_frames > YAWN_FRAMES_LIMIT:
            status = "YAWNING!"

        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"MAR: {mouth:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    color = (0, 255, 0) if status == "NORMAL" else (0, 0, 255)
    cv2.putText(frame, f"DRIVER STATUS: {status}", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("AI Driver Alert System", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()