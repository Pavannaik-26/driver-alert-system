import cv2
import numpy as np
import mediapipe as mp

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

EAR_THRESHOLD = 0.22
CLOSED_FRAMES_LIMIT = 20

def ear(pts):
    a = np.linalg.norm(pts[1] - pts[5])
    b = np.linalg.norm(pts[2] - pts[4])
    c = np.linalg.norm(pts[0] - pts[3])
    return (a + b) / (2.0 * c)

mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
cap = cv2.VideoCapture(0)
closed_frames = 0

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
        avg = (ear(get(LEFT_EYE)) + ear(get(RIGHT_EYE))) / 2

        closed_frames = closed_frames + 1 if avg < EAR_THRESHOLD else 0
        if closed_frames > CLOSED_FRAMES_LIMIT:
            status = "DROWSY!"

        cv2.putText(frame, f"EAR: {avg:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    color = (0, 255, 0) if status == "NORMAL" else (0, 0, 255)
    cv2.putText(frame, f"DRIVER STATUS: {status}", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("AI Driver Alert System", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()