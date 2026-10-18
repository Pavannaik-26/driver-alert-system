import cv2
import mediapipe as mp

mp_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
mesh = mp_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mesh.process(rgb)

    if result.multi_face_landmarks:
        for face in result.multi_face_landmarks:
            mp_draw.draw_landmarks(
                frame, face, mp_mesh.FACEMESH_TESSELATION,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=0),
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1))

    cv2.imshow("Face Mesh", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()