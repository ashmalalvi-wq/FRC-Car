import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

base_options = mp_python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)
    if result.hand_landmarks:
        tip = result.hand_landmarks[0][8]
        print(1 - tip.x, tip.y)
        h_adj = int(tip.y * h)
        w_adj = int((1 - tip.x) * w)
        print(w - w_adj)
        cv2.circle(frame, (w - w_adj, h_adj), 10, (0, 255, 0), -1)
    else:
        print("no hand")



    
    cv2.imshow('Video Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()




