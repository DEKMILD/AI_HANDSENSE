import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from PIL import ImageFont, ImageDraw, Image

MODEL_PATH = 'sign_language_model_tunedV5555.h5'
model = tf.keras.models.load_model(MODEL_PATH)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

font_path = 'THSarabun Bold Italic.ttf'
font = ImageFont.truetype(font_path, 32)

class_names = ['สวัสดี', 'ขอบคุณ', 'ขอโทษ', 'รัก', 'ฉัน', 'คุณ', 'รักสอง']
sequence = []

initial_threshold = 0.70
tracking_threshold = 0.50
current_prediction = ""
is_tracking = False
no_detection_counter = 0
RESET_DELAY = 30

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_results = hands.process(frame_rgb)
    face_results = face_mesh.process(frame_rgb)
    pose_results = pose.process(frame_rgb)

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    frame_landmarks = []

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                frame_landmarks.extend([landmark.x, landmark.y, landmark.z])

    frame_landmarks = frame_landmarks[:126]

    if len(frame_landmarks) < 126:
        frame_landmarks.extend([0.0] * (126 - len(frame_landmarks)))

    sequence.append(frame_landmarks)
    sequence = sequence[-30:]

    predicted_text_to_display = ""

    if len(sequence) == 30:
        input_data = np.expand_dims(sequence, axis=0)
        prediction = model.predict(input_data, verbose=0)[0]

        predicted_class_index = np.argmax(prediction)
        confidence = prediction[predicted_class_index]
        predicted_class_name = class_names[predicted_class_index]

        if is_tracking:
            if predicted_class_name == current_prediction and confidence > tracking_threshold:
                predicted_text_to_display = f'{current_prediction} ({confidence:.2f})'
                no_detection_counter = 0
            else:
                no_detection_counter += 1
                if no_detection_counter > RESET_DELAY:
                    is_tracking = False
                    current_prediction = ""
                else:
                    predicted_text_to_display = f'{current_prediction} (Tracking...)'

        elif confidence > initial_threshold:
            is_tracking = True
            current_prediction = predicted_class_name
            predicted_text_to_display = f'{current_prediction} ({confidence:.2f})'
            no_detection_counter = 0

    if predicted_text_to_display:
        cv2.rectangle(frame, (0, 0), (640, 40), (245, 117, 16), -1)

        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text((10, 5), predicted_text_to_display, font=font, fill=(255, 255, 255))
        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    cv2.imshow('Full Body Sign Language Detection', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
face_mesh.close()
hands.close()
pose.close()
cv2.destroyAllWindows()
