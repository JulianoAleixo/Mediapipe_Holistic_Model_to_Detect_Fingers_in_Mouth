import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Mouth landmarks
        if results.face_landmarks:
            mouth_landmarks = [results.face_landmarks.landmark[idx] for idx in [17, 146, 375, 37, 267]]
            for landmark in mouth_landmarks:
                mouth_x = int(landmark.x * frame.shape[1])
                mouth_y = int(landmark.y * frame.shape[0])
                cv2.circle(image, (mouth_x, mouth_y), 2, (0, 0, 255), -1)

                for hand_landmarks in [results.right_hand_landmarks, results.left_hand_landmarks]:
                    if hand_landmarks:
                        for idx, landmark_h in enumerate(hand_landmarks.landmark):
                            if idx == 4:  # Thumb's Landmark
                                thumb_x = int(landmark_h.x * frame.shape[1])
                                thumb_y = int(landmark_h.y * frame.shape[0])
                                cv2.circle(image, (thumb_x, thumb_y), 2, (255, 0, 0), -1)
                            elif idx == 8:  # Index finger's Landmark
                                index_finger_x = int(landmark_h.x * frame.shape[1])
                                index_finger_y = int(landmark_h.y * frame.shape[0])
                                cv2.circle(image, (index_finger_x, index_finger_y), 2, (255, 0, 0), -1)

                                if (mouth_x - 10 < thumb_x < mouth_x + 10 and
                                        mouth_y - 10 < thumb_y < mouth_y + 10 and
                                        mouth_x - 10 < index_finger_x < mouth_x + 10 and
                                        mouth_y - 10 < index_finger_y < mouth_y + 10):
                                    print("Pill in Mouth!")


        cv2.imshow('Webcam', image)

        if cv2.waitKey(10) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
