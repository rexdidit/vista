import mediapipe as mp
import cv2
import numpy as np

def detect_hands():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # Capturing video from webcam
    cap = cv2.VideoCapture()
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip image
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            # Performance
            image.flags.writeable = False
            results = hands.process(image)

            # Draw
            image.flags.writeable = True
            
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Display 
            cv2.imshow('Vista', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    # Release 
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_hands()

