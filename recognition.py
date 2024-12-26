import mediapipe as mp
import cv2
import numpy as np
import sounddevice as sd
import time

def generate_tone(frequency, duration=0.2, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    return 0.5 * np.sin(2 * np.pi * frequency * t)

def detect_hands():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    sample_rate = 44100

    
    finger_landmarks = {
        'thumb': (4, 3),    
        'index': (8, 6),
        'middle': (12, 10),
        'ring': (16, 14),
        'pinky': (20, 18)
    }

    # Note frequencies (A minor pentatonic)
    finger_notes = {
        'thumb': 440.0,    # A4
        'index': 523.25,   # C5
        'middle': 587.33,  # D5
        'ring': 659.25,    # E5
        'pinky': 783.99    # G5
    }

    prev_finger_states = {finger: False for finger in finger_notes}
    last_trigger_time = {finger: 0 for finger in finger_notes}
    debounce_time = 0.2

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Check each finger
                    current_time = time.time()
                    landmarks = hand_landmarks.landmark
                    
                    for finger, (tip_id, pip_id) in finger_landmarks.items():
                       
                        is_up = landmarks[tip_id].y < landmarks[pip_id].y
                        
                        
                        if (is_up and 
                            not prev_finger_states[finger] and 
                            current_time - last_trigger_time[finger] > debounce_time):
                            
                            # Play note
                            tone = generate_tone(finger_notes[finger])
                            sd.play(tone, sample_rate, blocking=False)
                            last_trigger_time[finger] = current_time
                            
                            # Visual feedback
                            cv2.putText(image, f"Playing {finger}", 
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                      1, (0, 255, 0), 2)
                        
                        prev_finger_states[finger] = is_up
                        
                        # Display finger states
                        y_pos = 60 + list(finger_landmarks.keys()).index(finger) * 30
                        status = "UP" if is_up else "DOWN"
                        cv2.putText(image, f"{finger}: {status}", 
                                  (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.7, (255, 0, 0), 2)

            cv2.imshow('Vista', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_hands()

