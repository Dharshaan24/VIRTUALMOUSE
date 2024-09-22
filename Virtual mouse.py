import cv2
import mediapipe as mp
import pyautogui
import math

# Initialize MediaPipe Hands and Drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe Hands model
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize webcam capture
cap = cv2.VideoCapture(0)

# Define screen size
screen_width, screen_height = pyautogui.size()

# Create a variable to track the state of the click
clicking = False

while True:
    # Read the frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get hand landmarks
    results = hands.process(rgb_frame)

    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get the position of the index finger tip (landmark 8)
            h, w, c = frame.shape
            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)

            # Convert coordinates to screen coordinates
            screen_x = int(screen_width * x / w)
            screen_y = int(screen_height * y / h)

            # Move the mouse cursor
            pyautogui.moveTo(screen_x, screen_y)

            # Check distance between index and thumb tips for click detection
            thumb_tip_x = int(hand_landmarks.landmark[4].x * w)
            thumb_tip_y = int(hand_landmarks.landmark[4].y * h)
            distance = math.hypot(x - thumb_tip_x, y - thumb_tip_y)
            
            # Set a threshold for clicking
            if distance < 30:
                if not clicking:
                    pyautogui.click()
                    clicking = True
            else:
                clicking = False

    # Display the frame
    cv2.imshow('Virtual Mouse', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
