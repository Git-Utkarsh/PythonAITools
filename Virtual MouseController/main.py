import cv2
import mediapipe as mp
import math
import pyautogui
import screeninfo

# Solution APIs
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Get screen dimensions
screen = screeninfo.get_monitors()[0]
screen_width, screen_height = screen.width, screen.height

# Webcam Setup
wCam = 640
hCam = int((screen_height / screen_width) * wCam)
cam = cv2.VideoCapture(0)
cam.set(3, wCam)
cam.set(4, hCam)

# Circle parameters
circle_radius = int(min(screen_width, screen_height) * 0.03)  # 3% of the minimum screen dimension
large_circle_radius = 50  # Larger circle radius in pixels
circle_color = (255, 0, 0)  # Blue color in BGR format
circle_thickness = 2

# MediaPipe Hand Landmark Model
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

    while cam.isOpened():
        success, image = cam.read()

        image = cv2.flip(image, 1)  # Flip horizontally for a mirror effect
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                # Get coordinates of thumb and index finger
                thumb_x, thumb_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * screen_width), \
                                    int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * screen_height)
                index_x, index_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * screen_width), \
                                    int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * screen_height)

                # Draw larger circles around thumb and index finger
                cv2.circle(image, (thumb_x, thumb_y), large_circle_radius, circle_color, circle_thickness)
                cv2.circle(image, (index_x, index_y), large_circle_radius, circle_color, circle_thickness)

                # Move cursor to index finger position
                pyautogui.moveTo(index_x, index_y, duration=0)

                # Check if thumb and index finger circles merge to perform a left click
                if math.dist((thumb_x, thumb_y), (index_x, index_y)) < (large_circle_radius * 2):
                    pyautogui.click()

        cv2.imshow('Virtual Cursor Controller', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()
