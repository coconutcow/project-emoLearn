import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils  # Drawing helpers
mp_holistic = mp.solutions.holistic  # Mediapipe Solutions
wCam, hCam = 1920, 1088
# Image capture
# cap = cv2.VideoCapture('jamal.jpg')

# Video capture
cap = cv2.VideoCapture('MVI_3053_Trim.mp4')


# Webcam capture
# cap = cv2.VideoCapture(0)

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        # Image Feed
        # image = cv2.imread("jamal.jpg")
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Video feed
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make Detections
        results = holistic.process(image)
        # print(results.face_landmarks)

        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks

        # Recolor image back to BGR for rendering
        image.flags.writeable = True

        # Needed for video
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 1. Draw face landmarks
        # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS,
        #                           mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1),
        #                           mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
        #                           )

        # 2. Right hand
        # mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        #                           mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=6),
        #                           mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
        #                           )

        # 3. Left Hand
        # mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        #                           mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        #                           mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
        #                           )

        # 4. Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(41, 131, 0), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
                                  )

        # Resizing the image
        image = cv2.resize(image, (1280, 720))
        cv2.imshow('Raw Webcam Feed', image)
        print(image.shape)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()