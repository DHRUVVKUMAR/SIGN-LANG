import pickle
import numpy as np
import cv2
import mediapipe as mp

# Load the trained model from the pickle file
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Gesture labels dictionary (for reference if needed)
gesture_labels = ['A', 'B', 'C', 'YES', 'NO', 'HELLO', 'THANKYOU', 'ILOVEYOU', 'V', 'W', 'SORRY', 'L']
labels_dict = {i: gesture_labels[i] for i in range(len(gesture_labels))}

while True:
    data_aux = []

    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image from camera.")
        break

    H, W, _ = frame.shape

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image to detect hand landmarks
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        # Initialize x_ and y_ coordinates
        hands_data = []

        # Loop through all detected hands
        for hand_landmarks in results.multi_hand_landmarks:
            x_ = []
            y_ = []
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Extract normalized landmark coordinates for this hand
            for landmark in hand_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                x_.append(x)
                y_.append(y)

            # Normalize landmarks by minimum x and y
            hand_data = []
            for landmark in hand_landmarks.landmark:
                hand_data.append(landmark.x - min(x_))
                hand_data.append(landmark.y - min(y_))

            # Add the hand data to hands_data
            hands_data.append(hand_data)

        # If there is only one hand, append zeros to make the total feature size 84 (42 features per hand)
        if len(hands_data) == 1:
            hands_data.append([0] * 42)  # Append zeros for the second hand

        # Flatten the hands_data list into a single list
        data_aux = hands_data[0] + hands_data[1]

        # Ensure the data_aux has exactly 84 features
        if len(data_aux) != 84:
            print(f"Skipping frame: Expected 84 features, but got {len(data_aux)} features.")
            continue

        # Calculate bounding box coordinates for the first hand (using its landmarks)
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10

        # Make a prediction using the trained model
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = prediction[0]  # Model returns the predicted label as a string

        # Display prediction and bounding box on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    # Show the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
