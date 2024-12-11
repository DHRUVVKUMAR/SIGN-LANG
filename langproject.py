import os
import cv2
import mediapipe as mp
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

# Set up the data directory
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 12  # Number of gestures to capture
dataset_size = 200  # Number of images per gesture

cap = cv2.VideoCapture(0)  # Use your camera (2 is the index; adjust if needed)

# Initialize MediaPipe Hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

for j in range(number_of_classes):
    gesture_path = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(gesture_path):
        os.makedirs(gesture_path)

    print('Collecting data for gesture class {}'.format(j))

    # Initial prompt to start capturing each gesture
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image from camera.")
            break

        # Flip the frame for a better user view (optional)
        frame = cv2.flip(frame, 1)

        # Convert the BGR frame to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        # Display the frame with hand annotations
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Show instructions
        cv2.putText(frame, 'Ready? Press "Q" to start!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)

        if cv2.waitKey(25) == ord('q'):
            break

    # Capture and save frames for the gesture
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image from camera.")
            break

        # Flip the frame for a better user view (optional)
        frame = cv2.flip(frame, 1)

        # Convert the BGR frame to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        # If hands are detected, save the image
        if result.multi_hand_landmarks:
            hand_count = len(result.multi_hand_landmarks)
            if hand_count == 1 or hand_count == 2:
                cv2.imshow('frame', frame)
                cv2.waitKey(25)
                # Save each frame with an incrementing filename
                cv2.imwrite(os.path.join(gesture_path, '{}.jpg'.format(counter)), frame)
                counter += 1

    # Wait until all gestures are captured
cap.release()
cv2.destroyAllWindows()

# Initialize MediaPipe Hands solution for data processing
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Set up data directory
DATA_DIR = './data'

# Initialize lists to store data and labels
data = []
labels = []

# Loop through each gesture class folder
for dir_ in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(class_dir):
        continue

    # Loop through each image in the gesture class folder
    for img_path in os.listdir(class_dir):
        data_aux = []  # List to store landmarks for the current image
        x_coords = []  # List to store x coordinates for normalization
        y_coords = []  # List to store y coordinates for normalization

        # Read image
        img = cv2.imread(os.path.join(class_dir, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image to detect hand landmarks
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract x, y coordinates for each landmark of the current hand
                for landmark in hand_landmarks.landmark:
                    x = landmark.x
                    y = landmark.y
                    x_coords.append(x)
                    y_coords.append(y)

                # Normalize landmarks by subtracting the minimum x and y
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_coords))
                    data_aux.append(landmark.y - min(y_coords))

            # Append processed data and label after detecting the hand(s)
            data.append(data_aux)
            labels.append(dir_)

# Save the processed data and labels to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Data has been processed and saved to 'data.pickle'.")

# Load the data and labels from the pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Extract the data and labels
data = data_dict['data']
labels = data_dict['labels']

# Check the length of each sequence in data to determine the max length
max_length = max([len(d) for d in data])

# Pad sequences with zeros to the max length
data_padded = [d + [0]*(max_length - len(d)) if len(d) < max_length else d for d in data]

# Convert the padded data and labels to numpy arrays
data = np.asarray(data_padded)
labels = np.asarray(labels)

# Define the correct gesture labels (your specified gestures)
gesture_labels = ['HELLO', 'THANKYOU', 'ILOVEYOU', 'YES', 'NO', 'A', 'B', 'C', 'V', 'W', 'O', 'L']

# Map numeric labels to gesture names
# Assuming labels in pickle file are numeric, map them to gesture names
labels = np.array([gesture_labels[int(label)] if label.isdigit() else label for label in labels])

# Split the data into training and testing sets with stratified sampling
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize and train the RandomForestClassifier model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions and evaluate the model
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print(f'{score * 100:.2f}% of samples were classified correctly!')

# Save the trained model to a pickle file
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

# Load the trained model from the pickle file
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands solution
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Gesture labels dictionary (for reference if needed)
gesture_labels = ['HELLO', 'THANKYOU', 'ILOVEYOU', 'YES', 'NO', 'A', 'B', 'C', 'V', 'W', 'O', 'L']
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

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')
