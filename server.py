from flask import Flask, request, jsonify
import cv2
import numpy as np
import pickle
import mediapipe as mp

app = Flask(__name__)

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['frame']
    frame = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    data_aux = []
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        x_ = []
        y_ = []
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                x_.append(x)
                y_.append(y)
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))
        if len(data_aux) == 84:
            prediction = model.predict([np.asarray(data_aux)])
            return jsonify({'gesture': prediction[0]})
    return jsonify({'gesture': 'none'})

if __name__ == '__main__':
    app.run(debug=True)
