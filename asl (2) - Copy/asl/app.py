from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pickle

app = Flask(__name__)

# MediaPipe and model initialization
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Load the model
try:
    model_dict = pickle.load(open('asl/model.p', 'rb'))
    model = model_dict['model']
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

labels_dict = {0: 'A', 1: 'B', 2: 'D'}

camera = None
is_detecting = False


def generate_frames():
    global camera, is_detecting
    while True:
        if not is_detecting:
            break

        success, frame = camera.read()
        if not success:
            break

        data_aux = []
        x_ = []
        y_ = []

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                try:
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_character = labels_dict[int(prediction[0])]
                except Exception as e:
                    predicted_character = "Error"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
        else:
            cv2.putText(frame, "No hand detected", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    global camera, is_detecting
    if not is_detecting:
        return jsonify({'status': 'error', 'message': 'Detection not started'})
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_detection', methods=['POST', 'GET'])
def start_detection():
    global camera, is_detecting
    if camera is None:
        camera = cv2.VideoCapture(0)
    is_detecting = True
    return jsonify({'status': 'success'})


@app.route('/stop_detection', methods=['POST', 'GET'])
def stop_detection():
    global is_detecting, camera
    is_detecting = False
    if camera:
        camera.release()
        camera = None
    return jsonify({'status': 'success'})


if __name__ == '__main__':
    app.run(debug=True)
