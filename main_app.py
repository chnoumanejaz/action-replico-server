import math as math
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import os
import tensorflow as tf
from werkzeug.utils import secure_filename
 
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
CORS(app)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

LRCN_model = tf.keras.models.load_model('./model13.h5')

# Constants
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
SEQUENCE_LENGTH = 20
IMAGE_HEIGHT = 64   
IMAGE_WIDTH = 64   
# CLASSES_LIST = [
#     "BaseballPitch",
#     "Basketball",
#     "BenchPress",
#     "Biking",
#     "Billiards",
#     "BreastStroke",
#     "CleanAndJerk",
#     "Diving",
#     "Drumming",
#     "Fencing",
#     "GolfSwing",
#     "HighJump",
#     "HorseRace",
#     "HorseRiding",
#     "HulaHoop",
#     "JavelinThrow",
#     "JugglingBalls",
#     "JumpingJack",
#     "JumpRope",
#     "Kayaking",
#     "Lunges",
#     "MilitaryParade",
#     "Mixing",
#     "Nunchucks",
#     "PizzaTossing",
#     "PlayingGuitar",
#     "PlayingPiano",
#     "PlayingTabla",
#     "PlayingViolin",
#     "PoleVault",
#     "PommelHorse",
#     "PullUps",
#     "Punch",
#     "PushUps",
#     "RockClimbingIndoor",
#     "RopeClimbing",
#     "Rowing",
#     "SalsaSpin",
#     "SkateBoarding",
#     "Skiing",
#     "Skijet",
#     "SoccerJuggling",
#     "Swing",
#     "TaiChi",
#     "TennisSwing",
#     "ThrowDiscus",
#     "TrampolineJumping",
#     "VolleyballSpiking",
#     "WalkingWithDog",
#     "YoYo"
# ]


CLASSES_LIST = [
    "Basketball",
    "BenchPress",
    "Fencing",
    "GolfSwing",
    "HighJump",
    "PlayingTabla",
    "PullUps",
    "Punch",
    "PushUps",
    "SalsaSpin",
    "SkateBoarding",
    "Swing",
    "TaiChi",
]

# CLASSES_LIST = ["WalkingWithDog", "TaiChi", "Swing", "HorseRace"]


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def detect_poses(video_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    cap = cv2.VideoCapture(video_path)

    pose_data = []
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect poses in the frame
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            # Collect landmarks
            landmarks = results.pose_landmarks.landmark
            pose_row = [frame_index]
            for landmark in landmarks:
                pose_row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            pose_data.append(pose_row)

            mp.solutions.drawing_utils.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=6, circle_radius=6)
            )
            mp.solutions.drawing_utils.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=6, circle_radius=6)
            )
        frame_index += 1

    cap.release()

    # Convert pose data to DataFrame
    columns = ['frame'] + [f'{name}_{axis}' for name in range(33) for axis in ['x', 'y', 'z', 'visibility']]
    pose_df = pd.DataFrame(pose_data, columns=columns)

    documents_folder = os.path.expanduser('~/Documents')
    output_file_path = os.path.join(documents_folder, os.path.basename(video_path) + "_landmarks.csv")
    
    # Save the DataFrame to a CSV file in the Documents folder
    pose_df.to_csv(output_file_path, index=False)

    return output_file_path


def predict_single_action(video_file_path, SEQUENCE_LENGTH):
    video_reader = cv2.VideoCapture(video_file_path)

    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frames_list = []    
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))    
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH), 1)    
    for frame_counter in range(SEQUENCE_LENGTH):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()
        
        if not success:
            break

        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255.0

        frames_list.append(normalized_frame)
    # Release the VideoCapture object.
    video_reader.release()

    # Passing the pre-processed frames to the model and get the predicted probabilities.
    predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_list, axis=0))[0]

    # Get the index of class with highest probability.
    predicted_label = np.argmax(predicted_labels_probabilities)

    # Get the class name using the retrieved index.
    predicted_class_name = CLASSES_LIST[predicted_label]

    # Return the predicted action and confidence.
    return predicted_class_name, float(predicted_labels_probabilities[predicted_label])


@app.route('/')
def home():
    return render_template('index.html')

  
@app.route('/api/v1/animate',  methods=['POST'])
def animate_route():
    if 'video' not in request.files:
        return jsonify({
            'status': 'error',
            'code': 400,
            'message': 'No video file provided'
        }), 400

    video = request.files['video']
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
    video.save(video_path)

    try:
        csv_file_path = detect_poses(video_path)
        return jsonify({
            'status': 'success',
            'code': 200,
            'message': 'Pose detection completed successfully',
            'csvFilePath': csv_file_path
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'code': 500,
            'message': f'An error occurred: {str(e)}'
        }), 500
    finally:
        # Clean up uploaded file
        if os.path.exists(video_path):
            os.remove(video_path)


@app.route('/api/v1/classify', methods=['POST'])
def classify_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        predicted_class_name, confidence = predict_single_action(file_path, SEQUENCE_LENGTH)
        
        response = {
            "video_path": file_path,
            "predicted_action": predicted_class_name,
            "confidence": confidence,
            "statuscode": 200,
            "message": "Prediction successful"
        }
        return jsonify(response)

@app.route('/api/v1/ping', methods=['GET'])
def pingServer():    
    return jsonify({'code': '200', 'message': 'Successfully connected with flask!'})
   

if __name__ == "__main__":
    app.run(debug=True)