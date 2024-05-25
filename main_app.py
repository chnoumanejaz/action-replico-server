import math as math
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import cv2
import mediapipe as mp
import pandas as pd
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
CORS(app)


os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


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




@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/v1/users', methods=['GET'])
def get_data():
    data =   [
            {
                'name': 'John Doe',
                'email': 'johndoe@example.com',
            }
        ];
    return jsonify(data)
  
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

@app.route('/api/v1/ping', methods=['GET'])
def pingServer():    
    return jsonify({'code': '200', 'message': 'Successfully connected with flask!'})
   

if __name__ == "__main__":
    app.run(debug=True)