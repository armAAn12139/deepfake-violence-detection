from flask import Flask, render_template, request, jsonify
import os
from utils.preprocessing import extract_frames, detect_and_crop_faces
from utils.inference import predict_deepfake, predict_violence, predict_emotion
from utils.helpers import clear_directory

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['FRAMES_FOLDER'] = 'static/uploads/frames'
app.config['FACES_FOLDER'] = 'static/uploads/faces'

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['FRAMES_FOLDER'], exist_ok=True)
os.makedirs(app.config['FACES_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Clear old frames/faces
    clear_directory(app.config['FRAMES_FOLDER'])
    clear_directory(app.config['FACES_FOLDER'])

    analysis_results = {
        "deepfake": [],
        "violence": [],
        "emotion": [],
        "suspect_faces": [],
        "victim_faces": []
    }

    # 1️d Extract frames (1 FPS)
    extract_frames(file_path, app.config['FRAMES_FOLDER'], frame_rate=1)

    # 2️d Process each frame
    frame_files = sorted(os.listdir(app.config['FRAMES_FOLDER']))
    for frame_file in frame_files:
        frame_path = os.path.join(app.config['FRAMES_FOLDER'], frame_file)

        # Detect faces in frame and crop them
        face_paths = detect_and_crop_faces(frame_path, app.config['FACES_FOLDER'])

        for face_path in face_paths:
            # Run inferences
            deepfake_result = predict_deepfake(face_path)
            violence_result = predict_violence(face_path)
            emotion_result = predict_emotion(face_path)

            # Append results
            analysis_results['deepfake'].append(deepfake_result)
            analysis_results['violence'].append(violence_result)
            analysis_results['emotion'].append(emotion_result)
            analysis_results['suspect_faces'].append(face_path)  # For demo purposes
            # Assuming victim detection is the same as suspect for now
            analysis_results['victim_faces'].append(face_path)

    return jsonify(analysis_results)


if __name__ == '__main__':
    app.run(debug=True)
