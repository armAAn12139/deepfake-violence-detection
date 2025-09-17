import cv2
import mediapipe as mp
import os

mp_face_detection = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def extract_frames(video_path, output_dir, frame_rate=1):
    """
    Extract frames from video at 1 FPS by default.
    Saves frames to output_dir as images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = fps // frame_rate

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"[INFO] Extracted {saved_count} frames to {output_dir}")
    return saved_count

def detect_and_crop_faces(image_path, output_dir, min_detection_confidence=0.5):
    """
    Detect faces in a single image and crop them.
    Saves cropped faces to output_dir.
    Returns list of cropped face image paths.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image = cv2.imread(image_path)
    results = mp_face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    face_paths = []

    if results.detections:
        for idx, detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = image.shape
            bbox = [
                int(bboxC.xmin * w),
                int(bboxC.ymin * h),
                int(bboxC.width * w),
                int(bboxC.height * h)
            ]

            x, y, w_box, h_box = bbox
            x, y = max(0, x), max(0, y)
            face_img = image[y:y+h_box, x:x+w_box]

            face_path = os.path.join(output_dir, f"face_{idx:04d}.jpg")
            cv2.imwrite(face_path, face_img)
            face_paths.append(face_path)

    return face_paths
