# import os
# import io
# import base64
# import uuid
# import json
# from typing import List, Dict

# from flask import Flask, request, jsonify, send_file
# from flask_cors import CORS
# from werkzeug.utils import secure_filename

# import cv2
# import numpy as np
# from PIL import Image
# import mediapipe as mp

# # -------------------------
# # Configuration
# # -------------------------
# UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "uploads")
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# # Flask app
# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# # -------------------------
# # Initialize MediaPipe Pose
# # -------------------------
# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# pose_detector = mp_pose.Pose(
#     static_image_mode=False,
#     model_complexity=1,
#     smooth_landmarks=True,
#     enable_segmentation=False,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5,
# )

# # -------------------------
# # Helpers
# # -------------------------
# def read_image_from_bytes(image_bytes: bytes) -> np.ndarray:
#     """Read image bytes into OpenCV BGR image."""
#     try:
#         nparr = np.frombuffer(image_bytes, np.uint8)
#         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#         return img
#     except Exception as e:
#         print(f"Error decoding image: {e}")
#         return None

# def encode_image_to_base64_jpeg(img_bgr: np.ndarray, quality: int = 80) -> str:
#     """Encode a BGR image to base64 JPEG string."""
#     try:
#         ret, buf = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
#         if not ret:
#             raise ValueError("Failed to encode image")
#         b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
#         return b64
#     except Exception as e:
#         print(f"Error encoding image: {e}")
#         return None

# def pose_landmarks_to_list(landmarks) -> List[Dict]:
#     """Convert MediaPipe landmarks into a list of dicts (x,y,z,visibility)."""
#     out = []
#     if landmarks:
#         for idx, lm in enumerate(landmarks.landmark):
#             out.append({
#                 "id": idx,
#                 "x": lm.x,
#                 "y": lm.y,
#                 "z": lm.z,
#                 "visibility": getattr(lm, "visibility", 0.0)
#             })
#     return out

# def draw_pose_landmarks(image_bgr, pose_landmarks):
#     """Draw pose landmarks and connections on the image."""
#     annotated_image = image_bgr.copy()
    
#     # Draw the pose landmarks and connections
#     mp_drawing.draw_landmarks(
#         annotated_image,
#         pose_landmarks,
#         mp_pose.POSE_CONNECTIONS,
#         landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
#     )
    
#     return annotated_image

# # -------------------------
# # Routes
# # -------------------------
# @app.route("/", methods=["GET"])
# def home():
#     return jsonify({
#         "message": "MediaPipe Pose Detection API",
#         "status": "active",
#         "endpoints": {
#             "/health": "GET - Health check",
#             "/process_frame": "POST - Process single frame with pose detection",
#             "/process_frame_annotated": "POST - Process frame and return annotated image"
#         }
#     }), 200

# @app.route("/health", methods=["GET"])
# def health():
#     return jsonify({"status": "ok", "service": "mediapipe-pose-api"}), 200

# @app.route("/process_frame", methods=["POST"])
# def process_frame():
#     """
#     Accepts one frame as multipart form-data 'frame' or raw bytes.
#     Runs MediaPipe Pose and returns landmarks data.
#     """
#     try:
#         if "frame" in request.files:
#             # Multipart form data
#             file = request.files["frame"]
#             image_bytes = file.read()
#         elif request.content_type and 'image/' in request.content_type:
#             # Raw image data
#             image_bytes = request.get_data()
#         else:
#             return jsonify({"error": "No frame provided. Send as multipart 'frame' or raw image data"}), 400

#         # Read and process image
#         img_bgr = read_image_from_bytes(image_bytes)
#         if img_bgr is None:
#             return jsonify({"error": "Unable to decode image"}), 400

#         # Convert to RGB for MediaPipe
#         img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
#         # Process with MediaPipe
#         results = pose_detector.process(img_rgb)

#         response = {
#             "status": "no_pose_detected", 
#             "landmarks": [],
#             "frame_info": {
#                 "width": img_bgr.shape[1],
#                 "height": img_bgr.shape[0]
#             }
#         }

#         if results.pose_landmarks:
#             response["status"] = "pose_detected"
#             response["landmarks"] = pose_landmarks_to_list(results.pose_landmarks)
            
#             # Add pose world landmarks if available
#             if results.pose_world_landmarks:
#                 response["world_landmarks"] = pose_landmarks_to_list(results.pose_world_landmarks)

#         return jsonify(response), 200

#     except Exception as e:
#         return jsonify({"error": f"Processing failed: {str(e)}"}), 500

# @app.route("/process_frame_annotated", methods=["POST"])
# def process_frame_annotated():
#     """
#     Process frame and return both landmarks and annotated image with pose mesh.
#     """
#     try:
#         if "frame" in request.files:
#             file = request.files["frame"]
#             image_bytes = file.read()
#         elif request.content_type and 'image/' in request.content_type:
#             image_bytes = request.get_data()
#         else:
#             return jsonify({"error": "No frame provided. Send as multipart 'frame' or raw image data"}), 400

#         # Read and process image
#         img_bgr = read_image_from_bytes(image_bytes)
#         if img_bgr is None:
#             return jsonify({"error": "Unable to decode image"}), 400

#         # Convert to RGB for MediaPipe
#         img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
#         # Process with MediaPipe
#         results = pose_detector.process(img_rgb)

#         response = {
#             "status": "no_pose_detected", 
#             "landmarks": [],
#             "annotated_frame": None,
#             "frame_info": {
#                 "width": img_bgr.shape[1],
#                 "height": img_bgr.shape[0]
#             }
#         }

#         if results.pose_landmarks:
#             response["status"] = "pose_detected"
#             response["landmarks"] = pose_landmarks_to_list(results.pose_landmarks)
            
#             # Create annotated image with pose mesh
#             annotated_image = draw_pose_landmarks(img_bgr, results.pose_landmarks)
            
#             # Encode annotated image to base64
#             b64_img = encode_image_to_base64_jpeg(annotated_image, quality=85)
#             if b64_img:
#                 response["annotated_frame"] = f"data:image/jpeg;base64,{b64_img}"

#         return jsonify(response), 200

#     except Exception as e:
#         return jsonify({"error": f"Processing failed: {str(e)}"}), 500

# @app.route("/batch_process", methods=["POST"])
# def batch_process():
#     """
#     Process multiple frames in one request.
#     Expects JSON with list of base64 encoded images.
#     """
#     try:
#         data = request.get_json()
#         if not data or 'frames' not in data:
#             return jsonify({"error": "No frames array provided"}), 400
        
#         frames = data['frames']
#         results = []
        
#         for i, frame_data in enumerate(frames):
#             if frame_data.startswith('data:image'):
#                 # Remove data URL prefix
#                 frame_data = frame_data.split(',')[1]
            
#             # Decode base64
#             try:
#                 image_bytes = base64.b64decode(frame_data)
#                 img_bgr = read_image_from_bytes(image_bytes)
                
#                 if img_bgr is not None:
#                     img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#                     pose_results = pose_detector.process(img_rgb)
                    
#                     frame_result = {
#                         "frame_index": i,
#                         "status": "pose_detected" if pose_results.pose_landmarks else "no_pose_detected",
#                         "landmarks": pose_landmarks_to_list(pose_results.pose_landmarks)
#                     }
#                     results.append(frame_result)
#                 else:
#                     results.append({
#                         "frame_index": i,
#                         "status": "error",
#                         "error": "Failed to decode image"
#                     })
                    
#             except Exception as e:
#                 results.append({
#                     "frame_index": i,
#                     "status": "error", 
#                     "error": str(e)
#                 })
        
#         return jsonify({
#             "processed_frames": len(results),
#             "results": results
#         }), 200
        
#     except Exception as e:
#         return jsonify({"error": f"Batch processing failed: {str(e)}"}), 500

# # Error handlers
# @app.errorhandler(413)
# def too_large(e):
#     return jsonify({"error": "File too large"}), 413

# @app.errorhandler(500)
# def internal_error(e):
#     return jsonify({"error": "Internal server error"}), 500

# @app.errorhandler(404)
# def not_found(e):
#     return jsonify({"error": "Endpoint not found"}), 404

# # Run locally with `python app.py`
# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 5005))
#     debug = os.environ.get("DEBUG", "False").lower() == "true"
#     app.run(host="0.0.0.0", port=port, debug=debug)



import os
import io
import base64
import json
from typing import List, Dict
import logging

from flask import Flask, request, jsonify
from flask_cors import CORS

import cv2
import numpy as np
import mediapipe as mp

# -------------------------
# Logging configuration
# -------------------------
logging.basicConfig(level=logging.INFO)

# -------------------------
# Configuration
# -------------------------
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# -------------------------
# Initialize MediaPipe Pose
# -------------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

pose_detector = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# -------------------------
# Helpers
# -------------------------
def read_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Read image bytes into OpenCV BGR image."""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logging.error(f"Error decoding image: {e}")
        return None

def encode_image_to_base64_jpeg(img_bgr: np.ndarray, quality: int = 90) -> str:
    """Encode a BGR image to base64 JPEG string."""
    try:
        ret, buf = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ret:
            raise ValueError("Failed to encode image")
        b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
        return b64
    except Exception as e:
        logging.error(f"Error encoding image: {e}")
        return None

def pose_landmarks_to_list(landmarks) -> List[Dict]:
    """Convert MediaPipe landmarks into a list of dicts (x,y,z,visibility)."""
    out = []
    if landmarks:
        for idx, lm in enumerate(landmarks.landmark):
            out.append({
                "id": idx,
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": getattr(lm, "visibility", 0.0)
            })
    return out

def draw_pose_landmarks(image_bgr, pose_landmarks):
    """Draw pose landmarks and connections with custom style on the image."""
    annotated_image = image_bgr.copy()
    
    # Custom styles
    landmark_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=5)  # Bright green landmarks
    connection_style = mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=3, circle_radius=0)  # Darker green connections

    mp_drawing.draw_landmarks(
        annotated_image,
        pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=landmark_style,
        connection_drawing_spec=connection_style
    )
    
    return annotated_image

# -------------------------
# Routes
# -------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "MediaPipe Pose Detection API",
        "status": "active",
        "endpoints": {
            "/health": "GET - Health check",
            "/process_frame": "POST - Process single frame with pose detection",
            "/process_frame_annotated": "POST - Process frame and return annotated image",
            "/batch_process": "POST - Process multiple frames (JSON list of base64 images)"
        }
    }), 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "mediapipe-pose-api"}), 200

@app.route("/process_frame", methods=["POST"])
def process_frame():
    """Accepts one frame and returns landmarks data."""
    try:
        if "frame" in request.files:
            file = request.files["frame"]
            image_bytes = file.read()
        elif request.content_type and 'image/' in request.content_type:
            image_bytes = request.get_data()
        else:
            return jsonify({"error": "No frame provided"}), 400

        img_bgr = read_image_from_bytes(image_bytes)
        if img_bgr is None:
            return jsonify({"error": "Unable to decode image"}), 400

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(img_rgb)

        response = {
            "status": "pose_detected" if results.pose_landmarks else "no_pose_detected",
            "landmarks": pose_landmarks_to_list(results.pose_landmarks),
            "frame_info": {"width": img_bgr.shape[1], "height": img_bgr.shape[0]}
        }

        logging.info(f"Processed frame: status={response['status']}, size={img_bgr.shape}")
        return jsonify(response), 200

    except Exception as e:
        logging.error(f"Error processing frame: {e}")
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app.route("/process_frame_annotated", methods=["POST"])
def process_frame_annotated():
    """Process frame and return landmarks + annotated image."""
    try:
        if "frame" in request.files:
            file = request.files["frame"]
            image_bytes = file.read()
        elif request.content_type and 'image/' in request.content_type:
            image_bytes = request.get_data()
        else:
            return jsonify({"error": "No frame provided"}), 400

        img_bgr = read_image_from_bytes(image_bytes)
        if img_bgr is None:
            return jsonify({"error": "Unable to decode image"}), 400

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(img_rgb)

        annotated_image = draw_pose_landmarks(img_bgr, results.pose_landmarks) if results.pose_landmarks else img_bgr
        b64_img = encode_image_to_base64_jpeg(annotated_image, quality=90)

        response = {
            "status": "pose_detected" if results.pose_landmarks else "no_pose_detected",
            "landmarks": pose_landmarks_to_list(results.pose_landmarks),
            "annotated_frame": f"data:image/jpeg;base64,{b64_img}" if b64_img else None,
            "frame_info": {"width": img_bgr.shape[1], "height": img_bgr.shape[0]}
        }

        logging.info(f"Processed annotated frame: status={response['status']}")
        return jsonify(response), 200

    except Exception as e:
        logging.error(f"Error processing annotated frame: {e}")
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app.route("/batch_process", methods=["POST"])
def batch_process():
    """Process multiple frames in one request (JSON list of base64 images)."""
    try:
        data = request.get_json()
        if not data or 'frames' not in data:
            return jsonify({"error": "No frames array provided"}), 400
        
        frames = data['frames']
        results = []

        for i, frame_data in enumerate(frames):
            if frame_data.startswith('data:image'):
                frame_data = frame_data.split(',')[1]

            try:
                image_bytes = base64.b64decode(frame_data)
                img_bgr = read_image_from_bytes(image_bytes)
                
                if img_bgr is not None:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    pose_results = pose_detector.process(img_rgb)
                    frame_result = {
                        "frame_index": i,
                        "status": "pose_detected" if pose_results.pose_landmarks else "no_pose_detected",
                        "landmarks": pose_landmarks_to_list(pose_results.pose_landmarks)
                    }
                    results.append(frame_result)
                else:
                    results.append({"frame_index": i, "status": "error", "error": "Failed to decode image"})
            except Exception as e:
                results.append({"frame_index": i, "status": "error", "error": str(e)})

        logging.info(f"Batch processed {len(results)} frames")
        return jsonify({"processed_frames": len(results), "results": results}), 200

    except Exception as e:
        logging.error(f"Batch processing failed: {e}")
        return jsonify({"error": f"Batch processing failed: {str(e)}"}), 500

# -------------------------
# Error handlers
# -------------------------
@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large"}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

# -------------------------
# Run Flask
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5005))
    debug = os.environ.get("DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)

