from flask import Flask, render_template, Response, request
import cv2
import torch, time
import numpy as np
from func import YOLOv1, cellboxes_to_boxes, draw_boxes, non_max_suppression, draw_boxes_no_score
import threading

app = Flask(__name__)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLOv1(S=4, B=2, C=20)
checkpoint = torch.load("model/best_model.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# Classes
CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# Global variables
camera = None
camera_on = False
conf_threshold = 0.01
iou_threshold = 0.01

# UPDATE YOUR predict_frame FUNCTION (NO SCORES)
# def predict_frame(frame, conf_threshold, iou_threshold):
#     orig_h, orig_w = frame.shape[:2]
#     img = cv2.resize(frame, (224, 224))
#     img_tensor = torch.from_numpy(img.transpose(2,0,1)).float().unsqueeze(0)/255.0
#     img_tensor = img_tensor.to(device)

#     with torch.no_grad():
#         preds = model(img_tensor)
    
#     boxes = cellboxes_to_boxes(preds.squeeze(0).cpu(), conf_threshold=conf_threshold)
#     boxes = non_max_suppression(boxes, iou_threshold=iou_threshold, conf_threshold=conf_threshold)
    
#     scaled_boxes = []
#     for x1, y1, x2, y2, conf, cls_idx in boxes:
#         x1 = max(0, min(1, x1))
#         y1 = max(0, min(1, y1))
#         x2 = max(0, min(1, x2))
#         y2 = max(0, min(1, y2))
        
#         scaled_x1 = int(x1 * orig_w)
#         scaled_y1 = int(y1 * orig_h)
#         scaled_x2 = int(x2 * orig_w)
#         scaled_y2 = int(y2 * orig_h)
        
#         if scaled_x2 > scaled_x1 and scaled_y2 > scaled_y1:
#             scaled_boxes.append([scaled_x1, scaled_y1, scaled_x2, scaled_y2, conf, cls_idx])
    
#     # DRAW BOXES WITHOUT SCORES
#     frame = draw_boxes_no_score(frame, scaled_boxes, CLASSES)
#     return frame

def predict_frame(frame, conf_threshold, iou_threshold):
    orig_h, orig_w = frame.shape[:2]

    # --- Step 1: Resize to 224x224 for model input ---
    img_resized = cv2.resize(frame, (224, 224))
    img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)

    # --- Step 2: Forward pass ---
    with torch.no_grad():
        preds = model(img_tensor)

    # --- Step 3: Convert predictions to boxes ---
    boxes = cellboxes_to_boxes(preds.squeeze(0).cpu(), conf_threshold=conf_threshold)
    boxes = non_max_suppression(boxes, iou_threshold=iou_threshold, conf_threshold=conf_threshold)

    # --- Step 4: Scale boxes from 224x224 → original image size ---
    scaled_boxes = []
    for x1, y1, x2, y2, conf, cls_idx in boxes:
        # Clip to [0,1] range (YOLO outputs are relative)
        x1 = max(0, min(1, x1))
        y1 = max(0, min(1, y1))
        x2 = max(0, min(1, x2))
        y2 = max(0, min(1, y2))

        # Scale to 224x224 (since model trained on that)
        scaled_x1 = int(x1 * 224)
        scaled_y1 = int(y1 * 224)
        scaled_x2 = int(x2 * 224)
        scaled_y2 = int(y2 * 224)

        # Now rescale to original frame
        x1_final = int((scaled_x1 / 224) * orig_w)
        y1_final = int((scaled_y1 / 224) * orig_h)
        x2_final = int((scaled_x2 / 224) * orig_w)
        y2_final = int((scaled_y2 / 224) * orig_h)

        if x2_final > x1_final and y2_final > y1_final:
            scaled_boxes.append([x1_final, y1_final, x2_final, y2_final, conf, cls_idx])

    # --- Step 5: Draw boxes ---
    frame = draw_boxes_no_score(frame, scaled_boxes, CLASSES)
    return frame



def gen_frames():
    global conf_threshold, iou_threshold, camera, camera_on
    while camera_on and camera is not None:
        success, frame = camera.read()
        if not success:
            break
        frame = predict_frame(frame, conf_threshold, iou_threshold)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def start_camera():
    """Initialize camera safely"""
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("Error: Could not start camera. Make sure it is connected and not used elsewhere.")
            camera = None
            return False
    return True

def stop_camera():
    """Safely stop camera"""
    global camera
    if camera is not None:
        camera.release()
        camera = None


# ADD THESE IMPORTS at top
from werkzeug.utils import secure_filename
import os

# ADD THIS FOLDER
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        if 'file' not in request.files:
            return {"error": "No file part"}, 400
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return {"error": "Invalid file or file type"}, 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Verify the file was saved
        if not os.path.exists(filepath):
            return {"error": "Failed to save file"}, 500

        # Read and process the image
        frame = cv2.imread(filepath)
        if frame is None:
            return {"error": "Failed to read image"}, 500

        frame = predict_frame(frame, 0.01, 0.01)
        if frame is None:
            return {"error": "Prediction failed"}, 500

        result_filename = f'result_{filename}'
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        cv2.imwrite(result_path, frame)

        # Verify the result file was saved
        if not os.path.exists(result_path):
            return {"error": "Failed to save result image"}, 500

        # Return the path to the template
        return render_template('index.html', result_image=f"/{app.config['UPLOAD_FOLDER']}/{result_filename}")
    except Exception as e:
        print(f"Error in upload_image: {e}")
        return {"error": str(e)}, 500


def process_image(file):
    try:
        if file.filename == '' or not allowed_file(file.filename):
            return None, "Invalid file or file type"
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Verify the file was saved
        if not os.path.exists(filepath):
            return None, "Failed to save file"

        # Read and process the image
        frame = cv2.imread(filepath)
        if frame is None:
            return None, "Failed to read image"

        frame = predict_frame(frame, 0.01, 0.01)
        if frame is None:
            return None, "Prediction failed"

        result_filename = f'result_{filename}'
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        cv2.imwrite(result_path, frame)

        # Verify the result file was saved
        if not os.path.exists(result_path):
            return None, "Failed to save result image"

        return f"/{app.config['UPLOAD_FOLDER']}/{result_filename}", None
    except Exception as e:
        print(f"Error in process_image: {e}")
        return None, str(e)


@app.route('/', methods=['GET', 'POST'])
def index():
    global camera_on
    result_image = None
    error = None

    if request.method == 'POST':
        # Handle camera toggle
        if 'camera' in request.form:
            new_camera_on = request.form.get('camera') == 'on'
            if new_camera_on and not camera_on:
                if start_camera():
                    camera_on = True
                else:
                    error = "Failed to start camera"
            elif not new_camera_on and camera_on:
                stop_camera()
                camera_on = False

        # Handle image upload
        if 'file' in request.files:
            file = request.files['file']
            result_image, error = process_image(file)

    # Add cache-busting query parameter to prevent browser caching issues
    if result_image:
        result_image = f"{result_image}?t={int(time.time())}"


    return render_template('index.html', camera_on=camera_on, result_image=result_image, error=error)

@app.route('/video_feed')
def video_feed():
    global camera_on
    if not camera_on or camera is None:
        return Response("Camera not available", status=404)
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/shutdown', methods=['POST'])
def shutdown():
    global camera_on, camera
    camera_on = False
    stop_camera()
    return "Server shutting down..."

if __name__ == '__main__':
    try:
        app.run(debug=True, threaded=True)
    except KeyboardInterrupt:
        stop_camera()
    finally:
        stop_camera()