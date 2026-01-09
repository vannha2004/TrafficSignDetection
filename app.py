from flask import Flask, render_template, Response, request, jsonify, url_for
import cv2
import os
import time
import logging
from config import UPLOAD_FOLDER, AVAILABLE_MODELS, FLASK_DEBUG, FLASK_HOST, FLASK_PORT
from utils import setup_logging, validate_image_file, is_supported_image_format
from models_service import get_model_manager
from history_service import get_history_manager
import threading
import yaml
from ultralytics import YOLO

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16*1024*1024  

model_manager = get_model_manager()
history_manager = get_history_manager()

yolo_model = YOLO("models/yolo/best.pt")

with open("models/yolo/data.yaml", "r") as f:
    data_yaml = yaml.safe_load(f)
    class_names = data_yaml["names"]

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/realtime')
def realtime():
    return render_template('realtime.html')


@app.route('/history')
def history():
    try:
        history_data = history_manager.get_all_history()
        return render_template('history.html', json_data=history_data)
    except Exception as e:
        return render_template('history.html', json_data=[])


@app.route('/api/models')
def api_models():
    model_info = {k: v['name'] for k, v in AVAILABLE_MODELS.items()}
    return jsonify(model_info)


@app.route('/api/history')
def api_history():
    try:
        recent_history = history_manager.get_recent_history(10)
        return jsonify(recent_history)
    except Exception as e:
        return jsonify([])


@app.route('/api/statistics')
def api_statistics():
    try:
        stats = history_manager.get_statistics()
        return jsonify(stats)
    except Exception as e:
        return jsonify({})


def gen_frames():
    camera = None
    try:
        camera = cv2.VideoCapture(0)

        #640x480 30fps
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)

        if not camera.isOpened():
            return

        while True:
            success, frame = camera.read()
            if not success:
                break

            try:
                results = yolo_model(frame, verbose=False)

                detections = []
                for box in results[0].boxes:
                    class_id = int(box.cls)
                    class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{class_name} {float(box.conf):.2f}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 255, 0), 2)

                    detections.append({
                        "class_name": class_name,
                        "confidence": float(box.conf),
                        "bbox": [x1, y1, x2, y2],
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })

                if detections:
                    history_manager.add_detections(detections)

            except Exception as e:
                print(f"Error processing frame: {e}")

            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ret:
                continue

            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'

    except Exception as e:
        print(f"Error in video streaming: {e}")
    finally:
        if camera is not None:
            camera.release()


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/image_upload', methods=['GET', 'POST'])
def image_upload():
    if request.method == 'GET':
        model_options = {k: v['name'] for k, v in AVAILABLE_MODELS.items()}
        return render_template('image_upload.html', available_models=model_options)
    
    try:
        file = request.files.get('file')
        if not file or not file.filename:
            return jsonify({'error': 'No file uploaded'}), 400
        
        if not is_supported_image_format(file.filename):
            return jsonify({'error': 'Unsupported file format'}), 400
        
        selected_model = request.form.get('model_name', 'yolov12')
        if selected_model not in AVAILABLE_MODELS:
            return jsonify({'error': 'Invalid model selected'}), 400
        
        filename = f"{time.time()}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        if not validate_image_file(filepath):
            os.remove(filepath)  # Clean up invalid file
            return jsonify({'error': 'Invalid image file'}), 400
        
        img = cv2.imread(filepath)
        
        detections = model_manager.predict(selected_model, img, filepath)
        
        history_manager.add_detections(detections)
        
        response_data = {
            'detections': detections,
            'image_url': url_for('static', filename=f'upload/{filename}'),
            'model_used': AVAILABLE_MODELS[selected_model]['name'],
            'total_detections': len(detections)
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': 'Internal server error'}), 500



if __name__ == '__main__':
    try:
        app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
    except Exception as e:
        raise