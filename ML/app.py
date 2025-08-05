import torch
import cv2
import time
import threading
from flask import Flask, jsonify, Response, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# YOLOv5 모델 로드
model = torch.hub.load('/Users/giwonjun/Desktop/capstone/ML/yolov5', 'custom', path='/Users/giwonjun/Desktop/capstone/ML/biceps.pt', source='local')
model.eval()

# 비디오 캡처
cap = cv2.VideoCapture(0)

# 글로벌 변수로 squat_count 선언
squat_count = 0
detection_active = False

def detect_squats():
    global squat_count, detection_active
    start_time = time.time()

    while time.time() - start_time < 30:  # 30초 동안 측정
        ret, frame = cap.read()
        if not ret:
            print("Could not read frame from camera")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)
        results_df = results.pandas().xyxy[0]

        # 'squat' 클래스만 필터링하고 개수 누적
        squat_count += (results_df['name'] == 'squat').sum()

        # 30초 동안 감지 대기
        time.sleep(1)  # 1초마다 감지하도록 설정

    detection_active = False

@app.route('/')
def index():
    # 기본 페이지로 index.html 반환
    return render_template('index.html')

@app.route('/detect', methods=['GET'])
def detect():
    global squat_count, detection_active

    # 이미 감지가 활성화되어 있으면 500 에러 반환
    if detection_active:
        return jsonify({'error': 'Detection already in progress'}), 500

    # squat_count 초기화 및 감지 활성화
    squat_count = 0
    detection_active = True

    # 스쿼트 감지 쓰레드 시작
    threading.Thread(target=detect_squats).start()

    # 감지 완료 후 squat_count 반환 (30초 후)
    def wait_for_completion():
        while detection_active:
            time.sleep(1)
        return jsonify({'squat_count': int(squat_count)})

    return wait_for_completion()

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb)
            annotated_frame = results.render()[0]
            annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

            _, buffer = cv2.imencode('.jpg', annotated_frame_bgr)
            frame_encoded = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_encoded + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Flask 서버 실행
    app.run(host='0.0.0.0', port=3000)