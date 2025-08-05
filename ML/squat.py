import torch
import cv2
import time  # 시간 측정을 위한 모듈 추가

# YOLOv5 모델 로드 (best.pt 가중치 파일)
model = torch.hub.load('/Users/giwonjun/Desktop/capstone/ML/yolov5', 'custom', path='/Users/giwonjun/Desktop/capstone/ML/biceps.pt', source='local')
model.eval()  # 모델을 평가 모드로 전환 (추론 모드)

# 비디오 캡처 (동영상 파일로부터 탐지)
cap = cv2.VideoCapture('/Users/giwonjun/Desktop/capstone/train_data/curl3.mp4')

# 총 스쿼트 횟수를 저장할 변수
curl_count = 0
curl_detected = False  # 스쿼트 감지 여부
first_detection_time = 0  # 첫 번째 스쿼트 감지 시간 저장

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("비디오 프레임을 가져오지 못했습니다.")
        break

    # BGR을 RGB로 변환 및 크기 조정
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (640, 640))  # 모델의 입력 크기에 맞게 조정

    # YOLOv5 모델로 객체 탐지 수행
    results = model(frame_resized)  # 모델에 프레임을 전달

    # 탐지된 객체의 정보 DataFrame으로 변환
    results_df = results.pandas().xyxy[0]

    # 현재 프레임에서 'squat' 클래스 개수 세기
    current_curl_count = (results_df['name'] == 'curl').sum()

    # 현재 시간 가져오기
    current_time = time.time()

    # 스쿼트가 처음 2개 감지되면 카운트 증가
    if current_curl_count >= 2:
        if not curl_detected:
            curl_count += 1  # 카운트 증가
            curl_detected = True  # 스쿼트 감지 상태로 변경
            first_detection_time = current_time  # 첫 번째 감지 시간 업데이트

    # n초가 경과했는지 확인
    if curl_detected and (current_time - first_detection_time >= 2):
        curl_detected = False  # 스쿼트 감지 상태 초기화

    # 탐지된 결과를 이미지에 그리기
    annotated_frame = results.render()[0]

    # RGB에서 BGR로 변환
    annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

    # 탐지 결과 화면에 표시
    cv2.imshow('YOLOv5 Real-Time Detection', annotated_frame_bgr)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()

# 전체 'squat' 클래스의 개수 출력
print(f"curl_count : {curl_count}")
