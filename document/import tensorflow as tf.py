import tensorflow as tf
import numpy as np
import cv2

# 커스텀 DepthwiseConv2D 레이어 정의
class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, **kwargs):
        # 'groups' 키워드 인수를 무시하고 나머지 인수는 부모 클래스에 전달
        kwargs.pop('groups', None)  # 'groups' 키워드를 제거합니다.
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        # 입력과 추가 인수를 올바르게 처리
        return super().call(inputs, **kwargs)

# 모델 로드 시 커스텀 레이어 사용
model_filename = '/Users/giwonjun/Desktop/capstone/converted_keras/keras_model.h5'
model = tf.keras.models.load_model(model_filename, custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})

model.summary()  # 모델 구조 확인

# 카메라를 제어할 수 있는 객체
capture = cv2.VideoCapture(0)

# 카메라 길이 너비 조절
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# 이미지 처리하기
def preprocessing(frame):
    size = (224, 224)
    frame_resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

    # 이미지 정규화
    frame_normalized = (frame_resized.astype(np.float32) / 127.0) - 1

    # 이미지 차원 재조정 - 예측을 위해 reshape 해줍니다.
    frame_reshaped = frame_normalized.reshape((1, 224, 224, 3))
    return frame_reshaped

# 예측용 함수
def predict(frame):
    prediction = model.predict(frame)
    return prediction

while True:
    ret, frame = capture.read()

    if not ret:
        break

    preprocessed = preprocessing(frame)
    prediction = predict(preprocessed)

    if prediction[0, 0] < prediction[0, 1]:
        print('sit')
        cv2.putText(frame, 'sit', (0, 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
    else:
        cv2.putText(frame, 'stand', (0, 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
        print('stand')

    cv2.imshow("VideoFrame", frame)

    # 키보드 입력 감지 (ESC 키로 종료)
    if cv2.waitKey(1) == 27:
        break

capture.release()
cv2.destroyAllWindows()