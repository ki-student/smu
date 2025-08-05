

import tensorflow as tf

print(tf.__version__)





import keras

print(keras.__version__)



"""from roboflow import Roboflow

rf = Roboflow(api_key="D6Ni5uEHKrVYpa35HlBe")
project = rf.workspace().project("sft-rgizv")
model = project.version("2").model

# 비디오 예측 작업 전송
job_id, signed_url, expire_time = model.predict_video(
    "/Users/giwonjun/Desktop/capstone/IMG_4073.mp4",
    fps=5,
    prediction_type="batch-video",
)

# 예측 결과 받아오기
results = model.poll_until_video_results(job_id)

print(results)
"""