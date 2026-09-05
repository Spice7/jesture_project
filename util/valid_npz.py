import numpy as np

# 데이터 잘 뽑혔나 실험용
FILE_NAME = "dataset/make_fist/p004_make_fist_0001.npz"

data = np.load(
    FILE_NAME
)

print(data.files)
landmarks = data["landmarks"]

print("landmarks:", landmarks.shape)
print("timestamps:", data["timestamps"].shape)
print("detected:", data["detected"].shape)

print("label:", data["label"])
print("participant:", data["participant_id"])
print("handedness:", data["handedness"])

print("duration:", data["duration"])
print("detection rate:", data["detection_rate"])

print("첫 프레임:")
print(landmarks[0])

print()

print("중간 프레임:")
print(landmarks[len(landmarks) // 2])

print()

print("마지막 프레임:")
print(landmarks[-1])