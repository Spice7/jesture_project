import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
filepath = (
    PROJECT_ROOT
    / "data"
    / "dynamic"
    / "swipe_left"
    / "user00_swipe_left_0001.npz"
)

with np.load(filepath) as data:
    print("저장 항목:", data.files)
    print("라벨:", data["label"])
    print("참가자:", data["participant_id"])
    print("프레임 수:", len(data["landmarks"]))
    print("좌표 형상:", data["landmarks"].shape)
    print("검출률:", data["detection_rate"])
    print("첫 프레임 좌표:")
    print(data["landmarks"][0])

with np.load(filepath) as data:
    landmarks = data["landmarks"]

# 0번 랜드마크가 손목
wrist = landmarks[:, 0, :]

plt.plot(wrist[:, 0], wrist[:, 1], marker="o")
plt.scatter(wrist[0, 0], wrist[0, 1], color="green", label="start")
plt.scatter(wrist[-1, 0], wrist[-1, 1], color="red", label="end")

# 영상 좌표는 y가 아래로 증가
plt.gca().invert_yaxis()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Wrist trajectory")
plt.legend()
plt.axis("equal")
plt.show()
