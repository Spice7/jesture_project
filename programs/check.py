"""지정한 NPZ의 라벨/손 정보와 손목 경로를 확인합니다. import 시 파일을 열지 않습니다."""

import argparse
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILE = PROJECT_ROOT / "dataset" / "swipe_left" / "user00_swipe_left_0001.npz"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filepath", nargs="?", type=Path, default=DEFAULT_FILE)
    args = parser.parse_args(argv)
    with np.load(args.filepath, allow_pickle=False) as data:
        print("저장 항목:", data.files)
        for key in ("label", "participant_id", "handedness", "mirrored", "collection_schema_version",
                    "detection_rate", "detection_rate_ratio", "augmentation", "synthetic"):
            if key in data:
                print(f"{key}:", data[key])
        landmarks = data["landmarks"]
        print("프레임 수:", len(landmarks))
        print("좌표 형상:", landmarks.shape)
        print("첫 프레임 좌표:", landmarks[0])
    import matplotlib.pyplot as plt
    wrist = landmarks[:, 0, :]
    plt.plot(wrist[:, 0], wrist[:, 1], marker="o")
    plt.scatter(wrist[0, 0], wrist[0, 1], color="green", label="start")
    plt.scatter(wrist[-1, 0], wrist[-1, 1], color="red", label="end")
    plt.gca().invert_yaxis()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Wrist trajectory (original coordinates)")
    plt.legend()
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    main()
