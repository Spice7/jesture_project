import argparse
from pathlib import Path
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="NPZ 내용 확인")
    parser.add_argument("file", type=Path)
    args = parser.parse_args()
    path = args.file if args.file.is_absolute() else Path(__file__).resolve().parents[1] / args.file
    with np.load(path, allow_pickle=False) as data:
        print(data.files)
        for key in data.files:
            print(key, data[key].shape, data[key] if data[key].ndim == 0 else "")
        landmarks = data["landmarks"]
        for i in (0, len(landmarks) // 2, -1):
            print(i, landmarks[i])


if __name__ == "__main__":
    main()
