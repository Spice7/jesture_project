# Source inventory

Top-level definitions and imports, read from the original branch Python AST.

## lkh/gesture_model/__init__.py



Definitions:

```python
from .gesture_detector import GestureDetector, GestureResult
```

## lkh/gesture_model/gesture_detector.py



Definitions: GestureResult, GestureDetector

```python
from dataclasses import dataclass
from pathlib import Path
import time
from ultralytics import YOLO
```

## lkh/main.py



Definitions:

```python

```

## lkh/programs/collect_gesture.py



Definitions: sanitize_name, get_longest_missing_run, calculate_movement_metrics, initialize_hand_detector, extract_landmarks, get_next_sample_id, get_dominant_handedness, update_index_csv, save_sequence, draw_hand_landmarks, draw_status, open_camera, main

```python
from __future__ import annotations
import csv
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
```

## lkh/programs/collect_gesture_video.py



Definitions: sanitize_name, get_longest_missing_run, calculate_movement_metrics, initialize_hand_detector, extract_landmarks, get_next_sample_id, get_dominant_handedness, update_index_csv, save_sequence, draw_hand_landmarks, draw_status, open_video, main

```python
from __future__ import annotations
import csv
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
```

## lkh/programs/extract_gesture_videos.py



Definitions: sanitize_name, get_longest_missing_run, calculate_movement_metrics, initialize_hand_detector, extract_landmarks, get_next_sample_id, get_dominant_handedness, update_index_csv, save_sequence, find_video_files, extract_video, main

```python
from __future__ import annotations
import csv
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
```

## lkh/programs/flip_image.py



Definitions:

```python
from pathlib import Path
from PIL import Image, ImageOps
```

## lkh/programs/validate_gesture_dataset.py



Definitions: scalar_str, scalar_float, scalar_int, scalar_bool, longest_missing_run, valid_frame_mask, hand_scale_per_frame, wrist_metrics, make_fist_metric, validate_npz, find_npz_files, write_report, print_summary, main

```python
from __future__ import annotations
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
```

## lkh/util/slice.py



Definitions: get_next_video_path, get_next_image_index, get_image_count, record_and_extract

```python
import cv2
from pathlib import Path
```

## lkh/util/valid_npz.py



Definitions:

```python
import numpy as np
```

## lkh/yolo/evaluate_yolo.py



Definitions: main

```python
from pathlib import Path
from ultralytics import YOLO
```

## lkh/yolo/test_yolo_webcam.py



Definitions: main

```python
from pathlib import Path
import time
import cv2
from ultralytics import YOLO
```

## lkh/yolo/train_yolo.py



Definitions: main

```python
from pathlib import Path
from ultralytics import YOLO
```

## hwangsoon/gesture/__init__.py

동적 제스처(LSTM) 파이프라인 공용 패키지.

Definitions:

```python
import sys as _sys
```

## hwangsoon/gesture/actions.py

제스처 → PC 기능 매핑(gestures.json) + Windows 키 입력.

Definitions: key_display, normalize_keys, press_combo, ActionMapper

```python
from __future__ import annotations
import ctypes
import json
import sys
import time
from pathlib import Path
from . import config
```

## hwangsoon/gesture/baseline.py

baseline: 손으로 짠 룰 분류기. GRU 가 이걸 확실히 이겨야 딥러닝을 쓴 이유가 된다.

Definitions: _tip_dist, rule_classify, predict_features

```python
import numpy as np
from . import config, preprocess
```

## hwangsoon/gesture/config.py

학습 파이프라인 공통 설정. 데이터 규약은 JIN 브랜치 수집기(programs/collect_gesture.py)를 따른다.

Definitions:

```python
from pathlib import Path
```

## hwangsoon/gesture/dataset.py

수집기(JIN) npz 로딩, 사람 단위 분할, 학습 배열 생성.

Definitions: FormatError, Sample, load_sample, load_dataset, summarize, split_by_person, reversed_negatives, lowering_negatives, shortened_swipes, build_arrays

```python
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from . import config, preprocess
```

## hwangsoon/gesture/gate.py

YOLO 정적 포즈 게이트: 팀(lkh) 의 gesture_model.GestureDetector 를 우리 파이프라인에 잇는 얇은 층.

Definitions: GateStatus, StaticGate, FakeGate

```python
from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path
from . import config
import torch
from gesture_model.gesture_detector import GestureDetector
```

## hwangsoon/gesture/landmarks.py

MediaPipe Tasks API(HandLandmarker) 래퍼. mediapipe>=1.0 은 mp.solutions 가 없다.

Definitions: ensure_model, HandTracker, draw_landmarks

```python
from __future__ import annotations
import urllib.request
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision
from . import config
```

## hwangsoon/gesture/model.py

시계열 분류기(GRU / LSTM) 정의·저장·추론. PyTorch.

Definitions: model_path, meta_path, GestureRNN, build_model, load_model, build_lstm, build_gru, save_meta, GestureClassifier

```python
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from . import config
```

## hwangsoon/gesture/pipeline.py

실시간 인식 파이프라인을 한 덩어리로: 프레임 → 손 관절 → 구간 감지 → GRU → 상식 검사 → 게이트 → 기능 실행.

Definitions: SessionLog, Event, FrameState, gap_runs, finger_extension, Pipeline

```python
from __future__ import annotations
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from . import config, preprocess, sanity
from .actions import ActionMapper
from .landmarks import HandTracker
from .model import GestureClassifier
from .segmenter import MotionSegmenter
from .gate import StaticGate
```

## hwangsoon/gesture/preprocess.py

녹화 원본(가변 길이, 결측 포함) → LSTM 입력 (SEQ_LEN, FEATURE_DIM) 변환.

Definitions: detection_mask, longest_gap, interpolate_missing, resample, normalize, to_features, sample_to_features, augment

```python
from __future__ import annotations
import numpy as np
from . import config
```

## hwangsoon/gesture/presets.py

제스처 → 키 매핑 프리셋. UI 의 '프리셋' 메뉴에서 한 번에 적용한다.

Definitions:

```python
from __future__ import annotations
```

## hwangsoon/gesture/sanity.py

모델 판정을 실행하기 전의 '상식 검사'.

Definitions: _basic, check, snap_released_now, fist_closed_now

```python
from __future__ import annotations
import numpy as np
```

## hwangsoon/gesture/segmenter.py

실시간 스트림에서 "정지 → 동작 → 정지" 구간을 잘라내는 감지기.

Definitions: Segment, MotionSegmenter

```python
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import numpy as np
from . import config
from . import sanity
```

## hwangsoon/gesture/training.py

학습·평가 공용 로직. train_model.py 와 compare_models.py 가 함께 쓴다. PyTorch.

Definitions: History, set_seed, _to_tensor, _evaluate, fit, metrics, predict, infer_ms_per_sample, baseline_metrics, report

```python
from __future__ import annotations
import random
import time
from dataclasses import dataclass, field
import numpy as np
import torch
from torch import nn
from . import baseline, config
from .model import DEVICE, build_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix
```

## hwangsoon/gesture_model/__init__.py



Definitions:

```python
from .gesture_detector import GestureDetector, GestureResult
```

## hwangsoon/gesture_model/gesture_detector.py



Definitions: GestureResult, GestureDetector

```python
from dataclasses import dataclass
from pathlib import Path
import time
from ultralytics import YOLO
```

## hwangsoon/programs/check.py



Definitions:

```python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
```

## hwangsoon/programs/collect_gesture.py



Definitions: sanitize_name, get_longest_missing_run, calculate_movement_metrics, initialize_hand_detector, extract_landmarks, get_next_sample_id, get_dominant_handedness, update_index_csv, save_sequence, draw_hand_landmarks, draw_status, open_camera, main

```python
from __future__ import annotations
import csv
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
```

## hwangsoon/reports/gru_vs_lstm_0907/scripts/bench_infer.py

CPU 배치1 추론 시간 재측정 (가중치 무관): 우리 GRU/LSTM(2층64+FC) vs 팀원 LSTM 1층. 200회 중앙값, 1스레드.

Definitions: bench

```python
import sys, time, json
from pathlib import Path
import numpy as np, torch
from gesture.model import GestureRNN
from models import LSTMClassifier
```

## hwangsoon/reports/gru_vs_lstm_0907/scripts/make_report.py

ours_results.json + jin_ckpt_scores.json + jin_on_ours_scores.json → PNG 5장 + numbers.json + 표(markdown 조각).

Definitions: agg, agg_rec, cm_sum, draw_cm, read_hist_csv, fmt

```python
import csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
```

## hwangsoon/reports/gru_vs_lstm_0907/scripts/run_ours.py

우리 GRU(+우리 구현 LSTM 대조군)를 팀원과 같은 사람 단위 분할로 채점한다.

Definitions: full_metrics, infer_ms_cpu, main

```python
import copy, json, sys, time
from pathlib import Path
import numpy as np
import torch
from gesture import baseline, config, dataset, training
```

## hwangsoon/reports/gru_vs_lstm_0907/scripts/score_jin_ckpt.py

팀원(JIN) LSTM 체크포인트를 팀원 전처리 배열(X_val/X_test .npy)로 채점한다.

Definitions: metrics, load, predict, infer_ms, score

```python
import json, sys, time
from pathlib import Path
import numpy as np
import torch
from models import LSTMClassifier
```

## hwangsoon/scripts/compare_models.py

GRU (필요하면 LSTM 도) vs baseline(룰) 공정 비교.

Definitions: run_one, fmt, summarize, main

```python
from __future__ import annotations
import argparse
import csv
import os
import sys
from pathlib import Path
import numpy as np
from gesture import config, dataset, training
from gesture.model import ARCHS, DEFAULT_ARCH
```

## hwangsoon/scripts/dataset_summary.py

dataset/ 현황: 참가자×라벨 개수, 형식 오류, 품질 기준 미달 샘플.

Definitions: main

```python
import argparse
import sys
from pathlib import Path
from gesture import config, dataset, preprocess
import numpy as np
```

## hwangsoon/scripts/eval_model.py

저장된 모델을 임의 데이터 폴더·참가자로 채점한다 (학습 없음). 룰 기반 baseline 도 같은 시험지로.

Definitions: main

```python
from __future__ import annotations
import argparse
import os
import sys
from collections import Counter
from pathlib import Path
import numpy as np
from gesture import baseline, config, dataset, training
from gesture.model import GestureClassifier, model_path
```

## hwangsoon/scripts/gesture_app.py

시연용 서비스 UI (2차): 사이드바 + 페이지(홈 / 키 매핑 / 설정 / 기록), 다크 테마. customtkinter 사용.

Definitions: font, _pil_font, open_camera, key_chips, KeyRecorder, App, main

```python
from __future__ import annotations
import argparse
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
import customtkinter as ctk
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageTk
from gesture import config
from gesture.actions import MODIFIERS, key_display, normalize_keys, press_combo
from gesture.landmarks import draw_landmarks
from gesture.presets import FUNCTIONS, GESTURE_NAMES, PRESETS, STATIC_LABELS
from gesture.pipeline import Pipeline
```

## hwangsoon/scripts/label_stats.py

라벨별 상식검사(sanity) 측정값 분포. 새 제스처의 기준값을 정하거나 기존 기준이 진짜 동작을 막는지 볼 때 쓴다.

Definitions: main

```python
from __future__ import annotations
import argparse
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from gesture import config, dataset, sanity
```

## hwangsoon/scripts/realtime_demo.py

실시간 제스처 인식 데모: 웹캠 → 손 21관절 → 구간 감지 → GRU → gestures.json 기능 실행.

Definitions: gap_runs, open_camera, classify_segment, SessionLog, main

```python
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from gesture import config, preprocess, sanity
from gesture.actions import ActionMapper
from gesture.landmarks import HandTracker, draw_landmarks
from gesture.model import GestureClassifier
from gesture.segmenter import MotionSegmenter
```

## hwangsoon/scripts/replay_segments.py

카메라 없이 실시간 루프를 검증: 녹화 npz 를 이어 붙여 가짜 스트림을 만들고

Definitions: idle_frames, main

```python
from __future__ import annotations
import argparse
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from gesture import config, dataset, preprocess, sanity
from gesture.actions import ActionMapper
from gesture.model import GestureClassifier
from gesture.segmenter import MotionSegmenter
```

## hwangsoon/scripts/train_model.py

dataset/ 의 수집 데이터로 GRU(기본) 또는 LSTM 학습 + baseline 비교 + 평가 리포트.

Definitions: main

```python
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from gesture import baseline, config, dataset, training
from gesture.model import ARCHS, DEFAULT_ARCH, model_path, save_meta
```

## kmj02/data_preprocessing.py

Gradio app for quickly reviewing and cleaning an image folder.

Definitions: natural_sort_key, ReviewSession, build_app

```python
from __future__ import annotations
import json
import re
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import gradio as gr
from PIL import Image, ImageOps
```

## kmj02/main.py



Definitions:

```python

```

## kmj02/model_yolo/yolo.py

YOLOv8 object detection training, tuning, and inference entry point.

Definitions: parse_args, load_config, resolve_local_path, resolve_model_source, build_common_args, run_train, run_tune, get_split_image_paths, read_yolo_labels, box_iou_one_to_many, binary_roc_curve, calculate_r2, make_json_safe, flatten_metrics, calculate_custom_test_metrics, run_test, resolve_inference_source, run_predict, main

```python
from __future__ import annotations
import argparse
import csv
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import yaml
from ultralytics import YOLO
```

## kmj02/realtime_detect.py

Run webcam gesture detection with ckpoint/best.pt.

Definitions: main

```python
import argparse
from pathlib import Path
import time
import cv2
import torch
from ultralytics import YOLO
```

## kmj02/slicing.py



Definitions: cap_vdieo

```python
import cv2
import os
```
