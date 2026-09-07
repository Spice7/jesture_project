# 파일 목록
## slicing.py
- 영상을 여러장의 frames로 나누는 기능
## EDA.ipynb
- 데이터의 EDA 진행
## data_preprocessing.py
- 수집한 데이터 중 사용하지 않을 사진 filtering
## model_yolo
### yolo.py
- 모델 코드
    - test mode
        - 실행
        ```
        uv run python model_yolo/yolo.py --mode test
        ```
    - trian mode
        - 실행
        ```
        uv run python model_yolo/yolo.py --mode train
        ```
### yolo_param.yaml
- 모델 하이퍼파라미터
    - 파라미터 튜닝용
- 탐지 임계값 설정 가능
# 데이터 구조
```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```