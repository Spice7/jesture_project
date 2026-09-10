from pathlib import Path

from PIL import Image, ImageOps


# 원본 이미지 폴더
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "two_fingers"

# 좌우 반전한 이미지를 저장할 폴더
OUTPUT_DIR = PROJECT_ROOT / "images" / "flipped"

# 지원할 이미지 확장자
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


    for image_path in INPUT_DIR.iterdir():

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        try:
            with Image.open(image_path) as image:

                # 스마트폰 사진의 EXIF 회전 정보 적용
                image = ImageOps.exif_transpose(image)

                # 좌우 반전
                flipped_image = image.transpose(
                    Image.Transpose.FLIP_LEFT_RIGHT
                )

                # 저장 경로
                output_path = OUTPUT_DIR / image_path.name

                flipped_image.save(output_path)

                print(f"[완료] {image_path.name}")

        except Exception as e:
            print(f"[실패] {image_path.name}: {e}")


    print("모든 이미지 좌우 반전 완료")


if __name__ == "__main__":
    main()
