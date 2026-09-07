"""사용자가 승인한 번호 변경/중복 제외를 학습용 복사본에만 적용하는 일회성 스크립트."""

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "dataset"
DESTINATION = Path(__file__).resolve().parent / "input"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def same_array(a, b):
    return a.dtype == b.dtype and np.array_equal(a, b, equal_nan=True) if a.dtype.kind in "fc" and b.dtype.kind in "fc" else a.dtype == b.dtype and np.array_equal(a, b)


def main():
    if DESTINATION.exists():
        raise FileExistsError("기존 학습용 복사본은 덮어쓰지 않습니다.")
    files = sorted(SOURCE.rglob("*.npz"))
    index = SOURCE / "index.csv"
    protected = files + ([index] if index.is_file() else [])
    before = {p.relative_to(ROOT).as_posix(): sha256(p) for p in protected}
    # 기존 123개 뒤에 정확히 37개를 연결할 수 있는지 먼저 확인합니다.
    original_ids = []
    pilot_ids = []
    for path in files:
        data = read_npz(path)
        if data["participant_id"].item() == "p001" and data["label"].item() == "swipe_left":
            original_ids.append(data["sample_id"].item())
        if data["participant_id"].item() == "user00":
            if data["label"].item() != "swipe_left":
                raise ValueError("예상하지 못한 user00 라벨이 추가되었습니다.")
            pilot_ids.append(data["sample_id"].item())
    if sorted(original_ids) != list(range(1, 124)) or sorted(pilot_ids) != list(range(1, 38)):
        raise ValueError("원본 번호 구성이 승인 당시와 다릅니다. 재확인이 필요합니다.")
    skipped = {
        "make_fist/p004_make_fist_0033.npz": "make_fist/p004_make_fist_0001.npz",
        "make_fist/p004_make_fist_0034.npz": "make_fist/p004_make_fist_0002.npz",
    }
    for duplicate, retained in skipped.items():
        left, right = read_npz(SOURCE / duplicate), read_npz(SOURCE / retained)
        if set(left) != set(right) or any(not same_array(left[k], right[k]) for k in left if k != "sample_id"):
            raise ValueError(f"동일 자료가 아닙니다: {duplicate}")

    DESTINATION.mkdir()
    records = []
    for path in files:
        relative = path.relative_to(SOURCE).as_posix()
        row = {"source_path": relative, "source_sha256": before[path.relative_to(ROOT).as_posix()]}
        if relative in skipped:
            records.append({**row, "action": "excluded_identical_copy", "retained_source": skipped[relative]})
            continue
        original = read_npz(path)
        renamed = original["participant_id"].item() == "user00"
        if renamed:
            converted = dict(original)
            new_id = 123 + int(original["sample_id"].item())
            converted["participant_id"] = np.array("p001")
            converted["sample_id"] = np.array(new_id, dtype=original["sample_id"].dtype)
            target = DESTINATION / "swipe_left" / f"p001_swipe_left_{new_id:04d}.npz"
        else:
            converted = original
            target = DESTINATION / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f"대상 충돌: {target}")
        if renamed:
            np.savez_compressed(target, **converted)
        else:
            shutil.copy2(path, target)
        saved = read_npz(target)
        if set(saved) != set(converted) or any(not same_array(saved[k], converted[k]) for k in saved):
            raise ValueError(f"저장 검증 실패: {target}")
        if any(not same_array(saved[k], original[k]) for k in original if k not in ("participant_id", "sample_id")):
            raise ValueError(f"의도하지 않은 데이터 변경: {target}")
        records.append({**row, "action": "renumbered_copy" if renamed else "unchanged_copy",
                        "target_path": target.relative_to(DESTINATION).as_posix(), "target_sha256": sha256(target),
                        "original_participant_id": original["participant_id"].item(),
                        "original_sample_id": int(original["sample_id"].item()),
                        "participant_id": converted["participant_id"].item(),
                        "sample_id": int(converted["sample_id"].item())})
    after = {p.relative_to(ROOT).as_posix(): sha256(p) for p in protected}
    if before != after or sorted(SOURCE.rglob("*.npz")) != files:
        raise ValueError("실행 중 원본 목록/내용이 변경되었습니다.")
    result = {"source_dir": str(SOURCE), "input_dir": str(DESTINATION), "source_files": len(files),
              "output_files": len(files) - len(skipped), "renumbered_files": len(pilot_ids),
              "excluded_identical_copies": len(skipped), "originals_unchanged": before == after,
              "protected_source_sha256": before, "records": records}
    with (DESTINATION.parent / "input_provenance.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k not in ("records", "protected_source_sha256")}, ensure_ascii=True))


if __name__ == "__main__":
    main()
