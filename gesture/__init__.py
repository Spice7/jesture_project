"""동적 제스처(LSTM) 파이프라인 공용 패키지.

녹화(record) → 전처리(preprocess) → 데이터셋(dataset) → 모델(model) 순서로 사용한다.
팀 규약은 gesture/config.py 와 docs/team_conventions.md 를 따른다.
"""

# Windows 콘솔(cp949)에서 한글·±·→ 같은 문자를 print 할 때 UnicodeEncodeError 로 죽지 않게 한다.
# Windows Terminal/PowerShell 은 UTF-8 을 그대로 표시하고, 구형 cmd 는 깨진 글자로 보이되 멈추지는 않는다.
import sys as _sys

if _sys.platform == "win32":
    for _s in (_sys.stdout, _sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
del _sys
