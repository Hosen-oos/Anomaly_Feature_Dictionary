import sys
from pathlib import Path

# 让 `import mlusd` 和 `import tests.synthetic` 都能解析（包根 = 本文件的父目录）
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
