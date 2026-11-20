#!/usr/bin/env python3 -I
"""项目根入口：以隔离模式运行申诉分案流水线。

-I 保证不从数据目录/当前目录加载第三方或同名模块；
这里显式把 src 加入 sys.path，隔离模式下 PYTHONPATH 会被忽略。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from appeal.cli import main  # noqa: E402

if __name__ == '__main__':
    raise SystemExit(main())
