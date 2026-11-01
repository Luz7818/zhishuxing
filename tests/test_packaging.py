"""便携包构建编排的最小回归(MAINT-01)。

package_zip 的默认构建分支以子进程调用打包脚本;脚本路径曾漂移到已删除的
scripts/build_exe.py,导致默认命令 100% 失败。这里钉住:引用的构建脚本必须
真实存在且就是 packaging/ 下的 build_exe.py。
"""

from __future__ import annotations

import sys
from pathlib import Path

_PACKAGING_DIR = Path(__file__).resolve().parents[1] / "packaging"
sys.path.insert(0, str(_PACKAGING_DIR))

import package_zip  # noqa: E402


def test_build_script_exists():
    assert package_zip.BUILD_SCRIPT == _PACKAGING_DIR / "build_exe.py"
    assert package_zip.BUILD_SCRIPT.exists(), f"构建脚本缺失:{package_zip.BUILD_SCRIPT}"


def test_packaging_dir_self_hosted():
    """package_zip 自身也在 packaging/ 下,脚本-脚本引用不允许再指向别的顶层目录。"""
    assert package_zip.ROOT == _PACKAGING_DIR.parent
    assert package_zip.BUILD_SCRIPT.parent == package_zip.ROOT / "packaging"
