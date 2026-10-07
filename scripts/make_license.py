"""厂商侧授权签发工具:python scripts/make_license.py --customer 客户名称 --days 365 --out license.lic

只应运行在厂商环境:签发密钥在 zhishuxing/licensing.py 中,Docker 镜像与交付包都
不包含本脚本。days<=0 表示永久授权。生成后把文件交付给客户,客户侧执行
`zhishuxing license --file license.lic` 激活,或直接放到 workspace 根目录。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from zhishuxing import licensing  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="智枢星授权签发（厂商侧专用）")
    parser.add_argument("--customer", required=True, help="客户名称（写进授权并展示在界面上）")
    parser.add_argument("--days", type=int, default=365, help="有效天数；<=0 表示永久授权")
    parser.add_argument("--out", type=Path, default=Path("license.lic"), help="输出文件路径")
    args = parser.parse_args()

    record = licensing.write_license(customer=args.customer, days=args.days, out=args.out)
    print(f"已签发: {args.out}")
    print(json.dumps({k: v for k, v in record.items() if k != "signature"}, ensure_ascii=False, indent=2))
    print("交付方式: 发给客户后由客户执行 zhishuxing license --file <文件>,或直接放到其 workspace 根目录。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
