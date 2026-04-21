from __future__ import annotations

import argparse
from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from webapp.app import app


def main() -> None:
    parser = argparse.ArgumentParser("智枢星 Web 服务")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--production", action="store_true", help="使用 waitress 生产部署")
    args = parser.parse_args()

    if args.production:
        from waitress import serve

        serve(app, host=args.host, port=args.port)
        return

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
