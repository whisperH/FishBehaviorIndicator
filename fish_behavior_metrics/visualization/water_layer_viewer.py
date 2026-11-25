from __future__ import annotations

"""鱼体水层分布可视化服务。

只负责提供 water_layer_viewer.html 和视频流；水层计算由
water_layer_analysis.py 完成。
"""

import argparse
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


VISUALIZATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VISUALIZATION_DIR.parent
_HTML_PATH = Path()
_VIDEO_PATH = Path()


class WaterLayerViewerHandler(BaseHTTPRequestHandler):
    server_version = "WaterLayerViewer/1.0"

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_bytes(
        self,
        data: bytes,
        content_type: str,
        status: int = 200,
        headers: dict[str, str] | None = None,
        head_only: bool = False,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if not head_only:
            self.wfile.write(data)

    def _video(self, head_only: bool = False) -> None:
        if not _VIDEO_PATH.is_file():
            self.send_error(404)
            return
        size = _VIDEO_PATH.stat().st_size
        start = 0
        end = size - 1
        status = 200
        range_header = self.headers.get("Range", "")
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if not match:
                self.send_error(416)
                return
            start_text, end_text = match.groups()
            if start_text:
                start = int(start_text)
                end = int(end_text) if end_text else size - 1
            elif end_text:
                start = max(0, size - int(end_text))
            if start >= size or start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            end = min(end, size - 1)
            status = 206
        length = end - start + 1
        headers = {"Accept-Ranges": "bytes"}
        if status == 206:
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(length))
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        if head_only:
            return
        with _VIDEO_PATH.open("rb") as video:
            video.seek(start)
            remaining = length
            while remaining > 0:
                chunk = video.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    break
                remaining -= len(chunk)

    def _dispatch(self, head_only: bool) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self._send_bytes(b"ok", "text/plain; charset=utf-8", head_only=head_only)
        elif path == "/video":
            self._video(head_only=head_only)
        elif path in ("/", "/water_layer_viewer.html"):
            self._send_bytes(
                _HTML_PATH.read_bytes(),
                "text/html; charset=utf-8",
                head_only=head_only,
            )
        elif path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        else:
            self.send_error(404)

    def do_HEAD(self) -> None:
        self._dispatch(head_only=True)

    def do_GET(self) -> None:
        self._dispatch(head_only=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="water_layer_viewer.py",
        description="Serve the fish water-layer visualization.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8771)
    parser.add_argument(
        "--html",
        default=str(VISUALIZATION_DIR / "water_layer_viewer.html"),
    )
    parser.add_argument(
        "--video",
        default=str(PROJECT_ROOT / "datasets" / "angle.mp4"),
    )
    return parser


def main() -> None:
    global _HTML_PATH, _VIDEO_PATH
    args = build_parser().parse_args()
    _HTML_PATH = Path(args.html).resolve()
    _VIDEO_PATH = Path(args.video).resolve()
    if not _HTML_PATH.is_file():
        raise FileNotFoundError(f"Viewer HTML not found: {_HTML_PATH}")
    if not _VIDEO_PATH.is_file():
        raise FileNotFoundError(f"Video not found: {_VIDEO_PATH}")
    server = ThreadingHTTPServer((args.host, args.port), WaterLayerViewerHandler)
    print(f"http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
