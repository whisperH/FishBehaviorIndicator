from __future__ import annotations

"""鳃动频率可视化服务。

只负责提供 gill_viewer.html、视频流和 gill_data.json；分析计算由
gill_frequency.py 完成。
"""

import argparse
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


VISUALIZATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VISUALIZATION_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gill_frequency as analysis


DEFAULT_HTML = VISUALIZATION_DIR / "gill_viewer.html"
DEFAULT_DATA = (
    PROJECT_ROOT
    / analysis.RESULTS_DIRNAME
    / analysis.GILL_RESULTS_DIRNAME
    / analysis.GILL_DATA_FILENAME
)

_VIDEO_PATH = Path()
_DATA_BYTES = b"{}"
_HTML_BYTES = b""


class GillViewerHandler(BaseHTTPRequestHandler):
    server_version = "GillViewer/1.0"

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
            self.send_error(404, "Video not found")
            return
        size = _VIDEO_PATH.stat().st_size
        start = 0
        end = size - 1
        status = 200
        range_header = self.headers.get("Range", "")
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if not match:
                self.send_error(416, "Invalid range")
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

    def do_HEAD(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/video":
            self._video(head_only=True)
        elif path == "/data.json":
            self._send_bytes(
                _DATA_BYTES, "application/json; charset=utf-8", head_only=True
            )
        elif path in ("/", "/index.html"):
            self._send_bytes(
                _HTML_BYTES, "text/html; charset=utf-8", head_only=True
            )
        else:
            self.send_error(404)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self._send_bytes(b"ok", "text/plain; charset=utf-8")
        elif path == "/video":
            self._video()
        elif path == "/data.json":
            self._send_bytes(_DATA_BYTES, "application/json; charset=utf-8")
        elif path in ("/", "/index.html"):
            self._send_bytes(_HTML_BYTES, "text/html; charset=utf-8")
        elif path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        else:
            self.send_error(404)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gill_viewer.py",
        description="Serve the gill-frequency visualization.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--video", default=str(analysis.DEFAULT_VIDEO))
    parser.add_argument("--zip", default=str(analysis.DEFAULT_ZIP))
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--html", default=str(DEFAULT_HTML))
    parser.add_argument("--rebuild", action="store_true")
    return parser


def main() -> None:
    global _VIDEO_PATH, _DATA_BYTES, _HTML_BYTES
    args = build_parser().parse_args()
    video_path = Path(args.video).resolve()
    data_path = Path(args.data).resolve()
    html_path = Path(args.html).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not html_path.is_file():
        raise FileNotFoundError(f"Viewer HTML not found: {html_path}")
    if args.rebuild or not data_path.is_file():
        analysis.run_analysis(video_path, Path(args.zip), data_path.parent)

    _VIDEO_PATH = video_path
    _DATA_BYTES = data_path.read_bytes()
    _HTML_BYTES = html_path.read_bytes()
    server = ThreadingHTTPServer((args.host, args.port), GillViewerHandler)
    print(f"http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
