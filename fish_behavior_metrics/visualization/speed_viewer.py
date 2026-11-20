from __future__ import annotations

"""鱼体几何中心速度可视化服务。

只负责提供 speed_viewer.html、视频流、速度结果和掩膜轮廓；速度计算由
speed_analysis.py 完成。
"""

import argparse
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


VISUALIZATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VISUALIZATION_DIR.parent

_HTML_PATH = Path()
_VIDEO_PATH = Path()
_MASK_PATH = Path()
_POSTER_PATH = Path()
_RESULTS_DIR = Path()


class SpeedViewerHandler(BaseHTTPRequestHandler):
    server_version = "SpeedViewer/1.0"

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

    def _send_file(
        self,
        path: Path,
        content_type: str,
        head_only: bool = False,
    ) -> None:
        if not path.is_file():
            self.send_error(404)
            return
        if path.suffix.lower() != ".mp4":
            self._send_bytes(path.read_bytes(), content_type, head_only=head_only)
            return

        size = path.stat().st_size
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
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        if head_only:
            return
        with path.open("rb") as video:
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
            return
        if path == "/video":
            self._send_file(_VIDEO_PATH, "video/mp4", head_only=head_only)
            return
        if path == "/poster":
            self._send_file(_POSTER_PATH, "image/jpeg", head_only=head_only)
            return
        if path in ("/", "/speed_viewer.html"):
            self._send_file(_HTML_PATH, "text/html; charset=utf-8", head_only=head_only)
            return
        if path == "/mask-shapes":
            self._send_file(
                _MASK_PATH,
                "application/json; charset=utf-8",
                head_only=head_only,
            )
            return
        if path.startswith("/results/speed/"):
            relative = path.removeprefix("/results/speed/")
            if not relative or ".." in Path(relative).parts:
                self.send_error(404)
                return
            target = (_RESULTS_DIR / relative).resolve()
            if not target.is_relative_to(_RESULTS_DIR):
                self.send_error(404)
                return
            content_type = (
                mimetypes.guess_type(target.name)[0]
                or "application/octet-stream"
            )
            self._send_file(target, content_type, head_only=head_only)
            return
        if path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        self.send_error(404)

    def do_HEAD(self) -> None:
        self._dispatch(head_only=True)

    def do_GET(self) -> None:
        self._dispatch(head_only=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="speed_viewer.py",
        description="Serve the fish center-speed visualization.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8772)
    parser.add_argument(
        "--html",
        default=str(VISUALIZATION_DIR / "speed_viewer.html"),
    )
    parser.add_argument(
        "--video",
        default=str(PROJECT_ROOT / "datasets" / "angle.mp4"),
    )
    parser.add_argument(
        "--results",
        default=str(PROJECT_ROOT / "results" / "speed"),
    )
    parser.add_argument(
        "--mask-shapes",
        default=str(PROJECT_ROOT / "results" / "angle" / "mask_shapes.json"),
    )
    parser.add_argument(
        "--poster",
        default=str(PROJECT_ROOT / "results" / "angle" / "video_poster.jpg"),
    )
    return parser


def main() -> None:
    global _HTML_PATH, _VIDEO_PATH, _MASK_PATH, _POSTER_PATH, _RESULTS_DIR
    args = build_parser().parse_args()
    _HTML_PATH = Path(args.html).resolve()
    _VIDEO_PATH = Path(args.video).resolve()
    _MASK_PATH = Path(args.mask_shapes).resolve()
    _POSTER_PATH = Path(args.poster).resolve()
    _RESULTS_DIR = Path(args.results).resolve()
    if not _HTML_PATH.is_file():
        raise FileNotFoundError(f"Viewer HTML not found: {_HTML_PATH}")
    if not _VIDEO_PATH.is_file():
        raise FileNotFoundError(f"Video not found: {_VIDEO_PATH}")
    if not _MASK_PATH.is_file():
        raise FileNotFoundError(f"Mask shapes not found: {_MASK_PATH}")
    if not _RESULTS_DIR.is_dir():
        raise FileNotFoundError(f"Speed results not found: {_RESULTS_DIR}")

    server = ThreadingHTTPServer((args.host, args.port), SpeedViewerHandler)
    print(f"http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
