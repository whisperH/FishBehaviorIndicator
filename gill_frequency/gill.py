from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


import argparse
import io
import json
import math
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, savgol_filter, sosfiltfilt


ROOT = Path(__file__).resolve().parent
_LOCAL_VIDEO = ROOT / "datasets" / "video.mp4"
_LOCAL_ZIP = ROOT / "datasets" / "annotations_mot.zip"
_LOCAL_ANNOTATIONS = ROOT / "datasets" / "annotations_mot"
DEFAULT_VIDEO = _LOCAL_VIDEO if _LOCAL_VIDEO.exists() else Path(r"D:\鳃动.mp4")
DEFAULT_ZIP = (
    _LOCAL_ZIP
    if _LOCAL_ZIP.exists()
    else _LOCAL_ANNOTATIONS
    if _LOCAL_ANNOTATIONS.exists()
    else Path(r"D:\方形.zip")
)
BAND_LOW_HZ = 0.60
BAND_HIGH_HZ = 1.50
PEAK_PROMINENCE = 0.05
PEAK_DISTANCE_S = 0.25


def load_mot(zip_path: Path) -> pd.DataFrame:
    if zip_path.is_dir():
        frame = pd.read_csv(
            zip_path / "gt" / "gt.txt",
            header=None,
            names=["frame", "fish_id", "x", "y", "w", "h", "mark", "class", "conf"],
        )
    else:
        with zipfile.ZipFile(zip_path) as archive:
            frame = pd.read_csv(
                io.BytesIO(archive.read("gt/gt.txt")),
                header=None,
                names=["frame", "fish_id", "x", "y", "w", "h", "mark", "class", "conf"],
            )
    for column in ("frame", "fish_id", "mark", "class"):
        frame[column] = frame[column].astype(int)
    return frame


def interpolate(values: np.ndarray) -> np.ndarray:
    y = np.asarray(values, dtype=float)
    finite = np.isfinite(y)
    if finite.sum() < 2:
        return np.zeros_like(y)
    index = np.arange(len(y))
    return np.interp(index, index[finite], y[finite])


def bandpass(values: np.ndarray, fps: float) -> np.ndarray:
    y = interpolate(values)
    sos = butter(
        4,
        [BAND_LOW_HZ, BAND_HIGH_HZ],
        btype="bandpass",
        fs=fps,
        output="sos",
    )
    return sosfiltfilt(sos, y)


def robust_z(values: np.ndarray) -> np.ndarray:
    y = np.asarray(values, dtype=float)
    median = float(np.median(y))
    q25, q75 = np.percentile(y, [25, 75])
    scale = float(q75 - q25) or float(np.std(y)) or 1.0
    return (y - median) / scale


def peak_candidates(values: np.ndarray, fps: float, raw: bool) -> list[dict[str, float | int]]:
    base = interpolate(values) if raw else bandpass(values, fps)
    z = robust_z(base)
    smooth = savgol_filter(z, 11, 3)
    peaks, properties = find_peaks(
        smooth,
        distance=max(1, int(round(0.10 * fps))),
        prominence=0.001,
    )
    result: list[dict[str, float | int]] = []
    for peak, prominence in zip(peaks, properties["prominences"]):
        index = int(peak)
        if raw:
            lo = max(0, index - 6)
            hi = min(len(z), index + 7)
            index = lo + int(np.argmax(z[lo:hi]))
        result.append(
            {
                "index": index,
                "prominence": round(float(prominence), 6),
            }
        )
    return result


def selected_peaks(candidates: list[dict[str, float | int]], fps: float) -> list[dict[str, float | int]]:
    minimum = max(1, int(round(PEAK_DISTANCE_S * fps)))
    eligible = sorted(
        [item for item in candidates if item["prominence"] >= PEAK_PROMINENCE],
        key=lambda item: item["prominence"],
        reverse=True,
    )
    chosen: list[dict[str, float | int]] = []
    for item in eligible:
        if all(abs(int(other["index"]) - int(item["index"])) >= minimum for other in chosen):
            chosen.append(item)
    return sorted(chosen, key=lambda item: int(item["index"]))


def nullable(values: np.ndarray, digits: int = 3) -> list[float | None]:
    return [
        round(float(value), digits) if np.isfinite(value) else None
        for value in values
    ]


def extract_metrics(video_path: Path, tracking: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    by_frame = {
        int(frame): group.sort_values("fish_id")
        for frame, group in tracking.groupby("frame", sort=False)
    }
    rows: list[dict[str, float | int | bool]] = []
    for frame_index in range(min(frame_count, int(tracking.frame.max()))):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Video ended at frame {frame_index + 1}")
        frame_no = frame_index + 1
        tracks = by_frame.get(frame_no)
        if tracks is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for row in tracks.itertuples(index=False):
            x1 = max(0, min(width - 1, int(math.floor(row.x))))
            y1 = max(0, min(height - 1, int(math.floor(row.y))))
            x2 = max(x1 + 1, min(width, int(math.ceil(row.x + row.w))))
            y2 = max(y1 + 1, min(height, int(math.ceil(row.y + row.h))))
            roi = gray[y1:y2, x1:x2].astype(np.float64)
            visible = roi.size / max(1.0, float(row.w) * float(row.h))
            valid = bool(float(row.conf) >= 0.5 and visible >= 0.98 and roi.size)
            rows.append(
                {
                    "frame": frame_no,
                    "time_s": frame_index / fps,
                    "fish_id": int(row.fish_id),
                    "valid_tracking": valid,
                    "confidence": float(row.conf),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "gray_mean": float(roi.mean()) if valid else np.nan,
                    "gray_variance": float(roi.var()) if valid else np.nan,
                }
            )
    cap.release()
    metrics = pd.DataFrame(rows).sort_values(["fish_id", "frame"]).reset_index(drop=True)
    return metrics, {
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "fps": fps,
        "duration": frame_count / fps,
    }


def build_visualization_data(metrics: pd.DataFrame, video_info: dict[str, float | int]) -> dict[str, object]:
    fps = float(video_info["fps"])
    fish_records: list[dict[str, object]] = []
    for fish_id, group in metrics.groupby("fish_id", sort=True):
        group = group.sort_values("frame").reset_index(drop=True)
        first_frame = int(group.frame.min())
        last_frame = int(group.frame.max())
        mean = np.full(last_frame, np.nan)
        variance = np.full(last_frame, np.nan)
        valid = np.zeros(last_frame, dtype=bool)
        boxes: list[list[float] | None] = [None] * last_frame
        for row in group.itertuples(index=False):
            index = int(row.frame) - 1
            mean[index] = float(row.gray_mean)
            variance[index] = float(row.gray_variance)
            valid[index] = bool(row.valid_tracking)
            boxes[index] = [int(row.x1), int(row.y1), int(row.x2), int(row.y2), bool(row.valid_tracking)]

        filtered_mean = robust_z(bandpass(mean, fps))
        filtered_variance = robust_z(bandpass(variance, fps))
        smoothed_mean = savgol_filter(filtered_mean, 11, 3)
        smoothed_variance = savgol_filter(filtered_variance, 11, 3)
        mean_candidates = peak_candidates(mean, fps, raw=False)
        variance_candidates = peak_candidates(variance, fps, raw=False)
        mean_raw = peak_candidates(mean, fps, raw=True)
        variance_raw = peak_candidates(variance, fps, raw=True)
        mean_peaks = selected_peaks(mean_candidates, fps)
        variance_peaks = selected_peaks(variance_candidates, fps)
        mean_valid = mean[np.isfinite(mean)]
        variance_valid = variance[np.isfinite(variance)]
        fish_records.append(
            {
                "id": int(fish_id),
                "sourceTrackId": int(fish_id),
                "firstFrame": first_frame,
                "lastFrame": last_frame,
                "validFrames": int(valid.sum()),
                "baselineMean": float(np.median(mean_valid)),
                "baselineVariance": float(np.median(variance_valid)),
                "minMean": float(np.min(mean_valid)),
                "maxMean": float(np.max(mean_valid)),
                "minVariance": float(np.min(variance_valid)),
                "maxVariance": float(np.max(variance_valid)),
                "mean": nullable(mean),
                "variance": nullable(variance),
                "filteredMean": nullable(filtered_mean),
                "filteredVariance": nullable(filtered_variance),
                "smoothedMean": nullable(smoothed_mean),
                "smoothedVariance": nullable(smoothed_variance),
                "meanPeakCandidates": mean_candidates,
                "variancePeakCandidates": variance_candidates,
                "meanRawPeakCandidates": mean_raw,
                "varianceRawPeakCandidates": variance_raw,
                "meanPeakIndices": [item["index"] for item in mean_peaks],
                "variancePeakIndices": [item["index"] for item in variance_peaks],
                "meanPeakCount": len(mean_peaks),
                "variancePeakCount": len(variance_peaks),
                "boxes": boxes,
                "valid": valid.tolist(),
            }
        )
    return {
        "meta": {
            **video_info,
            "frameCount": int(video_info["frame_count"]),
            "fps": fps,
            "bandLowHz": BAND_LOW_HZ,
            "bandHighHz": BAND_HIGH_HZ,
            "regionType": "box",
            "source": str(DEFAULT_VIDEO) + " + " + str(DEFAULT_ZIP),
        },
        "fish": fish_records,
    }


def export_peak_tables(data: dict[str, object], output_dir: Path) -> pd.DataFrame:
    fps = float(data["meta"]["fps"])
    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    for fish in data["fish"]:
        duration = (int(fish["lastFrame"]) - int(fish["firstFrame"]) + 1) / fps
        curves = (
            ("raw_mean", "均值", "meanRawPeakCandidates"),
            ("raw_variance", "方差", "varianceRawPeakCandidates"),
            ("filtered_mean", "带通均值", "meanPeakCandidates"),
        )
        for curve, label, key in curves:
            peaks = selected_peaks(fish[key], fps)
            summary_rows.append(
                {
                    "fish_id": int(fish["id"]),
                    "curve": curve,
                    "metric_label": label,
                    "duration_s": duration,
                    "peak_count": len(peaks),
                    "peak_rate_per_min": len(peaks) * 60.0 / duration,
                    "peak_prominence_min": PEAK_PROMINENCE,
                    "peak_distance_min_s": PEAK_DISTANCE_S,
                }
            )
            for number, peak in enumerate(peaks, start=1):
                index = int(peak["index"])
                detail_rows.append(
                    {
                        "fish_id": int(fish["id"]),
                        "curve": curve,
                        "metric_label": label,
                        "peak_number": number,
                        "frame": index + 1,
                        "time_s": index / fps,
                        "prominence": float(peak["prominence"]),
                    }
                )
    summary = pd.DataFrame(summary_rows)
    detail = pd.DataFrame(detail_rows)
    summary.to_csv(
        output_dir / "peak_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )
    detail.to_csv(
        output_dir / "peak_details.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return summary


def decode_compressed_rle(text: str) -> list[int]:
    counts: list[int] = []
    position = 0
    while position < len(text):
        value = 0
        shift = 0
        more = 1
        while more:
            character = ord(text[position]) - 48
            value |= (character & 0x1F) << (5 * shift)
            more = character & 0x20
            position += 1
            shift += 1
            if not more and (character & 0x10):
                value |= -1 << (5 * shift)
        if counts:
            value += counts[-1]
        counts.append(value)
    return counts


def cumulative_counts(segmentation: dict) -> tuple[int, int, np.ndarray]:
    height, width = [int(value) for value in segmentation["size"]]
    counts = segmentation["counts"]
    total = height * width
    if isinstance(counts, str):
        cumulative = np.asarray(decode_compressed_rle(counts), dtype=np.int64)
    else:
        raw = np.asarray(counts, dtype=np.int64)
        is_cumulative = (
            len(raw) > 0
            and np.all(raw >= 0)
            and np.all(np.diff(raw) >= 0)
            and int(raw[-1]) == total
        )
        cumulative = raw if is_cumulative else np.cumsum(raw)
    if len(cumulative) == 0 or int(cumulative[-1]) != total:
        raise ValueError("Invalid RLE segmentation: counts do not cover the image")
    return height, width, cumulative


def segmentation_gray_stats(gray: np.ndarray, segmentation: dict) -> tuple[float, int]:
    height, width, cumulative = cumulative_counts(segmentation)
    if gray.shape[0] != height or gray.shape[1] != width:
        raise ValueError(
            f"Video frame shape {gray.shape[:2]} does not match RLE size {[height, width]}"
        )

    total = 0.0
    count = 0
    start = 0
    for run_index, end in enumerate(cumulative):
        end = int(end)
        if end < start:
            raise ValueError("Invalid RLE segmentation: negative run length")
        if run_index % 2 == 1 and end > start:
            first_row, first_column = divmod(start, width)
            last_row, last_column = divmod(end - 1, width)
            if first_row == last_row:
                values = gray[first_row, first_column : last_column + 1]
                total += float(values.sum())
                count += int(values.size)
            else:
                first = gray[first_row, first_column:]
                last = gray[last_row, : last_column + 1]
                total += float(first.sum()) + float(last.sum())
                count += int(first.size + last.size)
                if last_row > first_row + 1:
                    middle = gray[first_row + 1 : last_row, :]
                    total += float(middle.sum())
                    count += int(middle.size)
        start = end
    if count <= 0:
        raise ValueError("Segmentation mask contains no foreground pixels")
    return total / count, count


def collect_track_series(
    payload: dict,
    video_path: Path,
) -> tuple[dict[int, dict[int, float]], int, float]:
    records = payload.get("data")
    if not isinstance(records, list) or not records:
        raise ValueError("Input JSON must contain a non-empty data array")

    frame_ids = [int(record["frame_id"]) for record in records]
    zero_based = min(frame_ids) == 0
    by_video_index: dict[int, list[dict]] = {}
    for record in records:
        frame_id = int(record["frame_id"])
        video_index = frame_id if zero_based else frame_id - 1
        by_video_index.setdefault(video_index, []).append(record)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    maximum_index = max(by_video_index)
    series: dict[int, dict[int, float]] = {}

    for video_index in range(maximum_index + 1):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(
                f"Video ended before required frame {video_index + (1 if zero_based else 0)}"
            )
        frame_records = by_video_index.get(video_index)
        if not frame_records:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for record in frame_records:
            track_id = int(record["track_id"])
            frame_id = int(record["frame_id"])
            try:
                mean_value, _ = segmentation_gray_stats(gray, record["segmentation"])
            except ValueError:
                continue
            series.setdefault(track_id, {})[frame_id] = mean_value
    cap.release()
    return series, 0 if zero_based else 1, fps


def build_sliding_frequency(
    payload: dict,
    video_path: Path,
    window_seconds: float = 10.0,
    step_seconds: float = 1.0,
    min_distance_seconds: float = 0.25,
    prominence: float = 0.05,
    minimum_coverage: float = 0.80,
) -> dict:
    series, frame_offset, fps = collect_track_series(payload, video_path)
    window_frames = max(3, int(round(window_seconds * fps)))
    step_frames = max(1, int(round(step_seconds * fps)))
    output_rows: list[dict[str, object]] = []

    for track_id, values_by_frame in sorted(series.items()):
        frame_ids = sorted(values_by_frame)
        first_frame = frame_ids[0]
        last_frame = frame_ids[-1]
        array = np.full(last_frame - first_frame + 1, np.nan, dtype=float)
        present = np.zeros_like(array, dtype=bool)
        for frame_id, value in values_by_frame.items():
            index = frame_id - first_frame
            array[index] = value
            present[index] = True

        finite = np.isfinite(array)
        if finite.sum() < max(3, int(round(2.0 * fps))):
            continue
        indexes = np.arange(len(array))
        filled = np.interp(indexes, indexes[finite], array[finite])
        smooth_window = min(11, len(filled) if len(filled) % 2 == 1 else len(filled) - 1)
        sos = butter(
            4,
            [0.60, 1.50],
            btype="bandpass",
            fs=fps,
            output="sos",
        )
        filtered = sosfiltfilt(sos, filled)
        if smooth_window >= 5:
            smoothed = savgol_filter(filtered, smooth_window, 3)
        else:
            smoothed = filtered
        q25, q75 = np.percentile(smoothed, [25, 75])
        scale = float(q75 - q25) or float(np.std(smoothed)) or 1.0
        normalized = (smoothed - float(np.median(smoothed))) / scale
        peaks, _ = find_peaks(
            normalized,
            distance=max(1, int(round(min_distance_seconds * fps))),
            prominence=prominence,
        )

        for start_offset in range(0, len(array) - window_frames + 1, step_frames):
            end_offset = start_offset + window_frames - 1
            coverage = float(present[start_offset : end_offset + 1].mean())
            if coverage < minimum_coverage:
                continue
            start_frame = first_frame + start_offset
            end_frame = first_frame + end_offset
            peak_count = int(
                np.count_nonzero(
                    (peaks >= start_offset) & (peaks <= end_offset)
                )
            )
            duration = window_frames / fps
            output_rows.append(
                {
                    "track_id": int(track_id),
                    "frame_id": int(end_frame),
                    "behavior_indicator": "gill_frequency",
                    "behavior_value": round(peak_count * 60.0 / duration, 3),
                    "unit": "times/min",
                    "frame_range": [int(start_frame), int(end_frame)],
                }
            )
    return {"data": output_rows}


_HTML_BYTES = zlib.decompress(base64.b64decode(
    "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"
))
_SERVE_VIDEO = Path()
_SERVE_DATA = b"{}"


class ViewerHandler(BaseHTTPRequestHandler):
    server_version = "GillSingle/1.0"

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
        if not _SERVE_VIDEO.is_file():
            self.send_error(404, "Video not found")
            return
        size = _SERVE_VIDEO.stat().st_size
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
        with _SERVE_VIDEO.open("rb") as video:
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
            self._send_bytes(_SERVE_DATA, "application/json; charset=utf-8", head_only=True)
        elif path in ("/", "/index.html"):
            self._send_bytes(_HTML_BYTES, "text/html; charset=utf-8", head_only=True)
        else:
            self.send_error(404)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self._send_bytes(b"ok", "text/plain; charset=utf-8")
        elif path == "/video":
            self._video()
        elif path == "/data.json":
            self._send_bytes(_SERVE_DATA, "application/json; charset=utf-8")
        elif path in ("/", "/index.html"):
            self._send_bytes(_HTML_BYTES, "text/html; charset=utf-8")
        elif path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        else:
            self.send_error(404)


def analysis_command(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="gill.py analyze")
    parser.add_argument("--video", default=str(DEFAULT_VIDEO))
    parser.add_argument("--zip", default=str(DEFAULT_ZIP))
    parser.add_argument("--output", default=str(ROOT))
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    analysis_dir = output / "analysis_results"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    tracking = load_mot(Path(args.zip))
    metrics, video_info = extract_metrics(Path(args.video), tracking)
    metrics.to_csv(
        analysis_dir / "frame_gray_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    data = build_visualization_data(metrics, video_info)
    data["meta"]["source"] = f"{args.video} + {args.zip}"
    (output / "data.json").write_text(
        json.dumps(data, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    summary = export_peak_tables(data, analysis_dir)
    print(f"Fish regions: {len(data['fish'])}")
    for row in summary.itertuples(index=False):
        print(
            f"fish={row.fish_id} curve={row.curve} "
            f"peaks={row.peak_count} rate_per_min={row.peak_rate_per_min:.2f}"
        )
    print(output / "data.json")


def behavior_command(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="gill.py api")
    parser.add_argument("--input", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--step-seconds", type=float, default=1.0)
    parser.add_argument("--min-distance-seconds", type=float, default=0.25)
    parser.add_argument("--prominence", type=float, default=0.05)
    parser.add_argument("--minimum-coverage", type=float, default=0.80)
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    result = build_sliding_frequency(
        payload,
        Path(args.video),
        window_seconds=args.window_seconds,
        step_seconds=args.step_seconds,
        min_distance_seconds=args.min_distance_seconds,
        prominence=args.prominence,
        minimum_coverage=args.minimum_coverage,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Records: {len(result['data'])}")
    print(output.resolve())


def serve_command(argv: list[str]) -> None:
    global _SERVE_VIDEO, _SERVE_DATA
    parser = argparse.ArgumentParser(prog="gill.py serve")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--video", default=str(DEFAULT_VIDEO))
    parser.add_argument("--zip", default=str(DEFAULT_ZIP))
    parser.add_argument("--data", default=str(ROOT / "data.json"))
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args(argv)
    video_path = Path(args.video).resolve()
    data_path = Path(args.data).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if args.rebuild or not data_path.is_file():
        tracking = load_mot(Path(args.zip))
        metrics, video_info = extract_metrics(video_path, tracking)
        data = build_visualization_data(metrics, video_info)
        data["meta"]["source"] = f"{video_path} + {args.zip}"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_text(
            json.dumps(data, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
    _SERVE_VIDEO = video_path
    _SERVE_DATA = data_path.read_bytes()
    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    print(f"http://{args.host}:{args.port}/")
    server.serve_forever()


def main() -> None:
    commands = {
        "analyze": analysis_command,
        "api": behavior_command,
        "serve": serve_command,
    }
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print("Usage:")
        print("  python gill.py analyze")
        print("  python gill.py serve")
        print("  python gill.py api --input input.json --video video.mp4 --output output.json")
        return
    command = sys.argv[1]
    commands[command](sys.argv[2:])


if __name__ == "__main__":
    main()
