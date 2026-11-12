# Gill Beat Frequency

Single-file analysis and visualization tool for gill movement.

## Included data

- `datasets/video.mp4`
- `datasets/annotations_mot/gt/gt.txt`
- `datasets/annotations_mot/gt/labels.txt`

The extracted MOT directory is supported directly. A ZIP containing `gt/gt.txt` is also supported.

## Install dependencies

```powershell
python -m pip install -r requirements.txt
```

## Analyze

```powershell
python gill.py analyze
```

Outputs:

- `data.json`
- `analysis_results/frame_gray_metrics.csv`
- `analysis_results/peak_counts.csv`
- `analysis_results/peak_details.csv`

## Start visualization

```powershell
python gill.py serve
```

Open `http://127.0.0.1:8768/`.

If `data.json` is missing, the server automatically calculates it first:

```powershell
python gill.py serve --rebuild
```

## Sliding-window behavior API

```powershell
python gill.py api `
  --input input.json `
  --video "datasets/video.mp4" `
  --output output.json `
  --window-seconds 10 `
  --step-seconds 1
```

The output indicator is `gill_frequency` with unit `times/min` and a `frame_range` for each window. Peaks are detected on the `0.60–1.50 Hz` band-pass residual of the mask gray-mean signal.

## Requirements

- Windows, Linux, or macOS
- Python 3.12 64-bit
- Internet access for the first dependency installation

