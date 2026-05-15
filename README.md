# VisionStick

VisionStick is a real-time computer vision navigation assistant for visually impaired users. It detects obstacles from a webcam or video file, estimates relative depth, tracks objects over time, ranks the most relevant danger, and gives short audio warnings.

The system uses two YOLOv8 models together:

- `yolov8s.pt` for selected COCO classes such as people, vehicles, chairs, tables, and traffic objects.
- `models/best.pt` for the custom navigation classes: `Door`, `Tree`, and `Stairs`.

Both model outputs are merged before depth estimation, tracking, risk scoring, and audio alerting.

## Main Features

- Dual YOLOv8 detection: COCO YOLOv8s plus custom Door/Tree/Stairs detector.
- ByteTrack multi-object tracking for stable object identities.
- Depth Anything V2 for monocular relative depth estimation.
- Risk scoring based on depth-heavy closeness, walking-path relevance, class hazard, and confidence.
- One primary obstacle is selected at a time to avoid alert overload.
- Windows text-to-speech alerts using SAPI through PowerShell.
- Webcam and video-file input support.

## Project Structure

```text
My_CV_Project/
  src/
    visionstick/
      run.py          # CLI entry point
      pipeline.py     # model loading, video loop, rendering
      core.py         # detection, tracking, depth, risk, TTS
      config.py       # model paths, classes, thresholds, weights
      __init__.py

  models/
    best.pt           # custom Door/Tree/Stairs model, not tracked by Git

  data/
    videos/           # local test videos, not tracked by Git

  README.md
  .gitignore
```

## Requirements

Recommended:

- Windows
- Python 3.11
- NVIDIA GPU with CUDA support for best performance
- Webcam or test video file

Python packages:

- `torch`
- `torchvision`
- `torchaudio`
- `ultralytics`
- `transformers`
- `opencv-python`
- `numpy`
- `pillow`

## Setup

Open CMD or PowerShell in the project folder:

```cmd
cd C:\Users\user\Desktop\MyFiles\Uni\My_CV_Project
```

Create and activate a virtual environment:

```cmd
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
```

Install GPU PyTorch:

```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Install the remaining packages:

```cmd
pip install ultralytics transformers opencv-python pillow numpy
```

Check that CUDA is available:

```cmd
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

If it prints `True`, the app can use the GPU.

## Model Files

Model weights are intentionally ignored by Git because they are large.

Place the custom model here:

```text
models/best.pt
```

The COCO model can be placed in the project root:

```text
yolov8s.pt
```

If `yolov8s.pt` is missing, Ultralytics may download it automatically the first time it is used.

## Run The App

Activate the venv first:

```cmd
.venv\Scripts\activate.bat
```

### Run With Webcam

Default webcam:

```cmd
python src\visionstick\run.py
```

Specific webcam index:

```cmd
python src\visionstick\run.py --source 1
```

### Run On A Video

Create a video folder:

```cmd
mkdir data\videos
```

Place your video here:

```text
data\videos\test.mp4
```

Run:

```cmd
python src\visionstick\run.py --source data\videos\test.mp4
```

Recommended balanced test command:

```cmd
python src\visionstick\run.py --source data\videos\test.mp4 --conf 0.50 --depth-skip 3
```

### Run With Explicit Models

```cmd
python src\visionstick\run.py --source data\videos\test.mp4 --yolo yolov8s.pt --custom-yolo models\best.pt
```

### Disable The Custom Model

Use this to compare against COCO YOLO only:

```cmd
python src\visionstick\run.py --source data\videos\test.mp4 --no-custom-yolo
```

## Command-Line Options

```text
--source          Webcam index or video path. Default: 0
--yolo            COCO YOLO weights path. Default: yolov8s.pt
--custom-yolo     Custom Door/Tree/Stairs weights path. Default: models/best.pt
--no-custom-yolo  Disable custom model
--conf            YOLO confidence threshold. Default: 0.45
--depth-skip      Run depth every N frames. Default: 5
```

## Keyboard Controls

Inside the OpenCV window:

```text
q      quit
ESC    quit
p      pause / resume
```

## How Risk Scoring Works

The app does not simply choose the largest bounding box. That would over-prioritize naturally large objects such as cars, trees, doors, or stairs.

Instead, each detection receives a risk score based on:

- Depth-heavy closeness
- Walking-path overlap
- Class hazard
- Detection confidence

Current formula:

```text
risk = 0.45 * closeness + 0.30 * path_score + 0.15 * class_hazard + 0.10 * confidence
```

The highest-risk object above the minimum threshold becomes the primary obstacle. Only the primary obstacle is used for spoken alerts.

## Notes For GitHub

The following are ignored by `.gitignore`:

- `.venv/`
- YOLO weights such as `*.pt`
- exported model files such as `*.onnx`, `*.engine`, `*.tflite`
- local videos in `data/videos/`
- generated YOLO/Ultralytics output folders

After cloning the repository, restore the missing local files manually:

```text
models/best.pt
yolov8s.pt
data/videos/your_test_video.mp4
```

## Limitations

- Depth is relative, not exact metric distance.
- Low light and motion blur can reduce detection quality.
- Hidden or fully occluded objects cannot be detected.
- Running two YOLO models plus depth estimation needs a capable GPU for smooth performance.
- A larger labeled video test set would improve system-level evaluation.

## Authors

- Naji Bou Zeid
- Mathieu Moussa
- Michel Abou Rahal

Instructor: Dr. Ahmad Audi
