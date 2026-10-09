# Washing Machine Monitoring (RTSP)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)](https://opencv.org/)
[![FFmpeg](https://img.shields.io/badge/FFmpeg-required-red)](https://ffmpeg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 🇧🇷 *Precisava interromper meu trabalho várias vezes ao dia só para verificar se a máquina havia terminado. Para eliminar essa fricção, criei uma solução com OpenCV e FFmpeg que monitora o brilho do painel em tempo real e dispara um alarme assim que o ciclo encerra.*
>
> 🇺🇸 *I had to interrupt my work several times a day just to check if the machine had finished. To eliminate that friction, I built a solution with OpenCV and FFmpeg that monitors the panel's brightness in real time and triggers an alarm as soon as the cycle ends.*

---

## 📖 Table of Contents

- [The Problem](#-the-problem)
- [My Solution](#-my-solution)
- [How It Works / Architecture](#-how-it-works--architecture)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Configuration](#-configuration)
- [Calibrating the ROI](#-calibrating-the-roi)
- [Next Steps / Roadmap](#-next-steps--roadmap)
- [License](#-license)

---

## 🎯 The Problem

I had to interrupt my work several times a day just to check if the machine had finished. The uncertainty about when the cycle would end created unnecessary interruptions and wasted time.

I wanted a **lightweight, always-on monitor** that could tell me the instant the machine is done, without modifying the appliance itself, so I could stay focused until the cycle actually ends.

## 💡 My Solution

I built a real-time vision pipeline that watches the machine's **digital panel** through an IP camera (Yoosee / generic RTSP). When the panel lights go off, meaning the cycle ended, the system fires an audio alarm automatically.

**Why this approach:**

- **Non-invasive** — no hardware hacking, no smart plug. Just a camera I already had.
- **Robust** — brightness thresholding is simple, fast, and resilient to the kind of scene I have (fixed camera, fixed machine, controlled lighting).
- **Lightweight** — runs on any machine with Python + FFmpeg, no GPU or ML model required.

> I deliberately chose a classical CV approach over an OCR/ML solution here. The panel is a bright segmented display: counting white pixels after thresholding is faster, cheaper, and easier to tune than running a detector every frame.

---

## ⚙️ How It Works / Architecture

```
 RTSP Camera ──► FFmpeg (raw BGR pipe) ──► Grayscale + Resize (960×540)
                                                  │
                                          ┌───────▼────────┐
                                          │  ROI crop       │  ← calibrated
                                          │  Threshold(200) │     once with
                                          │  Count white px │  roi_calibrator.py
                                          └───────┬────────┘
                                                  │
                                    ┌─────────────▼──────────────┐
                                    │  Debounce: 2 consecutive   │
                                    │  OFF readings → ALARM      │
                                    └─────────────┬──────────────┘
                                                  │
                                          ┌───────▼────────┐
                                          │  AlarmPlayer    │
                                          │  pygame → afplay│  cross-
                                          │  → ffplay/mpg123│  platform
                                          └────────────────┘
```

### Why I skip 30 frames

Processing every frame at 30 fps would mean 30 threshold + count operations per second, all on the CPU, for a signal that changes at most once every 30–60 minutes. By analysing **1 frame every 30** (~1 check/second), I cut CPU usage by ~97% without missing the event. The two-reading debounce adds another layer of safety against momentary glare or dropped frames.

### Key design decisions

| Decision | Rationale |
|---|---|
| FFmpeg pipe (not `cv2.VideoCapture`) | More reliable RTSP handling, explicit TCP transport, timeout control |
| Cross-platform audio (`pygame` → subprocess fallback) | The original `afplay` loop only worked on macOS; now it runs on Linux & Windows too |
| YAML config (`config.yaml`) | No hardcoded IPs or thresholds — reproducible across environments |
| Auto-reconnect with back-off | RTSP streams drop; the monitor recovers instead of crashing |
| Debounced detection (`consecutive_off: 2`) | Eliminates false positives from reflections or single-frame glitches |

---

## 📁 Project Structure

```
.
├── monitor.py              # Main entry point — RTSP capture + detection + alarm
├── roi_calibrator.py       # Interactive tool to pick the panel ROI
├── config.example.yaml     # Tracked template — copy to config.yaml and fill in credentials
├── config.yaml             # ← git-ignored (your local config with real credentials)
├── assets/
│   ├── teste_frame.jpg     # Calibration frame grabbed from the camera
│   └── musica.mp3          # Default alarm audio — replace with any .mp3 you own
├── requirements.txt
├── .github/workflows/
│   └── lint.yml            # GitHub Actions: ruff lint on every push / PR
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **FFmpeg** installed and on your `PATH` — [download here](https://ffmpeg.org/download.html)
  ```bash
  ffmpeg -version  # should print version info
  ```

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/washing-machine-monitor.git
cd washing-machine-monitor
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate.bat     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> `pygame` is optional but recommended for cross-platform audio. Without it the app falls back to `afplay` (macOS) or `ffplay`/`mpg123` (Linux).

### 4. Configure

```bash
cp config.example.yaml config.yaml
```

Then edit `config.yaml` and set your RTSP URL at minimum:

```yaml
rtsp:
  url: "rtsp://admin:YOUR_PASSWORD@192.168.x.x:554/onvif1"
```

> `config.yaml` is **git-ignored** — your credentials stay local.

> The default alarm is `assets/musica.mp3`. You can replace it with any `.mp3` file and update `alarm.audio_path` in `config.yaml`.

If you're calibrating for the first time, see [Calibrating the ROI](#-calibrating-the-roi) below.

### 5. Run

```bash
python monitor.py                 # uses config.yaml
python monitor.py -c my.yaml      # custom config
python monitor.py --verbose       # debug logging
```

Press **`q`** to quit. The alarm stops automatically on exit.

---

## 🎯 Calibrating the ROI

I included a helper script that makes it trivial to find the right ROI for your camera angle:

```bash
python roi_calibrator.py                        # uses assets/teste_frame.jpg
python roi_calibrator.py -i path/to/frame.jpg   # custom image
```

1. A window opens showing the grayscale frame.
2. Draw a rectangle around the illuminated panel.
3. Press **ENTER** — the script prints the YAML snippet and (optionally) patches `config.yaml` for you.

> **Tip:** grab a fresh frame with `ffmpeg -i <rtsp_url> -frames:v 1 assets/teste_frame.jpg -y` for the most accurate calibration.

---

## 🔮 Next Steps / Roadmap

I see this project as a foundation. Here's what I plan to add next:

- [ ] **Push notifications** — send a WhatsApp/Telegram message when the cycle ends (via [Z-API](https://z-api.io/), Twilio, or ntfy.sh) so I don't need to be near the speaker.
- [ ] **Web dashboard** — a tiny FastAPI + WebSocket page that shows the live feed and panel status.
- [ ] **Docker image** — one-command deployment on a Raspberry Pi or NAS.
- [ ] **Adaptive threshold** — auto-tune `threshold`/`pixel_limit` based on ambient light (histogram analysis).
- [ ] **Cycle history** — log start/end timestamps to a local SQLite DB and expose wash-duration stats.

Contributions and ideas are welcome, feel free to open an issue or PR!

---

## 📄 License

MIT — do whatever you want, just keep the notice.

---

<p align="center"><i>Built with 3 AM hyperfocus for a house too big.</i></p>
